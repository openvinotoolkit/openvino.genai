// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <ctime>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <cmath>
#include <filesystem>

#include "openvino/runtime/core.hpp"
#include "openvino_extensions/strings.hpp"

#include "cxxopts.hpp"
#include "scheduler_lcm.hpp"
#include "lora_cpp.hpp"
#include "imwrite.hpp"

ov::Tensor randn_tensor(uint32_t height, uint32_t width, bool use_np_latents, uint32_t seed = 42) {
    ov::Tensor noise(ov::element::f32, {1, 4, height / 8, width / 8});
    if (use_np_latents) {
        // read np generated latents with defaut seed 42
        const char * latent_file_name = "../scripts/np_latents_512x512.txt";
        std::ifstream latent_copy_file(latent_file_name, std::ios::ate);
        OPENVINO_ASSERT(latent_copy_file.is_open(), "Cannot open ", latent_file_name);

        size_t file_size = latent_copy_file.tellg() / sizeof(float);
        OPENVINO_ASSERT(file_size >= noise.get_size(), "Cannot generate ", noise.get_shape(), " with ", latent_file_name, ". File size is small");

        latent_copy_file.seekg(0, std::ios::beg);
        for (size_t i = 0; i < noise.get_size(); ++i)
            latent_copy_file >> noise.data<float>()[i];
    } else {
        std::mt19937 gen{static_cast<unsigned long>(seed)};
        std::normal_distribution<float> normal{0.0f, 1.0f};
        std::generate_n(noise.data<float>(), noise.get_size(), [&]() {
            return normal(gen);
        });
    }
    return noise;
}

struct StableDiffusionModels {
    ov::CompiledModel text_encoder;
    ov::CompiledModel unet;
    ov::CompiledModel vae_decoder;
    ov::CompiledModel tokenizer;
};

StableDiffusionModels compile_models(const std::string& model_path, const std::string& device,
                                     const std::string& type, const std::string& lora_path,
                                     const float alpha, const bool use_cache) {
    StableDiffusionModels models;

    ov::Core core;
    if (use_cache)
        core.set_property(ov::cache_dir("./cache_dir"));
    core.add_extension(TOKENIZERS_LIBRARY_PATH);

    std::map<std::string, float> lora_models;
    lora_models[lora_path] = alpha;

    // CLIP and UNet
    auto text_encoder_model = core.read_model((model_path + "/" + type + "/text_encoder/openvino_model.xml").c_str());
    auto unet_model = core.read_model((model_path + "/" + type + "/unet/openvino_model.xml").c_str());
    auto text_encoder_and_unet = load_lora_weights_cpp(core, text_encoder_model, unet_model, device, lora_models);
    models.text_encoder = text_encoder_and_unet[0];
    models.unet = text_encoder_and_unet[1];

    // VAE decoder
    auto vae_decoder_model = core.read_model((model_path + "/" + type + "/vae_decoder/openvino_model.xml").c_str());
    ov::preprocess::PrePostProcessor ppp(vae_decoder_model);
    ppp.output().model().set_layout("NCHW");
    ppp.output().tensor().set_layout("NHWC");
    models.vae_decoder = core.compile_model(vae_decoder_model = ppp.build(), device);

    // Tokenizer
    std::string tokenizer_model_path = "../models/tokenizer/tokenizer_encoder.xml";
    models.tokenizer = core.compile_model(tokenizer_model_path, device);

    return models;
}

ov::Tensor text_encoder(StableDiffusionModels models, const std::string& pos_prompt) {
    const size_t MAX_LENGTH = 77 /* 'model_max_length' from 'tokenizer_config.json' */, BATCH_SIZE = 1;
    const int32_t EOS_TOKEN_ID = 49407, PAD_TOKEN_ID = EOS_TOKEN_ID;
    const ov::Shape input_ids_shape({1, MAX_LENGTH});

    ov::InferRequest text_encoder_req = models.text_encoder.create_infer_request();
    ov::Tensor input_ids = text_encoder_req.get_input_tensor();
    
    input_ids.set_shape(input_ids_shape);
    // need to pre-fill 'input_ids' with 'PAD' tokens
    std::int32_t* input_ids_data = input_ids.data<int32_t>();
    std::fill_n(input_ids_data, input_ids.get_size(), PAD_TOKEN_ID);

    // Tokenization

    ov::InferRequest tokenizer_req = models.tokenizer.create_infer_request();
    ov::Tensor packed_strings = tokenizer_req.get_input_tensor();

    openvino_extensions::pack_strings(std::array<std::string, BATCH_SIZE>{pos_prompt}, packed_strings);
    tokenizer_req.infer();
    ov::Tensor input_ids_pos = tokenizer_req.get_tensor("input_ids");
    std::copy_n(input_ids_pos.data<std::int32_t>(), input_ids_pos.get_size(), input_ids_data);

    // Text embedding
    text_encoder_req.infer();

    return text_encoder_req.get_output_tensor(0);
}

ov::Tensor get_w_embedding(float guidance_scale = 7.5, uint32_t embedding_dim = 512) {
    float w = guidance_scale * 1000;
    uint32_t half_dim = embedding_dim / 2;

    float emb = log(10000) / (half_dim - 1);

    std::vector<float> emb_vec(half_dim);
    std::iota(emb_vec.begin(), emb_vec.end(), 0);

    std::transform(emb_vec.begin(), emb_vec.end(), emb_vec.begin(),
                    std::bind(std::multiplies<float>(), std::placeholders::_1, -emb));

    std::transform(emb_vec.begin(), emb_vec.end(), emb_vec.begin(), [](float x){return std::exp(x);});

    std::transform(emb_vec.begin(), emb_vec.end(), emb_vec.begin(),
                    std::bind(std::multiplies<float>(), std::placeholders::_1, w));

    std::vector<float> res_vec(half_dim), emb_cos(half_dim);
    std::transform(emb_vec.begin(), emb_vec.end(), res_vec.begin(), [](float x){return std::sin(x);});
    std::transform(emb_vec.begin(), emb_vec.end(), emb_cos.begin(), [](float x){return std::cos(x);});
    res_vec.insert(res_vec.end(), emb_cos.begin(), emb_cos.end());

    if (embedding_dim % 2 == 1)
        res_vec.insert(res_vec.end(), 0);

    assert(res_vec.size() == embedding_dim);

    ov::Shape embedding_shape = {1, embedding_dim};
    ov::Tensor w_embedding(ov::element::f32, embedding_shape);
    for (size_t i = 0; i < ov::shape_size(embedding_shape); ++i)
        w_embedding.data<float>()[i] = res_vec[i];

    return w_embedding;
}

ov::Tensor unet(ov::InferRequest req, ov::Tensor sample, ov::Tensor timestep, ov::Tensor text_embedding_1d, ov::Tensor guidance_scale_embedding) {
    req.set_tensor("sample", sample);
    req.set_tensor("timestep", timestep);
    req.set_tensor("encoder_hidden_states", text_embedding_1d);
    req.set_tensor("timestep_cond", guidance_scale_embedding);

    req.infer();

    return req.get_output_tensor();
}

ov::Tensor vae_decoder(ov::CompiledModel& decoder_compiled_model, ov::Tensor sample) {
    const float coeffs_const{1 / 0.18215};
    for (size_t i = 0; i < sample.get_size(); ++i)
        sample.data<float>()[i] *= coeffs_const;

    ov::InferRequest req = decoder_compiled_model.create_infer_request();
    req.set_input_tensor(sample);
    req.infer();

    return req.get_output_tensor();
}

ov::Tensor postprocess_image(ov::Tensor decoded_image) {
    ov::Tensor generated_image(ov::element::u8, decoded_image.get_shape());

    // convert to u8 image
    const float* decoded_data = decoded_image.data<const float>();
    std::uint8_t* generated_data = generated_image.data<std::uint8_t>();
    for (size_t i = 0; i < decoded_image.get_size(); ++i) {
        generated_data[i] = static_cast<std::uint8_t>(std::clamp(decoded_data[i] * 0.5f + 0.5f, 0.0f, 1.0f) * 255);
    }

    return generated_image;
}

int32_t main(int32_t argc, char* argv[]) {
    cxxopts::Options options("stable_diffusion", "Stable Diffusion implementation in C++ using OpenVINO\n");

    options.add_options()
    ("p,posPrompt", "Initial positive prompt for LCM ", cxxopts::value<std::string>()->default_value("a beautiful pink unicorn"))
    ("n,negPrompt","Defaut is empty with space", cxxopts::value<std::string>()->default_value(" "))
    ("d,device", "AUTO, CPU, or GPU", cxxopts::value<std::string>()->default_value("CPU"))
    ("step", "Number of diffusion steps", cxxopts::value<size_t>()->default_value("4"))
    ("s,seed", "Number of random seed to generate latent for one image output", cxxopts::value<size_t>()->default_value("42"))
    ("num", "Number of image output", cxxopts::value<size_t>()->default_value("1"))
    ("height", "destination image height", cxxopts::value<size_t>()->default_value("512"))
    ("width", "destination image width", cxxopts::value<size_t>()->default_value("512"))
    ("c,useCache", "use model caching", cxxopts::value<bool>()->default_value("false"))
    ("r,readNPLatent", "read numpy generated latents from file", cxxopts::value<bool>()->default_value("false"))
    ("m,modelPath", "Specify path of LCM model IRs", cxxopts::value<std::string>()->default_value("../scripts/SimianLuo/LCM_Dreamshaper_v7"))
    ("t,type", "Specify the type of LCM model IRs (e.g., FP16_static or FP16_dyn)", cxxopts::value<std::string>()->default_value("FP16_static"))
    ("l,loraPath", "Specify path of LoRA file. (*.safetensors).", cxxopts::value<std::string>()->default_value(""))
    ("a,alpha", "alpha for LoRA", cxxopts::value<float>()->default_value("0.75"))
    ("h,help", "Print usage");
    cxxopts::ParseResult result;

    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
        std::cout << e.what() << "\n\n";
        std::cout << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    const std::string positive_prompt = result["posPrompt"].as<std::string>();
    const std::string negative_prompt = result["negPrompt"].as<std::string>();
    const std::string device = result["device"].as<std::string>();
    const uint32_t num_inference_steps = result["step"].as<size_t>();
    const uint32_t user_seed = result["seed"].as<size_t>();
    const uint32_t num_images = result["num"].as<size_t>();
    const uint32_t height = result["height"].as<size_t>();
    const uint32_t width = result["width"].as<size_t>();
    const bool use_cache = result["useCache"].as<bool>();
    const bool read_np_latent = result["readNPLatent"].as<bool>();
    const std::string model_base_path = result["modelPath"].as<std::string>();
    const std::string model_type = result["type"].as<std::string>();
    const std::string lora_path = result["loraPath"].as<std::string>();
    const float alpha = result["alpha"].as<float>();

    const std::string folder_name = "images";
    try {
        std::filesystem::create_directory(folder_name);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create dir" << e.what() << std::endl;
    }

    std::cout << "OpenVINO version: " << ov::get_openvino_version() << std::endl;
    std::cout << "Running (may take some time) ..." << std::endl;

    // Stable Diffusion pipeline

    StableDiffusionModels models = compile_models(model_base_path, device, model_type, lora_path, alpha, use_cache);
    ov::InferRequest unet_infer_request = models.unet.create_infer_request();

    ov::PartialShape sample_shape = models.unet.input("sample").get_partial_shape();
    OPENVINO_ASSERT(sample_shape.is_dynamic() || (sample_shape[2] * 8 == width && sample_shape[3] * 8 == height),
        "UNet model has static shapes [1, 4, H/8, W/8] or dynamic shapes [?, 4, ?, ?]");

    // no negative prompt for LCM model: 
    // https://huggingface.co/docs/diffusers/api/pipelines/latent_consistency_models#diffusers.LatentConsistencyModelPipeline
    ov::Tensor text_embeddings = text_encoder(models, positive_prompt);

    std::shared_ptr<Scheduler> scheduler = std::make_shared<LCMScheduler>(LCMScheduler(
        1000, 0.00085f, 0.012f, BetaSchedule::SCALED_LINEAR,
        PredictionType::EPSILON, {}, 50, true, 10.0f, false,
        false, 1.0f, 0.995f, 1.0f, read_np_latent));
    scheduler->set_timesteps(num_inference_steps);
    std::vector<std::int64_t> timesteps = scheduler->get_timesteps();

    ov::Tensor denoised(ov::element::f32, {1, 4, height / 8, width / 8});
    for (uint32_t n = 0; n < num_images; n++) {
        std::uint32_t seed = num_images == 1 ? user_seed: n;

        ov::Tensor latent_model_input = randn_tensor(height, width, read_np_latent, seed);

        float guidance_scale = 8.0;
        ov::Tensor guidance_scale_embedding = get_w_embedding(guidance_scale, 256);

        for (size_t inference_step = 0; inference_step < num_inference_steps; inference_step++) {
            ov::Tensor timestep(ov::element::i64, {1}, &timesteps[inference_step]);
            ov::Tensor noisy_residual = unet(unet_infer_request, latent_model_input, timestep, text_embeddings, guidance_scale_embedding);

            auto step_res = scheduler->step(noisy_residual, latent_model_input, inference_step);
            latent_model_input = step_res["latent"], denoised = step_res["denoised"];
        }

        ov::Tensor decoded_image = vae_decoder(models.vae_decoder, denoised);
        imwrite(std::string("./images/seed_") + std::to_string(seed) + ".bmp", postprocess_image(decoded_image), true);
        std::cout << "Result image saved to: " << std::string("./images/seed_") + std::to_string(seed) + ".bmp" << std::endl;
    }

    return EXIT_SUCCESS;
}
