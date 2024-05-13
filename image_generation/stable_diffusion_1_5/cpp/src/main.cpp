// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "cxxopts.hpp"
#include "imwrite.hpp"
#include "lora.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"
#include "scheduler_lms_discrete.hpp"

const size_t TOKENIZER_MODEL_MAX_LENGTH = 77;   // 'model_max_length' parameter from 'tokenizer_config.json'
const size_t VAE_SCALE_FACTOR = 8;

class Timer {
    const decltype(std::chrono::steady_clock::now()) m_start;

public:
    Timer(const std::string& scope) : m_start(std::chrono::steady_clock::now()) {
        (std::cout << scope << ": ").flush();
    }

    ~Timer() {
        auto m_end = std::chrono::steady_clock::now();
        std::cout << std::chrono::duration<double, std::milli>(m_end - m_start).count() << " ms" << std::endl;
    }
};

ov::Tensor randn_tensor(ov::Shape shape, bool use_np_latents, uint32_t seed = 42) {
    ov::Tensor noise(ov::element::f32, shape);
    if (use_np_latents) {
        // read np generated latents with defaut seed 42
        const char* latent_file_name = "../np_latents_512x512.txt";
        std::ifstream latent_copy_file(latent_file_name, std::ios::ate);
        OPENVINO_ASSERT(latent_copy_file.is_open(), "Cannot open ", latent_file_name);

        size_t file_size = latent_copy_file.tellg() / sizeof(float);
        OPENVINO_ASSERT(file_size >= noise.get_size(),
                        "Cannot generate ",
                        noise.get_shape(),
                        " with ",
                        latent_file_name,
                        ". File size is small");

        latent_copy_file.seekg(0, std::ios::beg);
        for (size_t i = 0; i < noise.get_size(); ++i)
            latent_copy_file >> noise.data<float>()[i];
    } else {
        std::mt19937 gen{seed};
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

void apply_lora(std::shared_ptr<ov::Model> model, InsertLoRA::LoRAMap& lora_map) {
    if (!lora_map.empty()) {
        ov::pass::Manager manager;
        manager.register_pass<InsertLoRA>(lora_map);
        manager.run_passes(model);
    }
}

void reshape_text_encoder(std::shared_ptr<ov::Model> model, size_t batch_size, size_t tokenizer_model_max_length) {
    ov::PartialShape input_shape = model->input(0).get_partial_shape();
    input_shape[0] = batch_size;
    input_shape[1] = tokenizer_model_max_length;
    std::map<size_t, ov::PartialShape> idx_to_shape{{0, input_shape}};
    model->reshape(idx_to_shape);
}

void reshape_unet_encoder(std::shared_ptr<ov::Model> model,
                          int64_t batch_size,
                          int64_t height,
                          int64_t width,
                          int64_t tokenizer_model_max_length) {
    // The factor of 2 comes from the guidance scale > 1
    for (auto input : model->inputs()) {
        if (input.get_any_name().find("timestep_cond") == std::string::npos) {
            batch_size *= 2;
            break;
        }
    }

    height = height / VAE_SCALE_FACTOR;
    width = width / VAE_SCALE_FACTOR;

    std::map<std::string, ov::PartialShape> name_to_shape;

    for (auto input : model->inputs()) {
        std::string input_name = input.get_any_name();
        name_to_shape[input_name] = input.get_partial_shape();
        if (input_name == "timestep") {
            name_to_shape[input_name][0] = 1;
        } else if (input_name == "sample") {
            name_to_shape[input_name] = {batch_size, name_to_shape[input_name][1], height, width};
        } else if (input_name == "time_ids") {
            name_to_shape[input_name][0] = batch_size;
        } else {
            name_to_shape[input_name][0] = batch_size;
            name_to_shape[input_name][1] = TOKENIZER_MODEL_MAX_LENGTH;
        }
    }

    model->reshape(name_to_shape);
}

void reshape_vae_decoder(std::shared_ptr<ov::Model> model, int64_t height, int64_t width) {
    height = height / VAE_SCALE_FACTOR;
    width = width / VAE_SCALE_FACTOR;

    ov::PartialShape input_shape = model->input(0).get_partial_shape();
    std::map<size_t, ov::PartialShape> idx_to_shape{{0, {1, input_shape[1], height, width}}};
    model->reshape(idx_to_shape);
}

StableDiffusionModels compile_models(const std::string& model_path,
                                     const std::string& device,
                                     const std::string& lora_path,
                                     const float alpha,
                                     const bool use_cache,
                                     const bool use_dynamic_shapes,
                                     const size_t batch_size,
                                     const size_t height,
                                     const size_t width) {
    StableDiffusionModels models;

    ov::Core core;
    if (use_cache)
        core.set_property(ov::cache_dir("./cache_dir"));

    core.add_extension(TOKENIZERS_LIBRARY_PATH);

    // read LoRA weights
    std::map<std::string, InsertLoRA::LoRAMap> lora_weights;
    if (!lora_path.empty()) {
        Timer t("Loading and multiplying LoRA weights");
        lora_weights = read_lora_adapters(lora_path, alpha);
    }

    // Text encoder
    {
        Timer t("Loading and compiling text encoder");
        auto text_encoder_model = core.read_model(model_path + "/text_encoder/openvino_model.xml");
        if (!use_dynamic_shapes) {
            reshape_text_encoder(text_encoder_model, batch_size, TOKENIZER_MODEL_MAX_LENGTH);
        }
        apply_lora(text_encoder_model, lora_weights["text_encoder"]);
        models.text_encoder = core.compile_model(text_encoder_model, device);
    }

    // UNet
    {
        Timer t("Loading and compiling UNet");
        auto unet_model = core.read_model(model_path + "/unet/openvino_model.xml");
        if (!use_dynamic_shapes) {
            reshape_unet_encoder(unet_model, batch_size, height, width, TOKENIZER_MODEL_MAX_LENGTH);
        }
        apply_lora(unet_model, lora_weights["unet"]);
        models.unet = core.compile_model(unet_model, device);
    }

    // VAE decoder
    {
        Timer t("Loading and compiling VAE decoder");
        auto vae_decoder_model = core.read_model(model_path + "/vae_decoder/openvino_model.xml");
        if (!use_dynamic_shapes) {
            reshape_vae_decoder(vae_decoder_model, height, width);
        }
        ov::preprocess::PrePostProcessor ppp(vae_decoder_model);
        ppp.output().model().set_layout("NCHW");
        ppp.output().tensor().set_layout("NHWC");
        models.vae_decoder = core.compile_model(vae_decoder_model = ppp.build(), device);
    }

    // Tokenizer
    {
        Timer t("Loading and compiling tokenizer");
        // Tokenizer model wil be loaded to CPU: OpenVINO Tokenizers can be inferred on a CPU device only.
        models.tokenizer = core.compile_model(model_path + "/tokenizer/openvino_tokenizer.xml", "CPU");
    }

    return models;
}

ov::Tensor text_encoder(StableDiffusionModels models, std::string& pos_prompt, std::string& neg_prompt) {
    const size_t HIDDEN_SIZE = static_cast<size_t>(models.text_encoder.output(0).get_partial_shape()[2].get_length());
    const int32_t EOS_TOKEN_ID = 49407, PAD_TOKEN_ID = EOS_TOKEN_ID;
    const ov::Shape input_ids_shape({1, TOKENIZER_MODEL_MAX_LENGTH});

    ov::InferRequest tokenizer_req = models.tokenizer.create_infer_request();
    ov::InferRequest text_encoder_req = models.text_encoder.create_infer_request();

    auto compute_text_embeddings = [&](std::string& prompt, ov::Tensor encoder_output_tensor) {
        ov::Tensor input_ids(ov::element::i32, input_ids_shape);
        std::fill_n(input_ids.data<int32_t>(), input_ids.get_size(), PAD_TOKEN_ID);

        // tokenization
        tokenizer_req.set_input_tensor(ov::Tensor{ov::element::string, {1}, &prompt});
        tokenizer_req.infer();
        ov::Tensor input_ids_token = tokenizer_req.get_tensor("input_ids");
        std::copy_n(input_ids_token.data<std::int64_t>(), input_ids_token.get_size(), input_ids.data<std::int32_t>());

        // text embeddings
        text_encoder_req.set_tensor("input_ids", input_ids);
        text_encoder_req.set_output_tensor(0, encoder_output_tensor);
        text_encoder_req.infer();
    };

    ov::Tensor text_embeddings(ov::element::f32, {2, TOKENIZER_MODEL_MAX_LENGTH, HIDDEN_SIZE});

    compute_text_embeddings(neg_prompt,
                            ov::Tensor(text_embeddings, {0, 0, 0}, {1, TOKENIZER_MODEL_MAX_LENGTH, HIDDEN_SIZE}));
    compute_text_embeddings(pos_prompt,
                            ov::Tensor(text_embeddings, {1, 0, 0}, {2, TOKENIZER_MODEL_MAX_LENGTH, HIDDEN_SIZE}));

    return text_embeddings;
}

ov::Tensor unet(ov::InferRequest req, ov::Tensor sample, ov::Tensor timestep, ov::Tensor text_embedding_1d) {
    req.set_tensor("sample", sample);
    req.set_tensor("timestep", timestep);
    req.set_tensor("encoder_hidden_states", text_embedding_1d);

    req.infer();

    ov::Tensor noise_pred_tensor = req.get_output_tensor();
    ov::Shape noise_pred_shape = noise_pred_tensor.get_shape();
    noise_pred_shape[0] = 1;

    // perform guidance
    const float guidance_scale = 7.5f;
    const float* noise_pred_uncond = noise_pred_tensor.data<const float>();
    const float* noise_pred_text = noise_pred_uncond + ov::shape_size(noise_pred_shape);

    ov::Tensor noisy_residual(noise_pred_tensor.get_element_type(), noise_pred_shape);
    for (size_t i = 0; i < ov::shape_size(noise_pred_shape); ++i)
        noisy_residual.data<float>()[i] =
            noise_pred_uncond[i] + guidance_scale * (noise_pred_text[i] - noise_pred_uncond[i]);

    return noisy_residual;
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

int32_t main(int32_t argc, char* argv[]) try {
    cxxopts::Options options("stable_diffusion", "Stable Diffusion implementation in C++ using OpenVINO\n");

    options.add_options()
    ("p,posPrompt", "Initial positive prompt for SD ", cxxopts::value<std::string>()->default_value("cyberpunk cityscape like Tokyo New York  with tall buildings at dusk golden hour cinematic lighting"))
    ("n,negPrompt", "Defaut is empty with space", cxxopts::value<std::string>()->default_value(" "))
    ("d,device", "AUTO, CPU, or GPU.\nDoesn't apply to Tokenizer model, OpenVINO Tokenizers can be inferred on a CPU device only", cxxopts::value<std::string>()->default_value("CPU"))
    ("step", "Number of diffusion steps", cxxopts::value<size_t>()->default_value("20"))
    ("s,seed", "Number of random seed to generate latent for one image output", cxxopts::value<size_t>()->default_value("42"))
    ("num", "Number of image output", cxxopts::value<size_t>()->default_value("1"))
    ("height", "Destination image height", cxxopts::value<size_t>()->default_value("512"))
    ("width", "Destination image width", cxxopts::value<size_t>()->default_value("512"))
    ("c,useCache", "Use model caching", cxxopts::value<bool>()->default_value("false"))
    ("r,readNPLatent", "Read numpy generated latents from file", cxxopts::value<bool>()->default_value("false"))
    ("m,modelPath", "Specify path of SD model IRs", cxxopts::value<std::string>()->default_value("./models/dreamlike_anime_1_0_ov"))
    ("t,type", "Specify the type of SD model IRs (FP32, FP16 or INT8)", cxxopts::value<std::string>()->default_value("FP16"))
    ("dynamic", "Specify the model input shape to use dynamic shape", cxxopts::value<bool>()->default_value("false"))
    ("l,loraPath", "Specify path of LoRA file. (*.safetensors).", cxxopts::value<std::string>()->default_value(""))
    ("a,alpha", "alpha for LoRA", cxxopts::value<float>()->default_value("0.75"))("h,help", "Print usage");
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

    std::string positive_prompt = result["posPrompt"].as<std::string>();
    std::string negative_prompt = result["negPrompt"].as<std::string>();
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
    const bool use_dynamic_shapes = result["dynamic"].as<bool>();
    const std::string lora_path = result["loraPath"].as<std::string>();
    const float alpha = result["alpha"].as<float>();

    OPENVINO_ASSERT(
        !read_np_latent || (read_np_latent && (num_images == 1)),
        "\"readNPLatent\" option is only supported for one output image. Number of image output was set to " +
            std::to_string(num_images));

    const std::string folder_name = "images";
    try {
        std::filesystem::create_directory(folder_name);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create dir" << e.what() << std::endl;
    }

    std::cout << "OpenVINO version: " << ov::get_openvino_version() << std::endl;

    const std::string model_path = model_base_path + "/" + model_type;
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "Model IRs for type " << model_type << " don't exist in directory " << model_path << "\n";
        std::cerr << "Refer to README.md to know how to export OpenVINO model with particular data type." << std::endl;
        return EXIT_FAILURE;
    }

    // Stable Diffusion pipeline
    const size_t batch_size = 1;
    StableDiffusionModels models =
        compile_models(model_path, device, lora_path, alpha, use_cache, use_dynamic_shapes, batch_size, height, width);
    ov::InferRequest unet_infer_request = models.unet.create_infer_request();

    ov::PartialShape sample_shape = models.unet.input("sample").get_partial_shape();
    OPENVINO_ASSERT(sample_shape.is_dynamic() ||
                        (sample_shape[2] * VAE_SCALE_FACTOR == height && sample_shape[3] * VAE_SCALE_FACTOR == width),
                    "UNet model has static shapes [1, 4, H/8, W/8] or dynamic shapes [?, 4, ?, ?]");

    Timer t("Running Stable Diffusion pipeline");

    ov::Tensor text_embeddings = text_encoder(models, positive_prompt, negative_prompt);

    std::shared_ptr<Scheduler> scheduler = std::make_shared<LMSDiscreteScheduler>();
    scheduler->set_timesteps(num_inference_steps);
    std::vector<std::int64_t> timesteps = scheduler->get_timesteps();

    for (uint32_t n = 0; n < num_images; n++) {
        std::uint32_t seed = num_images == 1 ? user_seed : user_seed + n;

        const size_t unet_in_channels = static_cast<size_t>(sample_shape[1].get_length());

        // latents are multiplied by 'init_noise_sigma'
        ov::Shape latent_shape = ov::Shape({batch_size, unet_in_channels, height / VAE_SCALE_FACTOR, width / VAE_SCALE_FACTOR});
        ov::Shape latent_model_input_shape = latent_shape;
        ov::Tensor noise = randn_tensor(latent_shape, read_np_latent, seed);
        latent_model_input_shape[0] = 2;  // Unet accepts batch 2
        ov::Tensor latent(ov::element::f32, latent_shape),
            latent_model_input(ov::element::f32, latent_model_input_shape);
        for (size_t i = 0; i < noise.get_size(); ++i) {
            latent.data<float>()[i] = noise.data<float>()[i] * scheduler->get_init_noise_sigma();
        }

        for (size_t inference_step = 0; inference_step < num_inference_steps; inference_step++) {
            // concat the same latent twice along a batch dimension
            latent.copy_to(
                ov::Tensor(latent_model_input, {0, 0, 0, 0}, {1, latent_shape[1], latent_shape[2], latent_shape[3]}));
            latent.copy_to(
                ov::Tensor(latent_model_input, {1, 0, 0, 0}, {2, latent_shape[1], latent_shape[2], latent_shape[3]}));

            scheduler->scale_model_input(latent_model_input, inference_step);

            ov::Tensor timestep(ov::element::i64, {1}, &timesteps[inference_step]);
            ov::Tensor noisy_residual = unet(unet_infer_request, latent_model_input, timestep, text_embeddings);

            latent = scheduler->step(noisy_residual, latent, inference_step)["latent"];
        }

        ov::Tensor decoded_image = vae_decoder(models.vae_decoder, latent);
        imwrite(std::string("./images/seed_") + std::to_string(seed) + ".bmp", postprocess_image(decoded_image), true);
    }

    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
