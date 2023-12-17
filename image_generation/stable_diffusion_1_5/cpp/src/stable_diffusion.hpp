// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @brief a header file for SD pipeline
 * @file stable_diffusion.hpp
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <queue>

#include "openvino/openvino.hpp"

#include <openvino_extensions/strings.hpp>

#include "scheduler.hpp"
#include "lora_cpp.hpp"

#include "progress_bar.hpp"
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

void convert_bgr2rgb(ov::Tensor image) {
    auto shape = image.get_shape();
    OPENVINO_ASSERT(image.get_element_type() == ov::element::u8 &&
        shape.size() == 4 && shape[0] == 1 && shape[3] == 3,
        "Image of u8 type and [1, H, W, 3] shape is expected");

    auto image_data = image.data<std::uint8_t>();
    for (size_t i = 0, spatial_dims = shape[1] * shape[2]; i < spatial_dims; ++i) {
        // Swap the red and blue components (BGR to RGB)
        std::swap(image_data[i * 3], image_data[i * 3 + 2]);
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct StableDiffusionModels {
    ov::CompiledModel text_encoder;
    ov::CompiledModel unet;
    ov::CompiledModel vae_decoder;
    ov::CompiledModel tokenizer;
};

ov::Tensor vae_decoder_function(ov::CompiledModel& decoder_compiled_model, ov::Tensor sample) {
    const float coeffs_const{1 / 0.18215};
    for (size_t i = 0; i < sample.get_size(); ++i)
        sample.data<float>()[i] *= coeffs_const;

    ov::InferRequest req = decoder_compiled_model.create_infer_request();
    req.set_input_tensor(sample);
    req.infer();

    ov::Tensor decoded_image = req.get_output_tensor();
    ov::Tensor generated_image(ov::element::u8, decoded_image.get_shape());

    // convert to u8 image
    const float* decoded_data = decoded_image.data<const float>();
    std::uint8_t* generated_data = generated_image.data<std::uint8_t>();
    for (size_t i = 0; i < decoded_image.get_size(); ++i) {
        generated_data[i] = static_cast<std::uint8_t>(std::clamp(decoded_data[i] * 0.5f + 0.5f, 0.0f, 1.0f) * 255);
    }

    return generated_image;
}

ov::Tensor unet_infer_function(ov::CompiledModel unet_model,
                               ov::Tensor timestep,
                               ov::Tensor latent_input_1d,
                               ov::Tensor text_embedding_1d) {
    ov::InferRequest unet_infer_request = unet_model.create_infer_request();

    unet_infer_request.set_tensor("sample", latent_input_1d);
    unet_infer_request.set_tensor("timestep", timestep);
    unet_infer_request.set_tensor("encoder_hidden_states", text_embedding_1d);

    unet_infer_request.infer();

    ov::Tensor noise_pred_tensor = unet_infer_request.get_output_tensor();
    ov::Shape noise_pred_shape = noise_pred_tensor.get_shape();
    noise_pred_shape[0] = 1;

    const float guidance_scale = 7.5f;
    const float* noise_pred_uncond = noise_pred_tensor.data<const float>();
    const float* noise_pred_text = noise_pred_uncond + ov::shape_size(noise_pred_shape);

    ov::Tensor noisy_residual(noise_pred_tensor.get_element_type(), noise_pred_shape);
    for (size_t i = 0; i < ov::shape_size(noise_pred_shape); ++i)
        noisy_residual.data<float>()[i] = noise_pred_uncond[i] + guidance_scale * (noise_pred_text[i] - noise_pred_uncond[i]);

    return noisy_residual;
}

ov::Tensor text_encoder_infer_function(StableDiffusionModels models, const std::string& pos_prompt, const std::string& neg_prompt) {
    const size_t MAX_LENGTH = 77 /* 'model_max_length' from 'tokenizer_config.json' */, BATCH_SIZE = 1;
    const int32_t EOS_TOKEN_ID = 49407, PAD_TOKEN_ID = EOS_TOKEN_ID;
    const ov::Shape input_ids_shape({2, MAX_LENGTH});

    ov::InferRequest text_encoder_req = models.text_encoder.create_infer_request();
    ov::Tensor input_ids = text_encoder_req.get_input_tensor();
    
    input_ids.set_shape(input_ids_shape);
    // need to pre-fill 'input_ids' with 'PAD' tokens
    std::int32_t* input_ids_data = input_ids.data<int32_t>();
    std::fill_n(input_ids_data, input_ids.get_size(), PAD_TOKEN_ID);

    // Tokenization

    ov::InferRequest tokenizer_req = models.tokenizer.create_infer_request();
    ov::Tensor packed_strings = tokenizer_req.get_input_tensor();

    openvino_extensions::pack_strings(std::array<std::string, BATCH_SIZE>{neg_prompt}, packed_strings);
    tokenizer_req.infer();
    ov::Tensor input_ids_neg = tokenizer_req.get_tensor("input_ids");
    std::copy_n(input_ids_neg.data<std::int32_t>(), input_ids_neg.get_size(), input_ids_data);

    openvino_extensions::pack_strings(std::array<std::string, BATCH_SIZE>{pos_prompt}, packed_strings);
    tokenizer_req.infer();
    ov::Tensor input_ids_pos = tokenizer_req.get_tensor("input_ids");
    std::copy_n(input_ids_pos.data<std::int32_t>(), input_ids_pos.get_size(), input_ids_data + MAX_LENGTH);

    // Text embedding
    text_encoder_req.infer();

    return text_encoder_req.get_output_tensor(0);
}

StableDiffusionModels compile_models(const std::string& model_path,
                                     const std::string& device,
                                     const std::string& type,
                                     const std::string& lora_path,
                                     const float alpha,
                                     const bool use_cache) {
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

void stable_diffusion(const std::string& positive_prompt = std::string{},
                      const std::vector<std::string>& output_images = {},
                      const std::string& device = std::string{},
                      const size_t num_inference_steps = 20,
                      const std::vector<uint32_t>& seed_vec = {},
                      const uint32_t num_images = 1,
                      const uint32_t height = 512,
                      const uint32_t width = 512,
                      const std::string negative_prompt = std::string{},
                      const bool use_cache = false,
                      const std::string& model_base_path = std::string{},
                      const std::string& model_type = std::string{},
                      const std::string& lora_path = std::string{},
                      const float alpha = 0.75f,
                      const bool read_np_latent = false) {
    StableDiffusionModels models = compile_models(model_base_path, device, model_type, lora_path, alpha, use_cache);

    ov::PartialShape sample_shape = models.unet.input("sample").get_partial_shape();
    OPENVINO_ASSERT(sample_shape.is_dynamic() || (sample_shape[2] * 8 == width && sample_shape[3] * 8 == height),
        "UNet model has static shapes [1, 4, H/8, W/8] or dynamic shapes [?, 4, ?, ?]");

    ov::Tensor text_embeddings = text_encoder_infer_function(models, positive_prompt, negative_prompt);

    for (uint32_t n = 0; n < num_images; n++) {
        ov::Tensor noise = randn_tensor(height, width, read_np_latent, seed_vec[n]);

        Scheduler scheduler;
        scheduler.set_timesteps(num_inference_steps);
        std::vector<std::int64_t> timesteps = scheduler.get_timesteps();

        // latents are multiplied by sigmas
        ov::Shape latent_shape = noise.get_shape(), latent_model_input_shape = latent_shape;
        latent_model_input_shape[0] = 2; // Unet accepts batch 2
        const ov::element::Type latent_type = noise.get_element_type();
        ov::Tensor latent(latent_type, latent_shape), latent_model_input(latent_type, latent_model_input_shape);
        for (size_t i = 0; i < noise.get_size(); ++i) {
            latent.data<float>()[i]  = noise.data<float>()[i] * scheduler.get_init_noise_sigma();
        }

        for (size_t inference_step = 0; inference_step < num_inference_steps; inference_step++) {
            float* latent_model_input_data = latent_model_input.data<float>();
            const float* sample_data = latent.data<const float>();
            for (size_t i = 0, lanent_size = ov::shape_size(latent_shape); i < lanent_size; ++i)
                latent_model_input_data[i] = latent_model_input_data[i + lanent_size] = sample_data[i];

            scheduler.scale_model_input(latent_model_input, inference_step);
            ov::Tensor timestep(ov::element::i64, {1}, &timesteps[inference_step]);
            ov::Tensor noisy_residual = unet_infer_function(models.unet, timestep, latent_model_input, text_embeddings);

            latent = scheduler.step(noisy_residual, latent, inference_step);
        }

        ov::Tensor generated_image = vae_decoder_function(models.vae_decoder, latent);

        convert_bgr2rgb(generated_image);
        imwrite(output_images[n], generated_image);
    }
}
