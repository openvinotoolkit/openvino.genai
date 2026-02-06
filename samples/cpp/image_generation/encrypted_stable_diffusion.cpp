// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <fstream>

#include "imwrite.hpp"
#include "openvino/genai/image_generation/text2image_pipeline.hpp"
#include "progress_bar.hpp"


std::pair<std::string, ov::Tensor> decrypt_model(const std::filesystem::path& model_dir, const std::string& model_file_name, const std::string& weights_file_name) {
    std::ifstream model_file(model_dir / model_file_name);
    std::ifstream weights_file;
    if (!model_file.is_open()) {
        throw std::runtime_error("Cannot open model file");
    }
    std::string model_str((std::istreambuf_iterator<char>(model_file)), std::istreambuf_iterator<char>());

    // read weights file using mmap to reduce memory consumption
    auto weights_tensor = ov::read_tensor_data(model_dir / weights_file_name);

    // User can add file decryption of model_file and weights_file in memory here.

    return {model_str, weights_tensor};
}

ov::genai::Tokenizer decrypt_tokenizer(const std::filesystem::path& models_path) {
    auto [tok_model_str, tok_weights_tensor] = decrypt_model(models_path, "openvino_tokenizer.xml", "openvino_tokenizer.bin");
    auto [detok_model_str, detok_weights_tensor] = decrypt_model(models_path, "openvino_detokenizer.xml", "openvino_detokenizer.bin");

    return ov::genai::Tokenizer(tok_model_str, tok_weights_tensor, detok_model_str, detok_weights_tensor);
}

static const char codec_key[] = {0x30, 0x60, 0x70, 0x02, 0x04, 0x08, 0x3F, 0x6F, 0x72, 0x74, 0x78, 0x7F};

std::string codec_xor(const std::string& source_str) {
    auto key_size = sizeof(codec_key);
    int key_idx = 0;
    std::string dst_str = source_str;
    for (char& c : dst_str) {
        c ^= codec_key[key_idx % key_size];
        key_idx++;
    }
    return dst_str;
}

std::string encryption_callback(const std::string& source_str) {
    return codec_xor(source_str);
}

std::string decryption_callback(const std::string& source_str) {
    return codec_xor(source_str);
}

auto get_config_for_cache_encryption() {
    ov::AnyMap config;
    config.insert({ov::cache_dir("cache")});
    ov::EncryptionCallbacks encryption_callbacks;
    //use XOR-based encryption as an example
    encryption_callbacks.encrypt = encryption_callback;
    encryption_callbacks.decrypt = decryption_callback;
    config.insert(ov::cache_encryption_callbacks(encryption_callbacks));
    config.insert(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
    return config;
}

int32_t main(int32_t argc, char* argv[]) try {
    OPENVINO_ASSERT(argc >= 3 && argc <= 4, "Usage: ", argv[0], " <MODEL_DIR> '<PROMPT>' <DEVICE>");

    const std::filesystem::path models_path = argv[1];
    const std::string prompt = argv[2];

    const std::string device = (argc > 3) ? argv[3] : "CPU";

    ov::AnyMap config;
    if (device == "GPU") {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        config = get_config_for_cache_encryption();
    } 

    auto [text_encoder_model_str, text_encoder_model_weights] = decrypt_model(models_path / "text_encoder", "openvino_model.xml", "openvino_model.bin");
    ov::genai::Tokenizer text_tokenizer = decrypt_tokenizer(models_path / "tokenizer");
    const ov::genai::CLIPTextModel text_encoder = ov::genai::CLIPTextModel(
            text_encoder_model_str, 
            text_encoder_model_weights, 
            ov::genai::CLIPTextModel::Config::Config(models_path / "text_encoder" / "config.json"), 
            text_tokenizer, device, config);

    auto [text_encoder_2_model_str, text_encoder_2_model_weights] = decrypt_model(models_path / "text_encoder_2", "openvino_model.xml", "openvino_model.bin");
    ov::genai::Tokenizer text_tokenizer_2 = decrypt_tokenizer(models_path / "tokenizer_2");
    const ov::genai::CLIPTextModelWithProjection text_encoder_2 = ov::genai::CLIPTextModelWithProjection(
            text_encoder_2_model_str, 
            text_encoder_2_model_weights, 
            ov::genai::CLIPTextModelWithProjection::Config::Config(models_path / "text_encoder_2" / "config.json"), 
            text_tokenizer_2, device, config);

    auto [vae_decoder_model_str, vae_decoder_model_weights] = decrypt_model(models_path / "vae_decoder", "openvino_model.xml", "openvino_model.bin");
    const ov::genai::AutoencoderKL vae_decoder = ov::genai::AutoencoderKL(
            vae_decoder_model_str, 
            vae_decoder_model_weights, 
            ov::genai::AutoencoderKL::Config::Config(models_path / "vae_decoder" / "config.json"),
            device, config);
    
    auto [unet_model_str, unet_model_weights] = decrypt_model(models_path / "unet", "openvino_model.xml", "openvino_model.bin");
    const ov::genai::UNet2DConditionModel unet = ov::genai::UNet2DConditionModel(
            unet_model_str, 
            unet_model_weights,
            ov::genai::UNet2DConditionModel::Config::Config(models_path / "unet" / "config.json"),
            vae_decoder.get_vae_scale_factor(),
            device, config);

    ov::genai::Text2ImagePipeline pipe = ov::genai::Text2ImagePipeline::stable_diffusion_xl(
        ov::genai::Scheduler::from_config(models_path / "scheduler" / "scheduler_config.json"),
        text_encoder,
        text_encoder_2,
        unet,
        vae_decoder
    );

    ov::Tensor image = pipe.generate(prompt,
        ov::genai::width(512),
        ov::genai::height(512),
        ov::genai::num_inference_steps(20),
        ov::genai::num_images_per_prompt(1),
        ov::genai::callback(progress_bar));

    // writes `num_images_per_prompt` images by pattern name
    imwrite("image_%d.bmp", image, true);

    return EXIT_SUCCESS;
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {
    }
    return EXIT_FAILURE;
}
