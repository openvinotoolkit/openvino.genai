// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <fstream>

#include "load_image.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"

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
    config.insert({ov::cache_dir("llm_cache")});
    ov::EncryptionCallbacks encryption_callbacks;
    //use XOR-based encryption as an example
    encryption_callbacks.encrypt = encryption_callback;
    encryption_callbacks.decrypt = decryption_callback;
    config.insert(ov::cache_encryption_callbacks(encryption_callbacks));
    config.insert(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
    return config;
}

bool print_subword(std::string&& subword) {
    return !(std::cout << subword << std::flush);
}

int main(int argc, char* argv[]) try {
    if (4 != argc) {
        throw std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE OR DIR_WITH_IMAGES> <PROMPT>");
    }

    //read and encrypt models
    std::filesystem::path models_path = argv[1];
    ov::genai::ModelsMap models_map;

    std::map<std::string, std::string> model_name_to_file_map = {
        {"language", "openvino_language_model"},
        {"resampler", "openvino_resampler_model"},
        {"text_embeddings", "openvino_text_embeddings_model"},
        {"vision_embeddings", "openvino_vision_embeddings_model"}};

    for (const auto& [model_name, file_name] : model_name_to_file_map) {
        models_map.emplace(model_name, decrypt_model(models_path, file_name + ".xml", file_name + ".bin"));
    }

    ov::genai::Tokenizer tokenizer = decrypt_tokenizer(models_path);

    // GPU can be used as well.
    std::string device = "CPU";
    ov::AnyMap enable_compile_cache;
    if (device == "GPU") {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        enable_compile_cache = get_config_for_cache_encryption();
    }
    ov::genai::VLMPipeline pipe(models_map, tokenizer, models_path,  device, enable_compile_cache);

    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 100;

    std::vector<ov::Tensor> rgbs = utils::load_images(argv[2]);

    std::string prompt = argv[3];
    pipe.generate(prompt,
                  ov::genai::images(rgbs),
                  ov::genai::generation_config(generation_config),
                  ov::genai::streamer(print_subword));

} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
