// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

#include "openvino/genai/llm_pipeline.hpp"

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

int main(int argc, char* argv[]) try {
    if (3 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<PROMPT>\"");

    std::string models_path = argv[1];
    std::string prompt = argv[2];

    std::string device = "CPU";  // GPU, NPU can be used as well

    ov::AnyMap config;
    if (device == "GPU") {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        config = get_config_for_cache_encryption();
    } 

    auto [model_str, model_weights] = decrypt_model(models_path, "openvino_model.xml", "openvino_model.bin");
    ov::genai::Tokenizer tokenizer = decrypt_tokenizer(models_path);
    
    ov::genai::LLMPipeline pipe(model_str, model_weights, tokenizer, device, config);

    std::string result = pipe.generate(prompt, ov::genai::max_new_tokens(100));
    std::cout << result << std::endl;
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
