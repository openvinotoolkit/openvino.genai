// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <fstream>

#include "load_image.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"

std::pair<std::string, ov::Tensor> decrypt_model(const std::string& model_path, const std::string& weights_path) {
    std::ifstream model_file(model_path);
    std::ifstream weights_file(weights_path, std::ios::binary);
    if (!model_file.is_open() || !weights_file.is_open()) {
        throw std::runtime_error("Cannot open model or weights file");
    }

    // User can add file decryption of model_file and weights_file in memory here.

    std::string model_str((std::istreambuf_iterator<char>(model_file)), std::istreambuf_iterator<char>());

    weights_file.seekg(0, std::ios::end);
    auto weight_size = static_cast<unsigned>(weights_file.tellg());
    weights_file.seekg(0, std::ios::beg);
    auto weights_tensor = ov::Tensor(ov::element::u8, {weight_size});
    if (!weights_file.read(static_cast<char*>(weights_tensor.data()), weight_size)) {
        throw std::runtime_error("Cannot read weights file");
    }

    return {model_str, weights_tensor};
}

ov::genai::Tokenizer decrypt_tokenizer(const std::string& models_path) {
    std::string tok_model_path = models_path + "/openvino_tokenizer.xml";
    std::string tok_weights_path = models_path + "/openvino_tokenizer.bin";
    auto [tok_model_str, tok_weights_tensor] = decrypt_model(tok_model_path, tok_weights_path);

    std::string detok_model_path = models_path + "/openvino_detokenizer.xml";
    std::string detok_weights_path = models_path + "/openvino_detokenizer.bin";
    auto [detok_model_str, detok_weights_tensor] = decrypt_model(detok_model_path, detok_weights_path);

    return ov::genai::Tokenizer(tok_model_str, tok_weights_tensor, detok_model_str, detok_weights_tensor);
}


bool print_subword(std::string&& subword) {
    return !(std::cout << subword << std::flush);
}

int main(int argc, char* argv[]) try {
    if (4 != argc) {
        throw std::runtime_error(std::string{"Usage "} + argv[0] + " <MODEL_DIR> <IMAGE_FILE OR DIR_WITH_IMAGES> <PROMPT>");
    }

    //read and encrypt models
    std::string models_path = argv[1];
    ov::genai::ModelsMap models_map;

    std::map<std::string, std::string> model_name_to_file_map = {
        {"language", "openvino_language_model"},
        {"resampler", "openvino_resampler_model"},
        {"text_embeddings", "openvino_text_embeddings_model"},
        {"vision_embeddings", "openvino_vision_embeddings_model"}};

    for (const auto& [model_name, file_name] : model_name_to_file_map) {
        auto model_pair = decrypt_model(file_name + model_name + ".xml", file_name + model_name + ".bin");
        models_map.emplace(model_name, std::move(model_pair));
    }

    ov::genai::Tokenizer tokenizer = decrypt_tokenizer(models_path);

    // GPU can be used as well.
    std::string device = "CPU";
    ov::AnyMap enable_compile_cache;
    if (device == "GPU") {
        // Cache compiled models on disk for GPU to save time on the
        // next run. It's not beneficial for CPU.
        enable_compile_cache.insert({ov::cache_dir("vlm_cache")});
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
