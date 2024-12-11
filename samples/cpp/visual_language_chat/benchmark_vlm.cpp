// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cxxopts.hpp>
#include <filesystem>

#include "load_image.hpp"
#include <openvino/genai/visual_language/pipeline.hpp>


int main(int argc, char* argv[]) try {
    cxxopts::Options options("benchmark_vlm", "Help command");

    options.add_options()
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>()->default_value("."))
    ("p,prompt", "Prompt", cxxopts::value<std::string>()->default_value("What is on the image?"))
    ("i,image", "Image", cxxopts::value<std::string>()->default_value("image.jpg"))
    ("nw,num_warmup", "Number of warmup iterations", cxxopts::value<size_t>()->default_value(std::to_string(1)))
    ("n,num_iter", "Number of iterations", cxxopts::value<size_t>()->default_value(std::to_string(3)))
    ("mt,max_new_tokens", "Maximal number of new tokens", cxxopts::value<size_t>()->default_value(std::to_string(20)))
    ("d,device", "device", cxxopts::value<std::string>()->default_value("CPU"))
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

    std::string prompt = result["prompt"].as<std::string>();
    const std::string models_path = result["model"].as<std::string>();
    const std::string image_path = result["image"].as<std::string>();
    std::string device = result["device"].as<std::string>();
    size_t num_warmup = result["num_warmup"].as<size_t>();
    size_t num_iter = result["num_iter"].as<size_t>();
    ov::Tensor image = utils::load_image(image_path);
  
    ov::genai::GenerationConfig config;
    config.max_new_tokens = result["max_new_tokens"].as<size_t>();

    ov::genai::VLMPipeline pipe(models_path, device);
    
    for (size_t i = 0; i < num_warmup; i++)
        pipe.generate(prompt, ov::genai::image(image), ov::genai::generation_config(config));
    
    auto res = pipe.generate(prompt, ov::genai::image(image), ov::genai::generation_config(config));
    auto metrics = res.perf_metrics;
    for (size_t i = 0; i < num_iter - 1; i++) {
        res = pipe.generate(prompt, ov::genai::image(image), ov::genai::generation_config(config));
        metrics = metrics + res.perf_metrics;
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Load time: " << metrics.get_load_time() << " ms" << std::endl;
    std::cout << "Generate time: " << metrics.get_generate_duration().mean << " ± " << metrics.get_generate_duration().std << " ms" << std::endl;
    std::cout << "Tokenization time: " << metrics.get_tokenization_duration().mean << " ± " << metrics.get_tokenization_duration().std << " ms" << std::endl;
    std::cout << "Detokenization time: " << metrics.get_detokenization_duration().mean << " ± " << metrics.get_detokenization_duration().std << " ms" << std::endl;
    std::cout << "Embeddings preparation time: " << metrics.get_prepare_embeddings_duration().mean << " ± " << metrics.get_prepare_embeddings_duration().std << " ms" << std::endl;
    std::cout << "TTFT: " << metrics.get_ttft().mean  << " ± " << metrics.get_ttft().std << " ms" << std::endl;
    std::cout << "TPOT: " << metrics.get_tpot().mean  << " ± " << metrics.get_tpot().std << " ms/token " << std::endl;
    std::cout << "Throughput: " << metrics.get_throughput().mean  << " ± " << metrics.get_throughput().std << " tokens/s" << std::endl;

    return 0;
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
