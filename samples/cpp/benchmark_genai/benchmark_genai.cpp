// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"
#include <cxxopts.hpp>

int main(int argc, char* argv[]) try {
    cxxopts::Options options("benchmark_vanilla_genai", "Help command");

    options.add_options()
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>()->default_value("."))
    ("p,prompt", "Prompt", cxxopts::value<std::string>()->default_value("The Sky is blue because"))
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
    const std::string model_path = result["model"].as<std::string>();
    std::string device = result["device"].as<std::string>();
    size_t num_warmup = result["num_warmup"].as<size_t>();
    size_t num_iter = result["num_iter"].as<size_t>();
  
    ov::genai::GenerationConfig config;
    config.max_new_tokens = result["max_new_tokens"].as<size_t>();

    ov::genai::LLMPipeline pipe(model_path, device);
    
    for (size_t i = 0; i < num_warmup; i++)
        pipe.generate(prompt, config);
    
    ov::genai::DecodedResults res = pipe.generate(prompt, config);
    ov::genai::PerfMetrics metrics = res.perf_metrics;
    for (size_t i = 0; i < num_iter - 1; i++) {
        res = pipe.generate(prompt, config);
        metrics = metrics + res.perf_metrics;
    }

    std::cout << "Load time: " << metrics.load_time << " ms" << std::endl;
    std::cout << "Generate time: " << metrics.generate_duration.mean << " ± " << metrics.generate_duration.std << " ms" << std::endl;
    std::cout << "Tokenization time: " << metrics.tokenization_duration.mean << " ± " << metrics.tokenization_duration.std << " ms" << std::endl;
    std::cout << "Detokenization time: " << metrics.detokenization_duration.mean << " ± " << metrics.detokenization_duration.std << " ms" << std::endl;
    std::cout << "ttft: " << metrics.ttft.mean  << " ± " << metrics.ttft.std << " ms" << std::endl;
    std::cout << "tpot: " << metrics.tpot.mean  << " ± " << metrics.tpot.std << " ms " << std::endl;
    std::cout << "Tokens/s: " << metrics.throughput.mean  << " ± " << metrics.throughput.std << std::endl;

    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
