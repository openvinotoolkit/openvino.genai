// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"
#include <cxxopts.hpp>
#include <fstream>
#include <sstream>

std::string read_prompt(const std::string& file_path) {
    std::string prompt;
    std::ifstream file(file_path);
    if (file.is_open()) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        prompt = buffer.str();
        file.close();        
    }
    return prompt;
}

int main(int argc, char* argv[]) try {
    cxxopts::Options options("benchmark_vanilla_genai", "Help command");

    options.add_options()
    ("m,model", "Path to model and tokenizers base directory", cxxopts::value<std::string>())
    ("p,prompt", "Prompt", cxxopts::value<std::string>()->default_value("The Sky is blue because"))
    ("pf,promptfile", "Prompt from file")
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
    std::string device = result["device"].as<std::string>();
    size_t num_warmup = result["num_warmup"].as<size_t>();
    size_t num_iter = result["num_iter"].as<size_t>();

    if (result.count("promptfile")) {
        prompt = read_prompt(result["promptfile"].as<std::string>());
    }

    ov::genai::GenerationConfig config;
    config.max_new_tokens = result["max_new_tokens"].as<size_t>();

    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.enable_prefix_caching = false;
    scheduler_config.max_num_batched_tokens = 2147483647;

    ov::genai::LLMPipeline pipe(models_path, device, ov::genai::scheduler_config(scheduler_config));

    for (size_t i = 0; i < num_warmup; i++)
        pipe.generate(prompt, config);

    ov::genai::DecodedResults res = pipe.generate(prompt, config);
    ov::genai::PerfMetrics metrics = res.perf_metrics;
    for (size_t i = 0; i < num_iter - 1; i++) {
        res = pipe.generate(prompt, config);
        metrics = metrics + res.perf_metrics;
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Load time: " << metrics.get_load_time() << " ms" << std::endl;
    std::cout << "Generate time: " << metrics.get_generate_duration().mean << " ± " << metrics.get_generate_duration().std << " ms" << std::endl;
    std::cout << "Tokenization time: " << metrics.get_tokenization_duration().mean << " ± " << metrics.get_tokenization_duration().std << " ms" << std::endl;
    std::cout << "Detokenization time: " << metrics.get_detokenization_duration().mean << " ± " << metrics.get_detokenization_duration().std << " ms" << std::endl;
    std::cout << "TTFT: " << metrics.get_ttft().mean  << " ± " << metrics.get_ttft().std << " ms" << std::endl;
    std::cout << "TPOT: " << metrics.get_tpot().mean  << " ± " << metrics.get_tpot().std << " ms/token " << std::endl;
    std::cout << "Throughput: " << metrics.get_throughput().mean  << " ± " << metrics.get_throughput().std << " tokens/s" << std::endl;

    return EXIT_SUCCESS;
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
