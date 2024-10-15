// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (3 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<PROMPT>\"");

    std::string model_path = argv[1];
    std::string prompt = argv[2];
    std::string device = "CPU";  // GPU can be used as well

    ov::genai::LLMPipeline pipe(model_path, device);
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    auto result = pipe.generate(prompt, config);
    std::cout << result << std::endl;

    std::cout << "\n";

    std::cout << "get_load_time " << result.perf_metrics.get_load_time() << '\n';
    std::cout << "get_num_generated_tokens " << result.perf_metrics.get_num_generated_tokens() << '\n';
    std::cout << "get_num_input_tokens " << result.perf_metrics.get_num_input_tokens() << '\n';
    std::cout << "get_ttft " << result.perf_metrics.get_ttft().mean << '\n';
    std::cout << "get_tpot " << result.perf_metrics.get_tpot().mean << '\n';
    std::cout << "get_ipot " << result.perf_metrics.get_ipot().mean << '\n';
    std::cout << "get_throughput " << result.perf_metrics.get_throughput().mean << '\n';
    std::cout << "get_inference_duration " << result.perf_metrics.get_inference_duration().mean << '\n';
    std::cout << "get_generate_duration " << result.perf_metrics.get_generate_duration().mean << '\n';
    std::cout << "get_tokenization_duration " << result.perf_metrics.get_tokenization_duration().mean << '\n';
    std::cout << "get_detokenization_duration " << result.perf_metrics.get_detokenization_duration().mean << '\n';
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
