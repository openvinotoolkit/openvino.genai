// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (3 > argc)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> \"<PROMPT>\"");

    std::string models_path = argv[1];
    std::string prompt = argv[2];
    std::string device = "CPU";  // GPU can be used as well

    ov::genai::LLMPipeline pipe(models_path, device);
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    int iter = 0;
    while (iter < 10) {
        auto result = pipe.generate(prompt, config);
        std::cout << result.texts << std::endl;
        iter++;
        std::cout << "\n pipeline generate finish iter:" << iter << std::endl;
        std::cout << "generate duration s:" << result.perf_metrics.get_generate_duration().mean * 0.001 << std::endl;
        std::cout << "inference duration s:" << result.perf_metrics.get_inference_duration().mean * 0.001 << std::endl;
        std::cout << "first token time s:" << result.perf_metrics.ttft.mean * 0.001 << std::endl;
        std::cout << "output token size:" << result.perf_metrics.raw_metrics.m_token_infer_durations.size()
                  << std::endl;
    }
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
