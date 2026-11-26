// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

int main(int argc, char* argv[]) try {
    if (argc < 4 || argc > 5)
        throw std::runtime_error(std::string{"Usage: "} + argv[0] +
                                 " <MODEL_DIR> <PREDICTOR_SAFETENSORS_FILE> \"<PROMPT>\" [OFFLOAD_RATIO]");

    std::string models_path = argv[1];
    std::string predictor_path = argv[2];
    std::string prompt = argv[3];
    size_t offload_ratio = argc > 4 ? std::stoul(argv[4]) : 50;
    std::string device = "GPU";  // LES is implemented in the GPU plugin

    using namespace ov::genai;

    // Lookahead expert scheduling (LES) for Mixture-of-Experts models with offloaded experts: OFFLOAD_RATIO % of the
    // routed experts stay in host memory and are uploaded to the GPU on demand. The LES predictor tells the GPU plugin
    // which experts the next MoE layer is likely to select, so it can start uploading them in advance.
    // The predictor does not change the generated text, only the decode speed.
    // The GPU plugin reads the offloaded experts from the weights file of the model.
    LESPredictor predictor(predictor_path);
    LLMPipeline pipe(models_path, device,
                     les_predictor(predictor),
                     ov::intel_gpu::offload_ratio(offload_ratio),
                     ov::weights_path((std::filesystem::path(models_path) / "openvino_model.bin").string()));

    GenerationConfig config;
    config.max_new_tokens = 100;
    DecodedResults result = pipe.generate(prompt, config);
    std::cout << result << std::endl;

    PerfMetrics metrics = result.perf_metrics;
    std::cout << "\nOFFLOAD_RATIO " << offload_ratio << ": TTFT " << metrics.get_ttft().mean << " ms, decode "
              << metrics.get_throughput().mean << " tokens/s" << std::endl;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Non-exception object thrown\n";
    return EXIT_FAILURE;
}
