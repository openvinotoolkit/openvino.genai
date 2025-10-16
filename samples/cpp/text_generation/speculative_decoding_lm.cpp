// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/speculative_decoding/perf_metrics.hpp"

int main(int argc, char* argv[]) try {
    if (4 != argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <DRAFT_MODEL_DIR> '<PROMPT>'");
    }

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    // Speculative decoding generation parameters like `num_assistant_tokens` and `assistant_confidence_threshold` are mutually excluded.
    // Add parameter to enable speculative decoding to generate `num_assistant_tokens` candidates by draft_model per iteration.
    // NOTE: ContinuousBatching backend uses `num_assistant_tokens` as is. Stateful backend uses `num_assistant_tokens`'s copy as initial
    // value and adjusts it based on recent number of accepted tokens. If `num_assistant_tokens` is not set, it defaults to `5` for both
    // backends.
    config.num_assistant_tokens = 4;
    // Add parameter to enable speculative decoding to generate candidates by draft_model while candidate probability is higher than
    // `assistant_confidence_threshold`.
    // NOTE: `assistant_confidence_threshold` is supported only by ContinuousBatching backend.
    // config.assistant_confidence_threshold = 0.4;

    std::string main_model_path = argv[1];
    std::string draft_model_path = argv[2];
    std::string prompt = argv[3];

    // User can run main and draft model on different devices.
    // Please, set device for main model in `LLMPipeline` constructor and in `ov::genai::draft_model` for draft.
    // CPU, GPU and NPU can be used. For NPU, the preferred configuration is when both the main and draft models
    // use NPU.
    std::string main_device = "CPU", draft_device = "CPU";

    ov::genai::LLMPipeline pipe(
        main_model_path,
        main_device,
        ov::genai::draft_model(draft_model_path, draft_device));

    auto streamer = [](std::string subword) {
        std::cout << subword << std::flush;
        return ov::genai::StreamingStatus::RUNNING;
    };

    // Since the streamer is set, the results will
    // be printed each time a new token is generated.
    auto result = pipe.generate(prompt, config, streamer);
    auto sd_perf_metrics = std::dynamic_pointer_cast<ov::genai::SDPerModelsPerfMetrics>(result.extended_perf_metrics);

    if (sd_perf_metrics) {
        auto main_model_metrics = sd_perf_metrics->main_model_metrics;
        std::cout << "\nMAIN MODEL " << std::endl;
        std::cout << "  Generate time: " << main_model_metrics.get_generate_duration().mean << " ms" << std::endl;
        std::cout << "  TTFT: " << main_model_metrics.get_ttft().mean  << " ± " << main_model_metrics.get_ttft().std << " ms" << std::endl;
        std::cout << "  TTST: " << main_model_metrics.get_ttst().mean  << " ± " << main_model_metrics.get_ttst().std << " ms/token " << std::endl;
        std::cout << "  TPOT: " << main_model_metrics.get_tpot().mean  << " ± " << main_model_metrics.get_tpot().std << " ms/iteration " << std::endl;
        std::cout << "  AVG Latency: " << main_model_metrics.get_latency().mean  << " ± " << main_model_metrics.get_latency().std << " ms/token " << std::endl;
        std::cout << "  Num generated token: " << main_model_metrics.get_num_generated_tokens() << " tokens" << std::endl;
        std::cout << "  Total iteration number: " << main_model_metrics.raw_metrics.m_durations.size() << std::endl;
        std::cout << "  Num accepted token: " << sd_perf_metrics->get_num_accepted_tokens() << " tokens" << std::endl;

        auto draft_model_metrics = sd_perf_metrics->draft_model_metrics;
        std::cout << "\nDRAFT MODEL " << std::endl;
        std::cout << "  Generate time: " << draft_model_metrics.get_generate_duration().mean << " ms" << std::endl;
        std::cout << "  TTFT: " << draft_model_metrics.get_ttft().mean  << " ms" << std::endl;
        std::cout << "  TTST: " << draft_model_metrics.get_ttst().mean  << " ms/token " << std::endl;
        std::cout << "  TPOT: " << draft_model_metrics.get_tpot().mean  << " ± " << draft_model_metrics.get_tpot().std << " ms/token " << std::endl;
        std::cout << "  AVG Latency: " << draft_model_metrics.get_latency().mean  << " ± " << draft_model_metrics.get_latency().std << " ms/iteration " << std::endl;
        std::cout << "  Num generated token: " << draft_model_metrics.get_num_generated_tokens() << " tokens" << std::endl;
        std::cout << "  Total iteration number: " << draft_model_metrics.raw_metrics.m_durations.size() << std::endl;
    }
    std::cout << std::endl;
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