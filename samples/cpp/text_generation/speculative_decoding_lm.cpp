// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

#include "openvino/genai/llm_pipeline.hpp"
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
    if (4 != argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <DRAFT_MODEL_DIR> '<PROMPT_FILE_PATH>'");
    }

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 128;
    // Speculative decoding generation parameters like `num_assistant_tokens` and `assistant_confidence_threshold` are mutually excluded
    // add parameter to enable speculative decoding to generate `num_assistant_tokens` candidates by draft_model per iteration
    config.num_assistant_tokens = 5;
    // add parameter to enable speculative decoding to generate candidates by draft_model while candidate probability is higher than `assistant_confidence_threshold`
    // config.assistant_confidence_threshold = 0.4;

    std::string main_model_path = argv[1];
    std::string draft_model_path = argv[2];
    // std::string prompt = argv[3];
    std::string prompt_path = argv[3];
    std::string prompt = read_prompt(prompt_path);

    size_t num_warmup = 1;
    size_t num_iter = 3;

    // User can run main and draft model on different devices.
    // Please, set device for main model in `LLMPipeline` constructor and in in `ov::genai::draft_model` for draft.
    std::string main_device = "GPU", draft_device = "GPU";

    ov::genai::LLMPipeline pipe(main_model_path,
                                main_device,
                                ov::genai::draft_model(draft_model_path, draft_device),
                                ov::hint::enable_cpu_reservation(true));

    auto streamer = [](std::string subword) {
        std::cout << subword << std::flush;
        return ov::genai::StreamingStatus::RUNNING;
    };

    // Since the streamer is set, the results will
    // be printed each time a new token is generated.
    // pipe.generate(prompt, config, streamer);
    // std::cout << std::endl;

    for (size_t i = 0; i < num_warmup; i++)
        pipe.generate(prompt, config);

    ov::genai::DecodedResults res = pipe.generate(prompt, config);
    ov::genai::PerfMetrics metrics = res.perf_metrics;
    for (auto i = 0; i < res.texts.size(); i++) {
        std::cout << "generate text is:" << res.texts[0] << std::endl;
    }
    for (size_t i = 0; i < num_iter - 1; i++) {
        res = pipe.generate(prompt, config);
        metrics = metrics + res.perf_metrics;
        for (auto i = 0; i < res.texts.size(); i++) {
            std::cout << "generate text is:" << res.texts[0] << std::endl;
        }
    }

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Load time: " << metrics.get_load_time() << " ms" << std::endl;
    std::cout << "Generate time: " << metrics.get_generate_duration().mean << " ± "
              << metrics.get_generate_duration().std << " ms" << std::endl;
    std::cout << "Tokenization time: " << metrics.get_tokenization_duration().mean << " ± "
              << metrics.get_tokenization_duration().std << " ms" << std::endl;
    std::cout << "Detokenization time: " << metrics.get_detokenization_duration().mean << " ± "
              << metrics.get_detokenization_duration().std << " ms" << std::endl;
    std::cout << "TTFT: " << metrics.get_ttft().mean << " ± " << metrics.get_ttft().std << " ms" << std::endl;
    std::cout << "TPOT: " << metrics.get_tpot().mean << " ± " << metrics.get_tpot().std << " ms/token " << std::endl;
    std::cout << "Throughput: " << metrics.get_throughput().mean << " ± " << metrics.get_throughput().std << " tokens/s"
              << std::endl;
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
