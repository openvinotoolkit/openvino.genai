// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "icontinuous_batching.hpp"

namespace ov::genai {

GenerationConfig ContinuousBatchingPipeline::IContinuousBatchingPipeline::get_config() const {
    return m_generation_config;
}

PipelineMetrics ContinuousBatchingPipeline::IContinuousBatchingPipeline::get_metrics() const {
    return m_pipeline_metrics;
}

Tokenizer ContinuousBatchingPipeline::IContinuousBatchingPipeline::get_tokenizer() {
    return m_tokenizer;
}

void ContinuousBatchingPipeline::IContinuousBatchingPipeline::start_chat(const std::string& system_message) {
    if (!system_message.empty()) {
        m_history.push_back({{"role", "system"}, {"content", system_message}});
    }
    m_is_chat_conversation = true;
};

void ContinuousBatchingPipeline::IContinuousBatchingPipeline::finish_chat() {
    m_is_chat_conversation = false;
    m_history.clear();
};

std::pair<std::vector<GenerationResult>, PerfMetrics>
ContinuousBatchingPipeline::IContinuousBatchingPipeline::generate(
    const std::vector<std::string>& prompts,
    std::vector<ov::genai::GenerationConfig> sampling_params,
    const StreamerVariant& streamer) {
    std::vector<ov::Tensor> input_ids;
    auto start_time = std::chrono::steady_clock::now();

    static ManualTimer timer("tokenize");
    if (m_is_chat_conversation) {
        OPENVINO_ASSERT(1 == prompts.size(), "Can't chat with multiple prompts");
        m_history.push_back({{"role", "user"}, {"content", prompts.at(0)}});
        constexpr bool add_generation_prompt = true;
        std::string history = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
        timer.start();
        // ov::genai::add_special_tokens(false) is aligned with stateful pipeline
        input_ids.push_back(m_tokenizer.encode(history, ov::genai::add_special_tokens(false)).input_ids);
        timer.end();
    } else {
        input_ids.reserve(prompts.size());
        timer.start();
        for (const std::string& prompt : prompts) {
            input_ids.push_back(m_tokenizer.encode(prompt).input_ids);
        }
        timer.end();
    }

    // std::vector<EncodedGenerationResult> encoded = generate(input_ids, sampling_params, streamer);
    auto [encoded, perf_metrics] = generate(input_ids, sampling_params, streamer);
    std::vector<GenerationResult> decoded;
    auto decode_start_time =  std::chrono::steady_clock::now();
    decoded.reserve(encoded.size());
    for (EncodedGenerationResult& res : encoded) {
        std::vector<std::string> generated;
        generated.reserve(res.m_generation_ids.size());
        for (size_t idx = 0; idx < res.m_generation_ids.size(); ++idx) {
            generated.push_back(m_tokenizer.decode(res.m_generation_ids.at(idx)));
            if (m_is_chat_conversation && 0 == idx) {
                m_history.push_back({{"role", "assistant"}, {"content", generated.back()}});
            }
        }

        decoded.push_back(GenerationResult{
            res.m_request_id,
            std::move(generated),
            std::move(res.m_scores),
            res.m_status
        });
    }
    auto stop_time = std::chrono::steady_clock::now();

    auto& raw_counters = perf_metrics.raw_metrics;
    raw_counters.generate_durations = std::vector<MicroSeconds>();
    raw_counters.generate_durations.emplace_back(PerfMetrics::get_microsec(stop_time - start_time));
    raw_counters.tokenization_durations.emplace_back(PerfMetrics::get_microsec(decode_start_time - start_time));
    raw_counters.detokenization_durations.emplace_back(PerfMetrics::get_microsec(stop_time - decode_start_time));
    perf_metrics.m_evaluated = false;
    perf_metrics.evaluate_statistics(start_time);

    return {decoded, perf_metrics};
}
}
