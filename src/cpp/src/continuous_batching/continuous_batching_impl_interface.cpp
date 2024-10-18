// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/continuous_batching_impl_interface.hpp"

namespace ov::genai {
GenerationConfig ContinuousBatchingPipeline::ImplInterface::get_config() const {
    return m_generation_config;
}

PipelineMetrics ContinuousBatchingPipeline::ImplInterface::get_metrics() const {
    return m_pipeline_metrics;
}

Tokenizer ContinuousBatchingPipeline::ImplInterface::get_tokenizer() {
    return m_tokenizer;
}

void ContinuousBatchingPipeline::ImplInterface::start_chat(const std::string& system_message) {
    if (!system_message.empty()) {
        m_history.push_back({{"role", "system"}, {"content", system_message}});
    }
    m_is_chat_conversation = true;
};

void ContinuousBatchingPipeline::ImplInterface::finish_chat() {
    m_is_chat_conversation = false;
    m_history.clear();
};
std::vector<GenerationResult>
ContinuousBatchingPipeline::ImplInterface::generate(
    const std::vector<std::string>& prompts,
    std::vector<ov::genai::GenerationConfig> sampling_params,
    const StreamerVariant& streamer) {
    std::vector<ov::Tensor> input_ids;
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
        for (const std::string& prompt : prompts) {
            timer.start();
            input_ids.push_back(m_tokenizer.encode(prompt).input_ids);
            timer.end();
        }
    }
    std::vector<EncodedGenerationResult> encoded = generate(input_ids, sampling_params, streamer);
    std::vector<GenerationResult> decoded;
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
    return decoded;
}
}