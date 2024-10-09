// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching_impl_interface.hpp"

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
}