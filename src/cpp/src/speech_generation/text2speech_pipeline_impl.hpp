// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/speech_generation/speech_generation_config.hpp"
#include "openvino/genai/speech_generation/speech_generation_perf_metrics.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

namespace ov {
namespace genai {

class Text2SpeechPipelineImpl {
public:
    GenerationConfig get_generation_config() const {
        return m_generation_config;
    }

    void set_generation_config(const GenerationConfig& generation_config) {
        m_generation_config = generation_config;
        m_generation_config.validate();
    }

    virtual Text2SpeechDecodedResults generate(const std::vector<std::string>& texts,
                                               const ov::Tensor& speaker_embedding,
                                               const SpeechGenerationConfig& generation_config) = 0;

    virtual SpeechGenerationPerfMetrics get_performance_metrics();

    virtual ~Text2SpeechPipelineImpl() = default;

protected:
    void save_load_time(std::chrono::steady_clock::time_point start_time);

    GenerationConfig m_generation_config;
    float m_load_time_ms = 0.0f;
    SpeechGenerationPerfMetrics m_perf_metrics;
};

}  // namespace genai
}  // namespace ov
