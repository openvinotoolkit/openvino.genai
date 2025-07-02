// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "text2speech_pipeline_impl.hpp"

namespace ov {
namespace genai {

SpeechGenerationPerfMetrics Text2SpeechPipelineImpl::get_performance_metrics() {
    m_perf_metrics.load_time = m_load_time_ms;
    return m_perf_metrics;
}

void Text2SpeechPipelineImpl::save_load_time(std::chrono::steady_clock::time_point start_time) {
    auto stop_time = std::chrono::steady_clock::now();
    m_load_time_ms += std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
}
}  // namespace genai
}  // namespace ov
