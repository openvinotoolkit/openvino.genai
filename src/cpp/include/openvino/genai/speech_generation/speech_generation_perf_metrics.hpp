// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov::genai {

struct OPENVINO_GENAI_EXPORTS SpeechGenerationPerfMetrics : public PerfMetrics {
    size_t num_generated_samples = 0;

    void evaluate_statistics(std::optional<TimePoint> start_time = std::nullopt) override;
};
}  // namespace ov::genai
