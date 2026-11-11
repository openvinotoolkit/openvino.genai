// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "openvino/genai/omni/talker_perf_metrics.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

/**
 * @brief Speech output: the waveform plus its perf metrics.
 * Returned by TalkerBase::generate, and held by VLMDecodedResults::speech_result.
 *
 * @note This is a preview API and is subject to change.
 */
struct OPENVINO_GENAI_EXPORTS TalkerResults {
    /// Speech output waveforms. Empty when `talker_speech_config.return_audio == false`.
    std::vector<ov::Tensor> waveforms;

    /// Speech-side perf metrics: wall-clock time and sample count.
    /// Populated even when `waveforms` is empty so callers can see how long the no-op path took.
    TalkerPerfMetrics perf_metrics;
};

}  // namespace ov::genai
