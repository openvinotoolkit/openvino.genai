// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/speech_generation/speech_generation_perf_metrics.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"

namespace ov::genai {

/**
 * @brief Omni-specific decoded results including speech outputs.
 *
 * Extends VLMDecodedResults with speech output waveforms for Qwen3-Omni models.
 */
class OPENVINO_GENAI_EXPORTS OmniDecodedResults : public VLMDecodedResults {
public:
    /// Speech output waveforms (one per generated result).
    /// Empty if speech generation was not requested (return_audio=false).
    std::vector<ov::Tensor> speech_outputs;

    /// Speech-side perf metrics. Populated regardless of `return_audio`; when speech is
    /// disabled, holds the trivially-fast no-op path's timing. Use alongside the inherited
    /// `perf_metrics` (text-side) to break down end-to-end Omni latency.
    SpeechGenerationPerfMetrics speech_perf_metrics;
};

}  // namespace ov::genai
