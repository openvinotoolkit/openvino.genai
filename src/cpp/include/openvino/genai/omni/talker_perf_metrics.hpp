// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

#include "openvino/genai/visibility.hpp"

namespace ov::genai {

/**
 * @brief Performance metrics for Talker speech generation.
 *
 * Captures wall-clock time and output size for a single generate() call.
 * Unlike SpeechGenerationPerfMetrics, this type omits LLM-oriented fields
 * (TTFT, TPOT, tokenization durations) that are not applicable to speech codec workflows.
 *
 * @note This is a preview API and is subject to change.
 */
struct OPENVINO_GENAI_EXPORTS TalkerPerfMetrics {
    /// Number of audio samples generated (waveform length in samples, not bytes).
    size_t num_generated_samples = 0;

    /// Total speech generation time in milliseconds (autoregressive token prediction + vocoding).
    /// Vocoding = converting neural codec tokens to PCM waveform, not WAV file I/O.
    float generation_time_ms = 0;
};

}  // namespace ov::genai
