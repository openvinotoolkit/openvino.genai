// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/visibility.hpp"

namespace ov::genai {

struct OPENVINO_GENAI_EXPORTS SpeechGenerationPerfMetrics : public PerfMetrics {
    size_t num_generated_samples = 0;

    // Number of generated discrete acoustic codec frames (e.g. 12Hz frames for Qwen3-TTS).
    size_t num_generated_frames = 0;

    // Output audio sample rate in Hz, used to compute Real-Time Factor (rtf).
    uint32_t output_sample_rate = 0;

    // Time To First Audio chunk in milliseconds (mean/std across requests).
    MeanStdPair ttfa = {0.0f, 0.0f};

    // Total acoustic/codec token generation duration in milliseconds (Talker + CodePredictor).
    MeanStdPair codegen_duration = {0.0f, 0.0f};

    // Total speech tokenizer / vocoder audio decoding duration in milliseconds.
    MeanStdPair audio_decode_duration = {0.0f, 0.0f};

    // Real-Time Factor = total generation duration / generated audio duration (lower is faster; <1.0 is faster than real time).
    float rtf = 0.0f;

    // Raw per-request measurements in microseconds, evaluated by evaluate_statistics().
    std::vector<MicroSeconds> raw_ttfa_durations;
    std::vector<MicroSeconds> raw_codegen_durations;
    std::vector<MicroSeconds> raw_audio_decode_durations;

    void evaluate_statistics(std::optional<TimePoint> start_time = std::nullopt) override;
};
}  // namespace ov::genai
