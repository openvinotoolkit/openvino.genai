// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "openvino/genai/automatic_speech_recognition/voice_activity_detector.hpp"
#include "openvino/runtime/infer_request.hpp"

namespace ov::genai {

/**
 * @brief Speech region as 16 kHz sample indices [start_sample, end_sample).
 */
struct SpeechSegment {
    size_t start_sample = 0;
    size_t end_sample = 0;
};

}  // namespace ov::genai

namespace ov::genai::silero_vad {

constexpr const char* SEQUENCE_MODEL_FILE = "silero_vad_16k_sequence.onnx";
constexpr size_t SAMPLE_RATE = 16000;
constexpr size_t WINDOW_SAMPLES = 512;
constexpr size_t CONTEXT_SAMPLES = 64;
constexpr size_t STATE_SIZE = 128;
constexpr size_t MAX_BLOCK_FRAMES = 512;

/**
 * @brief Silero VAD 16 kHz sequence model with v6.3 segmentation.
 */
class SileroVAD {
public:
    SileroVAD(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties);

    /**
     * @brief Processes audio in blocks, carrying recurrent state
     * between blocks and resetting it for each new audio call,
     * then segments speech.
     * @param audio Mono 16 kHz float PCM samples.
     * @param properties Optional ov::genai::silero_vad_config;
     *                   any other property is rejected.
     */
    std::vector<SpeechSegment> get_speech_segments(const std::vector<float>& audio, const ov::AnyMap& properties);

private:
    std::vector<float> get_speech_probabilities(const std::vector<float>& audio);

    ov::InferRequest m_request;
};

/**
 * @brief Port of upstream Silero VAD v6.3 get_speech_timestamps_from_probs() for 16 kHz audio.
 *
 * Expects one probability per 512-sample window of audio_length_samples, the last window possibly partial.
 */
std::vector<SpeechSegment> segments_from_probabilities(const std::vector<float>& probabilities,
                                                       size_t audio_length_samples,
                                                       const SileroVADConfig& config);

}  // namespace ov::genai::silero_vad
