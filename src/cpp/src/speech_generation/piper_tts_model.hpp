// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>

#include "openvino/genai/speech_generation/speech_generation_config.hpp"
#include "text2speech_pipeline_impl.hpp"

namespace ov {
namespace genai {

/// @brief Backend for Piper (VITS-family, single-graph, non-autoregressive) text-to-speech
/// voices. Unlike Kokoro/SpeechT5 the model has no separate encoder/decoder split and
/// performs a single forward pass to produce the full waveform. Each voice is single-speaker,
/// so a non-empty speaker_embedding is rejected.
class PiperTTSImpl : public Text2SpeechPipelineImpl {
public:
    PiperTTSImpl(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties);

    Text2SpeechDecodedResults generate(const std::vector<std::string>& texts,
                                       const ov::Tensor& speaker_embedding,
                                       const SpeechGenerationConfig& generation_config) override;

    ov::Shape get_speaker_embedding_shape() const override {
        // Piper voices bundled in this backend are single-speaker; no speaker embedding is used.
        return ov::Shape{0};
    }

private:
    std::vector<int64_t> build_phoneme_id_sequence(const std::string& text) const;

    std::filesystem::path m_models_path;
    ov::InferRequest m_request;
    std::string m_input_name;
    std::string m_input_lengths_name;
    std::string m_scales_name;
    std::unordered_map<std::string, int64_t> m_phoneme_id_map;
    uint32_t m_sample_rate = 22050;
    std::string m_espeak_voice = "en-us";
};

}  // namespace genai
}  // namespace ov
