// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/runtime/compiled_model.hpp"

namespace ov {
namespace genai {

/**
 * @brief Structure to keep speech generation config parameters.
 */
class OPENVINO_GENAI_EXPORTS SpeechGenerationConfig : public GenerationConfig {
public:
    SpeechGenerationConfig();
    explicit SpeechGenerationConfig(const std::filesystem::path& json_path);

    // ---------------------------------------------------------------------
    // SpeechT5-specific parameters
    // ---------------------------------------------------------------------

    // Minimum ratio of output length to input text length; prevents output that's too short
    float minlenratio = 0.0;

    // Maximum ratio of output length to input text length; prevents excessively long outputs
    float maxlenratio = 20.0;

    // Probability threshold for stopping decoding; when output probability exceeds above this, generation will stop
    float threshold = 0.5;

    // ---------------------------------------------------------------------
    // Kokoro-specific parameters
    // ---------------------------------------------------------------------

    // Speech speed multiplier.
    float speed = 1.0f;

    // Language code used by Kokoro G2P (for example: en-us, en-gb).
    std::string language = "en-us";

    // Maximum phoneme sequence length per Kokoro preprocessing chunk.
    uint32_t max_phoneme_length = 510;

    // Optional OpenVINO phonemizer fallback model directory.
    // This fallback is used only during text phonemization / G2P (graphemes -> phonemes),
    // before acoustic inference.
    // - set: use this OpenVINO fallback network for G2P fallback.
    // - unset: default to espeak-ng G2P fallback.
    std::optional<std::filesystem::path> phonemize_fallback_model_dir;

    // ---------------------------------------------------------------------
    // Qwen3-TTS-specific parameters
    // ---------------------------------------------------------------------

    // Optional predefined speaker name for CustomVoice variants.
    std::string speaker;

    // Optional instruction text that controls speaking style.
    std::string instruct;

    // Qwen prompt assembly mode.
    // - true: non-streaming prompt assembly (default).
    // - false: streaming-style prompt assembly.
    bool non_streaming_mode = true;

    // Whether to sample residual code groups via subtalker (code predictor).
    bool subtalker_dosample = true;

    // Top-k for subtalker sampling.
    uint32_t subtalker_top_k = 50;

    // Top-p for subtalker sampling.
    float subtalker_top_p = 1.0f;

    // Temperature for subtalker sampling.
    float subtalker_temperature = 0.9f;

    // Qwen3 Base voice-clone reference transcript for ICL mode.
    std::string voice_clone_ref_text;

    // Qwen3 Base voice-clone reference audio waveform for internal prompt extraction.
    // Expected tensor shape: [T], [1, T], or [1, 1, T].
    // Expected element type: f32.
    // Expected sample rate: 24000 Hz.
    // OV GenAI does not resample or decode files for this property.
    ov::Tensor voice_clone_ref_audio;

    // Qwen3 Base voice-clone reference codec ids for ICL mode.
    // Expected shape: [T, G] or [1, T, G].
    ov::Tensor voice_clone_ref_codec_ids;

    void update_generation_config(const ov::AnyMap& config_map = {}) override;

    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> update_generation_config(Properties&&... properties) {
        return update_generation_config(AnyMap{std::forward<Properties>(properties)...});
    }

    /// @brief checks that are no conflicting parameters.
    /// @throws Exception if config is invalid.
    void validate() const override;
};

static constexpr ov::Property<float> minlenratio{"minlenratio"};
static constexpr ov::Property<float> maxlenratio{"maxlenratio"};
static constexpr ov::Property<float> threshold{"threshold"};

static constexpr ov::Property<float> speed{"speed"};

static constexpr ov::Property<std::string> speech_language{"language"};
static constexpr ov::Property<uint32_t> max_phoneme_length{"max_phoneme_length"};
static constexpr ov::Property<std::filesystem::path> phonemize_fallback_model_dir{"phonemize_fallback_model_dir"};

static constexpr ov::Property<std::string> speaker{"speaker"};
static constexpr ov::Property<std::string> instruct{"instruct"};
static constexpr ov::Property<bool> non_streaming_mode{"non_streaming_mode"};
static constexpr ov::Property<bool> subtalker_dosample{"subtalker_dosample"};
static constexpr ov::Property<uint32_t> subtalker_top_k{"subtalker_top_k"};
static constexpr ov::Property<float> subtalker_top_p{"subtalker_top_p"};
static constexpr ov::Property<float> subtalker_temperature{"subtalker_temperature"};

static constexpr ov::Property<std::string> voice_clone_ref_text{"voice_clone_ref_text"};
static constexpr ov::Property<ov::Tensor> voice_clone_ref_audio{"voice_clone_ref_audio"};
static constexpr ov::Property<ov::Tensor> voice_clone_ref_codec_ids{"voice_clone_ref_codec_ids"};

}  // namespace genai
}  // namespace ov
