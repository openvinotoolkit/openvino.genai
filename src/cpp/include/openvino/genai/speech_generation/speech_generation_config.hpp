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
    // Piper-specific parameters
    // ---------------------------------------------------------------------

    // Stochastic duration/flow predictor noise scale; controls prosody variety.
    float noise_scale = 0.667f;

    // Multiplier applied to predicted phoneme durations; controls speaking rate.
    float length_scale = 1.0f;

    // Stochastic duration predictor noise weight.
    float noise_w = 0.8f;

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

static constexpr ov::Property<float> noise_scale{"noise_scale"};
static constexpr ov::Property<float> length_scale{"length_scale"};
static constexpr ov::Property<float> noise_w{"noise_w"};

}  // namespace genai
}  // namespace ov
