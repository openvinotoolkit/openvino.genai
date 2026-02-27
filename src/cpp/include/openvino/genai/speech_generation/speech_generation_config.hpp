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

    // Explicit backend override. Empty means auto-detect from model metadata/files.
    std::string model_type;

    // Shared speech-generation speed multiplier.
    float speed = 1.0f;

    // Output sample rate. Backend-specific behavior may apply.
    uint32_t sample_rate = 16000;

    // Minimum ratio of output length to input text length; prevents output that's too short
    float minlenratio = 0.0;

    // Maximum ratio of output length to input text length; prevents excessively long outputs
    float maxlenratio = 20.0;

    // Probability threshold for stopping decoding; when output probability exceeds above this, generation will stop
    float threshold = 0.5;

    // Kokoro: language code used by G2P.
    std::string language = "en-us";

    // Kokoro: voice name or identifier.
    std::string voice;

    // Kokoro: max phoneme sequence length per chunk.
    uint32_t max_phoneme_length = 510;

    void update_generation_config(const ov::AnyMap& config_map = {});

    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> update_generation_config(Properties&&... properties) {
        return update_generation_config(AnyMap{std::forward<Properties>(properties)...});
    }

    /// @brief checks that are no conflicting parameters.
    /// @throws Exception if config is invalid.
    void validate() const;
};

static constexpr ov::Property<float> minlenratio{"minlenratio"};
static constexpr ov::Property<float> maxlenratio{"maxlenratio"};
static constexpr ov::Property<float> threshold{"threshold"};
static constexpr ov::Property<std::string> speech_model_type{"speech_model_type"};
static constexpr ov::Property<float> speed{"speed"};
static constexpr ov::Property<uint32_t> sample_rate{"sample_rate"};
static constexpr ov::Property<std::string> language{"language"};
static constexpr ov::Property<std::string> voice{"voice"};
static constexpr ov::Property<uint32_t> max_phoneme_length{"max_phoneme_length"};

}  // namespace genai
}  // namespace ov
