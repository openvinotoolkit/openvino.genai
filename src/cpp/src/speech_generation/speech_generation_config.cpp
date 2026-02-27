// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/speech_generation/speech_generation_config.hpp"

#include <cstdlib>
#include <ctime>

#include "utils.hpp"

namespace ov {
namespace genai {

SpeechGenerationConfig::SpeechGenerationConfig() {
    validate();
}

SpeechGenerationConfig::SpeechGenerationConfig(const std::filesystem::path& json_path)
    : GenerationConfig::GenerationConfig(json_path) {
    validate();
}

void SpeechGenerationConfig::update_generation_config(const ov::AnyMap& config_map) {
    using ov::genai::utils::read_anymap_param;

    read_anymap_param(config_map, "speech_model_type", model_type);
    read_anymap_param(config_map, "speed", speed);
    read_anymap_param(config_map, "sample_rate", sample_rate);
    read_anymap_param(config_map, "minlenratio", minlenratio);
    read_anymap_param(config_map, "maxlenratio", maxlenratio);
    read_anymap_param(config_map, "threshold", threshold);
    read_anymap_param(config_map, "language", language);
    read_anymap_param(config_map, "voice", voice);
    read_anymap_param(config_map, "max_phoneme_length", max_phoneme_length);

    GenerationConfig::update_generation_config(config_map);
}

void SpeechGenerationConfig::validate() const {
    OPENVINO_ASSERT(speed > 0.0f, "speed must be positive");
    OPENVINO_ASSERT(sample_rate > 0, "sample_rate must be positive");
    OPENVINO_ASSERT(minlenratio >= 0.0f, "minlenratio must be non-negative");
    OPENVINO_ASSERT(maxlenratio > minlenratio, "maxlenratio must be greater than minlenratio");
    OPENVINO_ASSERT(0.0f <= threshold && threshold <= 1.0f, "threshold must be in the range [0; 1]");
    OPENVINO_ASSERT(max_phoneme_length > 0, "max_phoneme_length must be positive");

    const bool model_type_empty = model_type.empty();
    const bool model_type_speecht5 = model_type == "speecht5_tts";
    const bool model_type_kokoro = model_type == "kokoro";
    OPENVINO_ASSERT(model_type_empty || model_type_speecht5 || model_type_kokoro,
                    "speech_model_type must be one of: '', speecht5_tts, kokoro");

    (void)model_type_kokoro;
}

}  // namespace genai
}  // namespace ov
