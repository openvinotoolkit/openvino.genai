// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/speech_generation/speech_generation_config.hpp"

#include <cstdlib>
#include <ctime>
#include <fstream>

#include "json_utils.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

SpeechGenerationConfig::SpeechGenerationConfig() {
    validate();
}

SpeechGenerationConfig::SpeechGenerationConfig(const std::filesystem::path& json_path)
    : GenerationConfig::GenerationConfig(json_path) {
    using ov::genai::utils::read_json_param;
    std::ifstream ifs(json_path);
    if (ifs.good()) {
        const nlohmann::json data = nlohmann::json::parse(ifs);
        read_json_param(data, "speed", speed);
        read_json_param(data, "minlenratio", minlenratio);
        read_json_param(data, "maxlenratio", maxlenratio);
        read_json_param(data, "threshold", threshold);
        read_json_param(data, "language", language);
        read_json_param(data, "max_phoneme_length", max_phoneme_length);
        read_json_param(data, "phonemize_fallback_model_dir", phonemize_fallback_model_dir);

        read_json_param(data, "speaker", speaker);
        read_json_param(data, "instruct", instruct);
        read_json_param(data, "non_streaming_mode", non_streaming_mode);
        read_json_param(data, "subtalker_dosample", subtalker_dosample);
        read_json_param(data, "subtalker_top_k", subtalker_top_k);
        read_json_param(data, "subtalker_top_p", subtalker_top_p);
        read_json_param(data, "subtalker_temperature", subtalker_temperature);
        read_json_param(data, "voice_clone_ref_text", voice_clone_ref_text);
    }
    validate();
}

void SpeechGenerationConfig::update_generation_config(const ov::AnyMap& config_map) {
    using ov::genai::utils::read_anymap_param;

    read_anymap_param(config_map, "speed", speed);
    read_anymap_param(config_map, "minlenratio", minlenratio);
    read_anymap_param(config_map, "maxlenratio", maxlenratio);
    read_anymap_param(config_map, "threshold", threshold);
    read_anymap_param(config_map, "language", language);
    read_anymap_param(config_map, "max_phoneme_length", max_phoneme_length);
    read_anymap_param(config_map, "phonemize_fallback_model_dir", phonemize_fallback_model_dir);

    read_anymap_param(config_map, "speaker", speaker);
    read_anymap_param(config_map, "instruct", instruct);
    read_anymap_param(config_map, "non_streaming_mode", non_streaming_mode);
    read_anymap_param(config_map, "subtalker_dosample", subtalker_dosample);
    read_anymap_param(config_map, "subtalker_top_k", subtalker_top_k);
    read_anymap_param(config_map, "subtalker_top_p", subtalker_top_p);
    read_anymap_param(config_map, "subtalker_temperature", subtalker_temperature);
    read_anymap_param(config_map, "voice_clone_ref_text", voice_clone_ref_text);
    read_anymap_param(config_map, "voice_clone_ref_audio", voice_clone_ref_audio);
    read_anymap_param(config_map, "voice_clone_ref_codec_ids", voice_clone_ref_codec_ids);

    GenerationConfig::update_generation_config(config_map);
}

void SpeechGenerationConfig::validate() const {
    OPENVINO_ASSERT(speed > 0.0f, "speed must be positive");
    OPENVINO_ASSERT(minlenratio >= 0.0f, "minlenratio must be non-negative");
    OPENVINO_ASSERT(maxlenratio > minlenratio, "maxlenratio must be greater than minlenratio");
    OPENVINO_ASSERT(0.0f <= threshold && threshold <= 1.0f, "threshold must be in the range [0; 1]");
    OPENVINO_ASSERT(max_phoneme_length > 0, "max_phoneme_length must be positive");
    OPENVINO_ASSERT(!phonemize_fallback_model_dir.has_value() || !phonemize_fallback_model_dir->empty(),
                    "phonemize_fallback_model_dir must be unset or a non-empty path");
    if (subtalker_dosample) {
        OPENVINO_ASSERT(subtalker_top_k > 0, "subtalker_top_k must be positive");
        OPENVINO_ASSERT(0.0f <= subtalker_top_p && subtalker_top_p <= 1.0f,
                        "subtalker_top_p must be in the range [0; 1]");
        OPENVINO_ASSERT(subtalker_temperature > 0.0f, "subtalker_temperature must be positive");
    }
}

}  // namespace genai
}  // namespace ov
