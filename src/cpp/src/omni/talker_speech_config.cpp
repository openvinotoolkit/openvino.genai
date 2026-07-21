// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/omni/talker_speech_config.hpp"

#include <fstream>
#include <variant>

#include <nlohmann/json.hpp>

#include "omni/talker_speech_config_utils.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

OmniTalkerSpeechConfig::OmniTalkerSpeechConfig(const std::filesystem::path& models_dir) {
    const auto config_path = models_dir / "config.json";
    if (!std::filesystem::exists(config_path)) {
        return;
    }

    std::ifstream stream(config_path);
    if (!stream.is_open()) {
        return;
    }

    const nlohmann::json parsed = nlohmann::json::parse(stream);
    if (!parsed.contains("talker_config")) {
        return;
    }

    const auto& talker = parsed.at("talker_config");
    if (talker.contains("speaker_id") && talker.at("speaker_id").is_string()) {
        speaker = talker.at("speaker_id").get<std::string>();
    }
}

void update_omni_talker_speech_config(OmniTalkerSpeechConfig& config, const ov::AnyMap& properties) {
    using ov::genai::utils::read_anymap_param;

    read_anymap_param(properties, "return_audio", config.return_audio);
    if (properties.count("speaker_embedding")) {
        ov::Tensor tensor;
        read_anymap_param(properties, "speaker_embedding", tensor);
        config.speaker = tensor;
    }
    if (properties.count("speaker")) {
        const auto& val = properties.at("speaker");
        if (val.is<std::string>()) {
            config.speaker = val.as<std::string>();
        } else if (val.is<ov::Tensor>()) {
            config.speaker = val.as<ov::Tensor>();
        }
    }

    read_anymap_param(properties, "audio_chunk_frames", config.audio_chunk_frames);
    read_anymap_param(properties, "max_new_tokens", config.max_new_tokens);
    read_anymap_param(properties, "rng_seed", config.rng_seed);

    read_anymap_param(properties, "talker_temperature", config.talker_temperature);
    read_anymap_param(properties, "talker_top_k", config.talker_top_k);
    read_anymap_param(properties, "talker_repetition_penalty", config.talker_repetition_penalty);
    read_anymap_param(properties, "cp_temperature", config.cp_temperature);
    read_anymap_param(properties, "cp_top_k", config.cp_top_k);
    read_anymap_param(properties, "cp_repetition_penalty", config.cp_repetition_penalty);
}

void validate_omni_talker_speech_config(const OmniTalkerSpeechConfig& config) {
    if (!config.return_audio) {
        return;
    }

    if (std::holds_alternative<ov::Tensor>(config.speaker)) {
        const auto& tensor = std::get<ov::Tensor>(config.speaker);
        OPENVINO_ASSERT(static_cast<bool>(tensor),
                        "OmniTalkerSpeechConfig: speaker Tensor variant is set but the tensor is empty");
        OPENVINO_ASSERT(tensor.get_element_type() == ov::element::f32,
                        "OmniTalkerSpeechConfig: speaker embedding must be f32, got ",
                        tensor.get_element_type());
        const auto& shape = tensor.get_shape();
        OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1 && shape[1] == 1,
                        "OmniTalkerSpeechConfig: speaker embedding must have shape "
                        "[1, 1, talker_hidden_size], got ",
                        shape);
    }
    if (config.talker_temperature) {
        OPENVINO_ASSERT(*config.talker_temperature > 0.0f,
                        "OmniTalkerSpeechConfig: talker_temperature must be > 0, got ",
                        *config.talker_temperature);
    }
    if (config.talker_top_k) {
        OPENVINO_ASSERT(*config.talker_top_k >= 1,
                        "OmniTalkerSpeechConfig: talker_top_k must be >= 1, got ",
                        *config.talker_top_k);
    }
    if (config.talker_repetition_penalty) {
        OPENVINO_ASSERT(*config.talker_repetition_penalty > 0.0f,
                        "OmniTalkerSpeechConfig: talker_repetition_penalty must be > 0, got ",
                        *config.talker_repetition_penalty);
    }
    if (config.cp_temperature) {
        OPENVINO_ASSERT(*config.cp_temperature > 0.0f,
                        "OmniTalkerSpeechConfig: cp_temperature must be > 0, got ",
                        *config.cp_temperature);
    }
    if (config.cp_top_k) {
        OPENVINO_ASSERT(*config.cp_top_k >= 1,
                        "OmniTalkerSpeechConfig: cp_top_k must be >= 1, got ",
                        *config.cp_top_k);
    }
    if (config.cp_repetition_penalty) {
        OPENVINO_ASSERT(*config.cp_repetition_penalty > 0.0f,
                        "OmniTalkerSpeechConfig: cp_repetition_penalty must be > 0, got ",
                        *config.cp_repetition_penalty);
    }
    OPENVINO_ASSERT(config.audio_chunk_frames >= 1,
                    "OmniTalkerSpeechConfig: audio_chunk_frames must be >= 1, got ",
                    config.audio_chunk_frames);
    constexpr std::size_t kAudioChunkFramesUpperBound = 4096;
    OPENVINO_ASSERT(config.audio_chunk_frames <= kAudioChunkFramesUpperBound,
                    "OmniTalkerSpeechConfig: audio_chunk_frames is unreasonably large: ",
                    config.audio_chunk_frames,
                    ". Max allowed: ",
                    kAudioChunkFramesUpperBound);
}

}  // namespace genai
}  // namespace ov
