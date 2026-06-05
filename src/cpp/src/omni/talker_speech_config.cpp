// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/omni/talker_speech_config.hpp"

#include <fstream>

#include <nlohmann/json.hpp>

#include "json_utils.hpp"
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
        // Single-string speaker_id: a model with one fixed speaker. Multi-speaker
        // models expose speaker_id as an object/array — leaving `speaker` empty asks
        // the talker to pick its own default at generate time.
        speaker = talker.at("speaker_id").get<std::string>();
    }
}

void OmniTalkerSpeechConfig::update_generation_config(const ov::AnyMap& properties) {
    using ov::genai::utils::read_anymap_param;

    read_anymap_param(properties, "return_audio", return_audio);
    read_anymap_param(properties, "speaker", speaker);
    read_anymap_param(properties, "speaker_embedding", speaker_embedding);
    read_anymap_param(properties, "audio_chunk_frames", audio_chunk_frames);
    read_anymap_param(properties, "max_new_tokens", max_new_tokens);
    read_anymap_param(properties, "rng_seed", rng_seed);

    read_anymap_param(properties, "talker_temperature", talker_temperature);
    read_anymap_param(properties, "talker_top_k", talker_top_k);
    read_anymap_param(properties, "talker_repetition_penalty", talker_repetition_penalty);
    read_anymap_param(properties, "cp_temperature", cp_temperature);
    read_anymap_param(properties, "cp_top_k", cp_top_k);
    read_anymap_param(properties, "cp_repetition_penalty", cp_repetition_penalty);
}

void OmniTalkerSpeechConfig::validate() const {
    if (!return_audio) {
        // Talker disabled: every talker-only invariant is moot. Cross-config rules
        // (beam_search etc. on text_config) are not our concern — OmniPipelineImpl
        // checks those before calling validate().
        return;
    }

    if (speaker_embedding) {
        OPENVINO_ASSERT(speaker_embedding.get_element_type() == ov::element::f32,
                        "OmniTalkerSpeechConfig: speaker_embedding must be f32, got ",
                        speaker_embedding.get_element_type());
        const auto& shape = speaker_embedding.get_shape();
        OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1 && shape[1] == 1,
                        "OmniTalkerSpeechConfig: speaker_embedding must have shape "
                        "[1, 1, talker_hidden_size], got ",
                        shape);
    }
    if (talker_temperature) {
        OPENVINO_ASSERT(*talker_temperature > 0.0f,
                        "OmniTalkerSpeechConfig: talker_temperature must be > 0, got ",
                        *talker_temperature);
    }
    if (talker_top_k) {
        OPENVINO_ASSERT(*talker_top_k >= 1,
                        "OmniTalkerSpeechConfig: talker_top_k must be >= 1, got ",
                        *talker_top_k);
    }
    if (talker_repetition_penalty) {
        OPENVINO_ASSERT(*talker_repetition_penalty > 0.0f,
                        "OmniTalkerSpeechConfig: talker_repetition_penalty must be > 0, got ",
                        *talker_repetition_penalty);
    }
    if (cp_temperature) {
        OPENVINO_ASSERT(*cp_temperature > 0.0f,
                        "OmniTalkerSpeechConfig: cp_temperature must be > 0, got ",
                        *cp_temperature);
    }
    if (cp_top_k) {
        OPENVINO_ASSERT(*cp_top_k >= 1,
                        "OmniTalkerSpeechConfig: cp_top_k must be >= 1, got ",
                        *cp_top_k);
    }
    if (cp_repetition_penalty) {
        OPENVINO_ASSERT(*cp_repetition_penalty > 0.0f,
                        "OmniTalkerSpeechConfig: cp_repetition_penalty must be > 0, got ",
                        *cp_repetition_penalty);
    }
    OPENVINO_ASSERT(audio_chunk_frames >= 1,
                    "OmniTalkerSpeechConfig: audio_chunk_frames must be >= 1, got ",
                    audio_chunk_frames);
    // Sanity bound: 1ULL << 32 frames at 80 ms/frame is ~10 years of audio per chunk.
    // Anything above this means a typo — caller would wait effectively forever before
    // the first streamer callback. Reject up front.
    constexpr std::size_t kAudioChunkFramesUpperBound = 1ULL << 32;
    OPENVINO_ASSERT(audio_chunk_frames <= kAudioChunkFramesUpperBound,
                    "OmniTalkerSpeechConfig: audio_chunk_frames is unreasonably large: ",
                    audio_chunk_frames,
                    ". Max allowed: ",
                    kAudioChunkFramesUpperBound);
}

}  // namespace genai
}  // namespace ov
