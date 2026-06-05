// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/omni/speech_generation_config.hpp"

#include <fstream>

#include <nlohmann/json.hpp>

#include "json_utils.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

OmniSpeechGenerationConfig::OmniSpeechGenerationConfig(const std::filesystem::path& models_dir)
    : GenerationConfig::GenerationConfig(models_dir / "generation_config.json") {
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
        // Single-string speaker_id: a model with one fixed speaker. Multi-speaker models
        // expose speaker_id as an object/array — leaving `speaker` empty asks the talker
        // to pick its own default at generate time.
        speaker = talker.at("speaker_id").get<std::string>();
    }
}

void OmniSpeechGenerationConfig::update_generation_config(const ov::AnyMap& properties) {
    using ov::genai::utils::read_anymap_param;

    read_anymap_param(properties, "return_audio", return_audio);
    read_anymap_param(properties, "speaker", speaker);
    read_anymap_param(properties, "audio_chunk_frames", audio_chunk_frames);

    GenerationConfig::update_generation_config(properties);
}

void OmniSpeechGenerationConfig::validate() const {
    // Omni-specific invariants run before delegating to the base so that callers see
    // the speech-output failure mode directly, even when other stop-condition defaults
    // would otherwise trip the base validator first.
    if (return_audio) {
        OPENVINO_ASSERT(audio_chunk_frames >= 1,
                        "OmniSpeechGenerationConfig: audio_chunk_frames must be >= 1, got ",
                        audio_chunk_frames);
        // Sanity bound: 1ULL << 32 frames at 80 ms/frame is ~10 years of audio per chunk.
        // Anything above this means a typo — caller would wait effectively forever before
        // the first streamer callback. Reject up front rather than silently accepting it.
        constexpr size_t kAudioChunkFramesUpperBound = 1ULL << 32;
        OPENVINO_ASSERT(audio_chunk_frames <= kAudioChunkFramesUpperBound,
                        "OmniSpeechGenerationConfig: audio_chunk_frames is unreasonably large: ",
                        audio_chunk_frames,
                        ". Max allowed: ",
                        kAudioChunkFramesUpperBound);
        OPENVINO_ASSERT(!is_beam_search(),
                        "OmniSpeechGenerationConfig: return_audio is not compatible with beam search "
                        "(num_beams > 1). The talker consumes a single hidden-state stream.");
        OPENVINO_ASSERT(!is_prompt_lookup() && !is_assisting_generation(),
                        "OmniSpeechGenerationConfig: return_audio is not compatible with prompt lookup "
                        "or assistant/speculative decoding.");
    }

    GenerationConfig::validate();
}

}  // namespace genai
}  // namespace ov
