// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/omni/talker_speech_config.hpp"

namespace ov {
namespace genai {

/// @brief Ordered list of keys recognized by update_omni_talker_speech_config().
/// Single source of truth: is_omni_talker_speech_config_key() and callers that build
/// "recognized keys" error messages both read from this instead of hand-maintaining a copy.
const std::vector<std::string>& omni_talker_speech_config_keys();

/// @brief Populate fields of `config` from an AnyMap (kwargs-style properties).
/// Recognized keys: return_audio, speaker, speaker_embedding (legacy alias),
/// audio_chunk_frames, max_new_tokens, rng_seed, talker_temperature, talker_top_k,
/// talker_repetition_penalty, cp_temperature, cp_top_k, cp_repetition_penalty.
/// Unrecognized keys are ignored — callers that share the property bag with other
/// consumers (e.g. OmniPipeline mixes in GenerationConfig keys) rely on this.
void update_omni_talker_speech_config(OmniTalkerSpeechConfig& config, const ov::AnyMap& properties);

/// @brief True if `key` is a field recognized by update_omni_talker_speech_config().
/// Talker-only property-bag entry points use this to reject typos up front, since they do
/// not share the bag with any other consumer.
bool is_omni_talker_speech_config_key(const std::string& key);

/// @brief Validate talker-only invariants on `config`.
/// Cross-config rules (e.g. return_audio vs beam search on text_config) are NOT
/// checked here — the caller (OmniPipelineImpl) handles those separately.
/// @throws ov::Exception if config is invalid.
void validate_omni_talker_speech_config(const OmniTalkerSpeechConfig& config);

}  // namespace genai
}  // namespace ov
