// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/core/any.hpp"
#include "openvino/genai/omni/talker_speech_config.hpp"

namespace ov {
namespace genai {

/// @brief Populate fields of `config` from an AnyMap (kwargs-style properties).
/// Recognized keys: return_audio, speaker, speaker_embedding (legacy alias),
/// audio_chunk_frames, max_new_tokens, rng_seed, talker_temperature, talker_top_k,
/// talker_repetition_penalty, cp_temperature, cp_top_k, cp_repetition_penalty.
void update_omni_talker_speech_config(OmniTalkerSpeechConfig& config, const ov::AnyMap& properties);

/// @brief Validate talker-only invariants on `config`.
/// Cross-config rules (e.g. return_audio vs beam search on text_config) are NOT
/// checked here — the caller (OmniPipelineImpl) handles those separately.
/// @throws ov::Exception if config is invalid.
void validate_omni_talker_speech_config(const OmniTalkerSpeechConfig& config);

}  // namespace genai
}  // namespace ov
