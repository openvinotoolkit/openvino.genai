// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/speech_generation/audio_streamer_base.hpp"

namespace ov::genai {

/// @brief Backward-compatible alias for AudioStreamerBase used by OmniPipeline / Talker.
/// @note This is a preview API and is subject to change.
using OmniSpeechStreamerBase = AudioStreamerBase;

/// @brief Backward-compatible alias for AudioStreamerVariant used by OmniPipeline / Talker.
using OmniSpeechStreamerVariant = AudioStreamerVariant;

}  // namespace ov::genai
