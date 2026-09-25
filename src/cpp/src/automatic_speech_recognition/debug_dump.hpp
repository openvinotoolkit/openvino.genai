// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ov::genai {

/// Whether OPENVINO_GENAI_ASR_DEBUG_DIR is set. Checked once and cached; set the env var before
/// the process starts (it is not re-read after the first call).
bool asr_debug_dump_enabled();

/// Writes the exact encoder input and decoder text prefix used for one decode pass, so a
/// streaming session's chunk-by-chunk behavior can be inspected offline. No-op if
/// asr_debug_dump_enabled() is false. Writes, under OPENVINO_GENAI_ASR_DEBUG_DIR:
///   <model_tag>_chunk_<chunk_index>_audio.wav   — mono PCM16 @ sample_rate, the encoder input
///   <model_tag>_chunk_<chunk_index>_prefix.txt  — decoder_prefix_text verbatim (may be empty)
void asr_debug_dump_chunk(const std::string& model_tag,
                          size_t chunk_index,
                          const std::vector<float>& audio,
                          size_t sample_rate,
                          const std::string& decoder_prefix_text);

}  // namespace ov::genai
