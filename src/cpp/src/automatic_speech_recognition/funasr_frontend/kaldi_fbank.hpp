// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace ov::genai::funasr_frontend {

constexpr size_t SAMPLING_RATE = 16000;
constexpr size_t FRAME_LENGTH = 400;
constexpr size_t FRAME_SHIFT = 160;
constexpr size_t MEL_BINS = 80;

/**
 * Computes Kaldi-compatible log-Mel filter-bank energies as used by FunASR frontends.
 *
 * Scales normalized audio to the signed 16-bit PCM range, removes each frame's DC offset, applies
 * pre-emphasis and a Hamming window, and writes MEL_BINS natural-log Mel energies per frame.
 * Frame i covers samples[i * FRAME_SHIFT, i * FRAME_SHIFT + frame_length), so samples must hold
 * (frame_count - 1) * FRAME_SHIFT + frame_length values.
 *
 * @param frame_length Samples per frame, in [2, FRAME_LENGTH].
 * @param dither Dither amplitude applied before DC-offset removal and pre-emphasis. Set to 0 to disable.
 * @param output Receives frame_count * MEL_BINS values.
 */
void compute_fbank(const float* samples, size_t frame_count, size_t frame_length, float dither, float* output);

}  // namespace ov::genai::funasr_frontend
