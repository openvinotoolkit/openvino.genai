// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

/**
 * Extracts the Kaldi-compatible log-Mel filter-bank features expected by FunASR models.
 *
 * The extractor scales normalized audio to the signed 16-bit PCM range, removes each frame's
 * DC offset, applies pre-emphasis and a Hamming window, computes 80 natural-log Mel energies,
 * and stacks seven neighboring frames with a stride of six (low-frame-rate processing).
 *
 * Unlike WhisperFeatureExtractor, this extractor does not reflect-pad the waveform or apply
 * Whisper's periodic Hann window, area-normalized Slaney Mel filters, base-10 log clipping, and
 * output normalization. Whisper keeps a [Mel bins, frames] spectrogram; FunASR returns stacked
 * features with shape [1, low-frame-rate frames, 80 * 7]. The formats are not interchangeable.
 */
class FunASRFeatureExtractor {
public:
    // The original FunASR implementation hardcodes these preprocessing parameters.
    static constexpr size_t sampling_rate = 16000;
    static constexpr size_t frame_length = 400;
    static constexpr size_t frame_shift = 160;
    static constexpr size_t fft_size = 512;
    static constexpr size_t mel_bins = 80;
    static constexpr size_t lfr_window = 7;
    static constexpr size_t lfr_stride = 6;

    /**
     * Converts mono, normalized floating-point audio into a float32 tensor of stacked
     * FunASR filter-bank features with shape [1, ceil(fbank frames / lfr_stride),
     * mel_bins * lfr_window].
     */
    ov::Tensor extract(const std::vector<float>& audio) const;
};

}  // namespace ov::genai
