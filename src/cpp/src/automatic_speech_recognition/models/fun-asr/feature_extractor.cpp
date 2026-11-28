// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "feature_extractor.hpp"

#include <algorithm>

#include "openvino/core/except.hpp"

namespace ov::genai {

FunASRFeatureExtractor::FunASRFeatureExtractor(float dither) : m_dither(dither) {}

ov::Tensor FunASRFeatureExtractor::extract(const std::vector<float>& audio) const {
    OPENVINO_ASSERT(audio.size() >= 2, "Fun-ASR input audio must contain at least 2 samples");
    const size_t effective_frame_length = std::min(frame_length, audio.size());
    const size_t frame_count =
        audio.size() < effective_frame_length ? 0 : 1 + (audio.size() - effective_frame_length) / frame_shift;
    OPENVINO_ASSERT(frame_count > 0, "Fun-ASR input audio is too short to extract a frame");

    std::vector<float> fbank(frame_count * mel_bins);
    funasr_frontend::compute_fbank(audio.data(), frame_count, effective_frame_length, m_dither, fbank.data());

    const size_t lfr_frames = (frame_count + lfr_stride - 1) / lfr_stride;
    ov::Tensor result(ov::element::f32, {1, lfr_frames, mel_bins * lfr_window});
    float* output = result.data<float>();
    constexpr size_t left_context = (lfr_window - 1) / 2;
    for (size_t frame = 0; frame < lfr_frames; ++frame) {
        for (size_t context = 0; context < lfr_window; ++context) {
            const int64_t source = static_cast<int64_t>(frame * lfr_stride + context) - left_context;
            const size_t source_frame = static_cast<size_t>(std::clamp<int64_t>(source, 0, frame_count - 1));
            std::copy_n(fbank.data() + source_frame * mel_bins, mel_bins, output);
            output += mel_bins;
        }
    }
    return result;
}

}  // namespace ov::genai
