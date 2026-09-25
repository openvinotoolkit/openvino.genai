// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "automatic_speech_recognition/sliding_window.hpp"

#include <algorithm>

namespace ov::genai {

size_t apply_sliding_window_drop(std::vector<float>& audio_accum,
                                  size_t already_inferred_samples,
                                  size_t chunk_size_samples,
                                  size_t window_chunk_num,
                                  size_t window_rollback_chunk_num) {
    if (window_chunk_num == 0) {
        return 0;
    }

    const size_t window_samples = window_chunk_num * chunk_size_samples;
    if (audio_accum.size() <= window_samples) {
        return 0;
    }

    const size_t keep = (window_rollback_chunk_num + 1) * chunk_size_samples;
    const size_t drop_by_window = audio_accum.size() > keep ? audio_accum.size() - keep : 0;
    // Never drop audio that hasn't been through a decode pass yet, regardless of burst size.
    const size_t drop = std::min(drop_by_window, already_inferred_samples);
    if (drop > 0) {
        audio_accum.erase(audio_accum.begin(), audio_accum.begin() + drop);
    }
    return drop;
}

}  // namespace ov::genai
