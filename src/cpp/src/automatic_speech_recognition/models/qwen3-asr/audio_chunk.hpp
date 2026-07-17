// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <vector>

namespace ov::genai {

struct AudioChunk {
    /**
     * One chunk cut from an original audio.
     *
     * Attributes:
     *     orig_batch: Index of the original sample in the input batch.
     *     chunk_index: Index of this chunk within the original sample.
     *     wav: Mono float32 waveform.
     *     sr: Sampling rate.
     *     offset_sec: Start offset of this chunk in the original audio, in seconds.
     */
    size_t orig_batch;
    size_t chunk_index;
    std::vector<float> wav;
    size_t sr;
    float offset_sec;
};

std::vector<AudioChunk> split_audio_into_chunks(const std::vector<std::vector<float>>& raw_speech,
                                                const size_t sr,
                                                const size_t max_chunk_sec);
}  // namespace ov::genai
