// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <vector>

#include "automatic_speech_recognition/pipeline_base.hpp"

namespace ov::genai {

// Nominal maximum duration of an audio chunk transcribed in one pass.
constexpr size_t MAX_ASR_INPUT_SECONDS = 1200;
// Half-width of the window around a nominal cut searched for the lowest-energy chunk boundary.
// A chunk may exceed its nominal maximum duration by up to this value.
constexpr size_t CHUNK_BOUNDARY_SEARCH_SECONDS = 5;

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

std::vector<AudioChunk> split_audio_into_chunks(const AudiosView& raw_speech,
                                                const size_t sr,
                                                const size_t max_chunk_sec);
}  // namespace ov::genai
