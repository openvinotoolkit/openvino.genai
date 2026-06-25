// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_chunk.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>

#include "openvino/core/except.hpp"

namespace {
// Find a low-energy cut point near `start + max_len` to avoid splitting mid-speech.
//
// Algorithm:
//   1. Compute the nominal cut at `start + max_len`.
//   2. Open a search window of ±expand samples around that cut (clamped to [start, total_len]).
//   3. If the window is wider than one sliding-window frame:
//      a. Compute the absolute amplitude of every sample in the window.
//      b. Slide a `win`-wide frame across it and find the frame with the minimum
//         total energy (using a running sum to keep the scan O(N)).
//      c. Within that lowest-energy frame, find the single sample with the
//         smallest absolute amplitude — that sample becomes the boundary.
//   4. Clamp the boundary so it is at least `start + 1` and at most `total_len`.
//
// When the search window is too narrow for even one frame (rare: audio shorter than
// 2 * expand), fall back to the nominal cut without any energy search.
size_t find_chunk_boundary(const std::vector<float>& raw_speech,
                           const size_t start,
                           const size_t max_len,
                           const size_t expand,
                           const size_t win) {
    const size_t total_len = raw_speech.size();
    const size_t cut = start + max_len;
    const size_t left = std::max(start, cut > expand ? cut - expand : size_t{0});
    const size_t right = std::min(total_len, cut + expand);
    size_t boundary = cut;

    if ((right - left) > win) {
        const auto seg_begin = raw_speech.begin() + left;
        const auto seg_end = raw_speech.begin() + right;
        std::vector<float> seg_abs;
        seg_abs.reserve(right - left);
        std::transform(seg_begin, seg_end, std::back_inserter(seg_abs), [](float sample) {
            return std::abs(sample);
        });

        // Sliding-window energy scan: maintain a running sum for O(N) complexity.
        const size_t valid_windows = seg_abs.size() - win + 1;
        float window_sum = std::accumulate(seg_abs.begin(), seg_abs.begin() + win, 0.0f);
        float min_window_sum = window_sum;
        size_t min_pos = 0;

        for (size_t pos = 1; pos < valid_windows; ++pos) {
            window_sum += seg_abs[pos + win - 1] - seg_abs[pos - 1];
            if (window_sum < min_window_sum) {
                min_window_sum = window_sum;
                min_pos = pos;
            }
        }

        // Within the quietest frame, pick the single sample with the lowest amplitude.
        const auto local_begin = seg_abs.begin() + min_pos;
        const auto local_end = local_begin + win;
        const auto inner_it = std::min_element(local_begin, local_end);
        // std::distance returns ptrdiff_t; narrowing is safe because inner_it >= local_begin.
        const size_t inner = static_cast<size_t>(std::distance(local_begin, inner_it));
        boundary = left + min_pos + inner;
    }

    boundary = std::max(boundary, start + 1);
    return std::min(boundary, total_len);
}

ov::genai::AudioChunk make_audio_chunk(const std::vector<float>& raw_speech,
                                       const size_t start,
                                       const size_t end,
                                       const size_t sr,
                                       const size_t min_chunk_len,
                                       const size_t chunk_index,
                                       const float offset_sec) {
    std::vector<float> wav(raw_speech.begin() + start, raw_speech.begin() + end);
    if (wav.size() < min_chunk_len) {
        wav.resize(min_chunk_len, 0.0f);
    }

    ov::genai::AudioChunk chunk_info{};
    chunk_info.orig_batch = 0;
    chunk_info.chunk_index = chunk_index;
    chunk_info.wav = std::move(wav);
    chunk_info.sr = sr;
    chunk_info.offset_sec = offset_sec;
    return chunk_info;
}

std::vector<ov::genai::AudioChunk> split_audio_into_chunks(const std::vector<float>& raw_speech,
                                                           const size_t sr,
                                                           const size_t max_chunk_sec) {
    OPENVINO_ASSERT(sr > 0, "Sampling rate must be greater than 0.");
    OPENVINO_ASSERT(max_chunk_sec > 0, "max_chunk_sec must be greater than 0.");

    constexpr float SEARCH_EXPAND_SEC = 5.0f;
    constexpr float MIN_WINDOW_MS = 100.0f;
    constexpr float MIN_ASR_INPUT_SECONDS = 0.5f;

    const size_t max_len = max_chunk_sec * sr;
    // std::round returns float; outer static_cast<size_t> truncates to sample count.
    const size_t expand = static_cast<size_t>(std::round(SEARCH_EXPAND_SEC * static_cast<float>(sr)));
    const size_t min_chunk_len = static_cast<size_t>(std::round(MIN_ASR_INPUT_SECONDS * static_cast<float>(sr)));
    const size_t win =
        std::max<size_t>(4, static_cast<size_t>(std::round((MIN_WINDOW_MS / 1000.0f) * static_cast<float>(sr))));

    std::vector<ov::genai::AudioChunk> chunks;
    const size_t total_len = raw_speech.size();

    if (total_len <= max_len) {
        chunks.push_back(make_audio_chunk(raw_speech, 0, total_len, sr, min_chunk_len, 0, 0.0f));
        return chunks;
    }

    size_t start = 0;
    float offset_sec = 0.0f;
    size_t chunk_index = 0;

    while ((total_len - start) > max_len) {
        const size_t boundary = find_chunk_boundary(raw_speech, start, max_len, expand, win);
        chunks.push_back(make_audio_chunk(raw_speech, start, boundary, sr, min_chunk_len, chunk_index, offset_sec));

        // Cast both operands to float to get floating-point division instead of integer division.
        offset_sec += static_cast<float>(boundary - start) / static_cast<float>(sr);
        start = boundary;
        ++chunk_index;
    }

    chunks.push_back(make_audio_chunk(raw_speech, start, total_len, sr, min_chunk_len, chunk_index, offset_sec));

    return chunks;
}
}  // namespace

namespace ov::genai {

std::vector<AudioChunk> split_audio_into_chunks(const std::vector<std::vector<float>>& raw_speech,
                                                const size_t sr,
                                                const size_t max_chunk_sec) {
    std::vector<AudioChunk> chunks;
    for (size_t batch = 0; batch < raw_speech.size(); ++batch) {
        std::vector<AudioChunk> sample_chunks = ::split_audio_into_chunks(raw_speech[batch], sr, max_chunk_sec);
        for (AudioChunk& chunk : sample_chunks) {
            chunk.orig_batch = batch;
            chunks.push_back(std::move(chunk));
        }
    }

    return chunks;
}
}  // namespace ov::genai
