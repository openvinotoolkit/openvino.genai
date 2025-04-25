// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "timestamps.hpp"

namespace ov {
namespace genai {

ov::genai::ExtractedSegments extract_segments(const std::vector<int64_t>& tokens,
                                              const ov::genai::WhisperGenerationConfig& config,
                                              const size_t nb_max_frames,
                                              const float time_precision,
                                              const float time_offset) {
    ov::genai::ExtractedSegments extracted_segments;
    std::optional<int64_t> token_start = std::nullopt;
    const size_t timestamp_begin = config.no_timestamps_token_id + 1;
    size_t idx_start = 0;

    for (size_t i = 0; i < tokens.size(); i++) {
        int64_t token = tokens[i];

        bool is_timestamp = token >= timestamp_begin;

        if (!is_timestamp) {
            continue;
        }

        if (!token_start.has_value()) {
            token_start = token;
            idx_start = i;
        } else {
            if (token_start == token) {
                // from HF:
                // https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/tokenization_whisper.py#L1020
                // This is a bug in timestamp token output where we're taking the duplicate token as a stop where it
                // should be a start. This is an issue in the underlying model output. Let's just skip it so it becomes
                // de-factor a start again.
                continue;
            }

            ov::genai::Segment segment;
            segment.m_tokens = {tokens.begin() + idx_start + 1, tokens.begin() + i};
            segment.m_start = (*token_start - timestamp_begin) * time_precision + time_offset;
            segment.m_end = (token - timestamp_begin) * time_precision + time_offset;
            extracted_segments.segments.push_back(segment);

            // each next timestamp token represents .02 time diff
            extracted_segments.last_offset = (token - timestamp_begin) * 2;

            extracted_segments.non_timestamp_tokens.insert(extracted_segments.non_timestamp_tokens.end(),
                                                           tokens.begin() + idx_start + 1,
                                                           tokens.begin() + i);
            extracted_segments.segment_ranges.emplace_back(idx_start + 1, i);
            token_start = std::nullopt;
        }
    }

    // segment started but has no closing timestamp
    // add new segment only if it has non timestamps tokens
    // do not add new segment if previous segments exists
    bool has_tokens_to_add = idx_start < tokens.size() - 1;
    bool has_previous_segments = extracted_segments.segments.size() > 0;
    if (token_start.has_value() && has_tokens_to_add && !has_previous_segments) {
        ov::genai::Segment segment;
        segment.m_tokens = {tokens.begin() + idx_start + 1, tokens.end()};
        segment.m_start = (*token_start - timestamp_begin) * time_precision + time_offset;
        segment.m_end = -1.0f;
        extracted_segments.segments.push_back(segment);

        extracted_segments.last_offset = nb_max_frames;

        extracted_segments.non_timestamp_tokens.insert(extracted_segments.non_timestamp_tokens.end(),
                                                       tokens.begin() + idx_start + 1,
                                                       tokens.end());

        extracted_segments.segment_ranges.emplace_back(idx_start + 1, tokens.size());
    }

    // last timestamps generated in pairs <ts><ts><eos> -> speech segment continuation to the next chunk -> token_start
    // will have value single ending timestamp <ts><eos> -> no more speech till the end of current chunk -> set offset
    // to the end of frame
    if (!token_start.has_value()) {
        extracted_segments.last_offset = nb_max_frames;
    }

    return extracted_segments;
}

}  // namespace genai
}  // namespace ov
