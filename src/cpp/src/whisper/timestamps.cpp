// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "timestamps.hpp"

namespace ov {
namespace genai {

std::pair<std::vector<int64_t>, std::vector<ov::genai::Segment>> extract_segments(
    const std::vector<int64_t>& tokens,
    const ov::genai::WhisperGenerationConfig& config,
    const float time_precision) {
    std::vector<int64_t> non_timestamp_tokens;
    std::vector<ov::genai::Segment> segments;
    std::optional<int64_t> token_start = std::nullopt;
    size_t idx_start = 0;

    for (size_t i = 0; i < tokens.size(); i++) {
        int64_t token = tokens[i];

        bool is_timestamp = token >= config.begin_timestamps_token_id;

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
            segment.m_start = (*token_start - config.begin_timestamps_token_id) * time_precision;
            segment.m_end = (token - config.begin_timestamps_token_id) * time_precision;
            segments.push_back(segment);

            non_timestamp_tokens.insert(non_timestamp_tokens.end(), tokens.begin() + idx_start + 1, tokens.begin() + i);

            token_start = std::nullopt;
        }
    }

    // segment started but has no closing timestamp
    // add new segment only if it has non timestamps tokens
    // do not add new segment if previous segments exists
    bool has_tokens_to_add = idx_start < tokens.size() - 1;
    bool has_previous_segments = segments.size() > 0;
    if (token_start.has_value() && has_tokens_to_add && !has_previous_segments) {
        ov::genai::Segment segment;
        segment.m_tokens = {tokens.begin() + idx_start + 1, tokens.end()};
        segment.m_start = (*token_start - config.begin_timestamps_token_id) * time_precision;
        segment.m_end = -1.0f;
        segments.push_back(segment);

        non_timestamp_tokens.insert(non_timestamp_tokens.end(), tokens.begin() + idx_start + 1, tokens.end());
    }

    return {non_timestamp_tokens, segments};
}

}  // namespace genai
}  // namespace ov
