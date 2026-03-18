// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ov::genai {

/**
 * @brief Metadata describing the original video source. Controls video frames sampling before encoding.
 */
struct OPENVINO_GENAI_EXPORTS VideoMetadata {
    /// Total number of frames in the original video before any sampling. 0 means unknown.
    size_t total_num_frames = 0;
    /// Frame rate of the original video in frames per second. 0 means unknown.
    float fps = 0.0f;
    /// Indices of frames to sample from the provided video tensor.
    /// When empty (default), model-specific sampling is applied if defined, otherwise all frames are processed.
    /// When non-empty, only the specified frames are extracted and model-specific sampling logic is skipped (if any).
    std::vector<size_t> frames_indices;
};

} // namespace ov::genai
