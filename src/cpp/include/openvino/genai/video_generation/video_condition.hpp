// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"

#include "openvino/genai/visibility.hpp"

namespace ov::genai {

/**
 * Frames that a generated video must contain at a given position, for LTX-2 image-to-video and video-to-video
 * generation (diffusers' LTX2VideoCondition).
 *
 * A condition at latent index 0 replaces the first frames of the video. A condition at a later index is appended to
 * the sequence as extra tokens placed at that position, and dropped before decoding.
 */
struct OPENVINO_GENAI_EXPORTS VideoCondition {
    /// u8 RGB frames [F, H, W, 3], or an image [H, W, 3]. They are resized to cover the output size and
    /// center-cropped. A video is trimmed to 8 * k + 1 frames that fit into the output.
    /// @note diffusers re-encodes single-image conditions with H.264 (CRF 33 for LTX-2.0 / 2.3, 18 for LTX-2.5)
    /// before encoding them. Apply it to the image beforehand for results that match diffusers.
    ov::Tensor frames;
    /// Latent frame index of the first frame: latent frame k > 0 starts at video frame 8 * (k - 1) + 1. Negative
    /// values count from the end, -1 is the last latent frame. A condition past the end is skipped.
    int64_t index = 0;
    /// How strongly the frames are kept: 1.0 keeps them as they are, lower values let the model change them.
    float strength = 1.0f;
};

/// Conditions for LTX-2 image-to-video and video-to-video generation.
static constexpr ov::Property<std::vector<VideoCondition>> conditions{"conditions"};

}  // namespace ov::genai
