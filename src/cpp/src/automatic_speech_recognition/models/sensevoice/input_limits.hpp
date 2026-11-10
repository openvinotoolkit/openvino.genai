// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <limits>

namespace ov::genai {

// Constraints of the compiled model inputs. Default values mean no constraint.
struct SenseVoiceInputLimits {
    size_t max_batch = std::numeric_limits<size_t>::max();
    // Features are zero-padded to at least this number of frames.
    size_t padded_frames = 0;
    size_t max_audio_samples = std::numeric_limits<size_t>::max();
};

}  // namespace ov::genai
