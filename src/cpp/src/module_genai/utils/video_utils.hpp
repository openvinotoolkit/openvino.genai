// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>
#include <vector>
namespace ov::genai::module {
class VideoMetadata {
    int total_num_frames;
    float fps = 0.0f;
    int width = 0;
    int height = 0;
    float duration = 0.0f;
    std::string video_backend;
    std::vector<int> frames_indices;
};

}  // namespace ov::genai::module