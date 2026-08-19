// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/genai/visual_language/video_metadata.hpp>
#include <openvino/runtime/tensor.hpp>

#include <filesystem>
#include <utility>
#include <vector>

namespace utils {

std::pair<ov::Tensor, ov::genai::VideoMetadata> load_video(const std::filesystem::path& video_path,
                                                           size_t num_frames = 8);

std::pair<std::vector<ov::Tensor>, std::vector<ov::genai::VideoMetadata>> load_videos(
    const std::filesystem::path& input_path,
    size_t num_frames = 8);

}  // namespace utils
