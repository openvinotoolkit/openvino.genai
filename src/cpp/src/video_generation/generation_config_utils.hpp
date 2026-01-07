// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/generation_config.hpp"

namespace ov::genai::utils {

static constexpr char VIDEO_GENERATION_CONFIG[] = "VIDEO_GENERATION_CONFIG";

void validate_generation_config(const VideoGenerationConfig& config);

void update_generation_config(VideoGenerationConfig& config, const ov::AnyMap& properties);

std::pair<std::string, ov::Any> generation_config(const VideoGenerationConfig& generation_config);

} // namespace ov::genai::utils
