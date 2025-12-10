// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/generation_config.hpp"
#include "utils.hpp"

using namespace ov::genai;

void VideoGenerationConfig::validate() const {
    ImageGenerationConfig::validate();
}

void VideoGenerationConfig::update_generation_config(const ov::AnyMap& properties) {
    ImageGenerationConfig::update_generation_config(properties);
    using ov::genai::utils::read_anymap_param;
    read_anymap_param(properties, "guidance_rescale", guidance_rescale);
    read_anymap_param(properties, "num_frames", num_frames);
    read_anymap_param(properties, "frame_rate", frame_rate);
    read_anymap_param(properties, "num_videos_per_prompt", num_videos_per_prompt);
}
