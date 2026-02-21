// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "generation_config_utils.hpp"
#include "utils.hpp"

namespace ov::genai::utils {

void validate_generation_config(const VideoGenerationConfig& config) {
    OPENVINO_ASSERT(config.guidance_scale > 1.0f || config.negative_prompt == std::nullopt,
                    "Guidance scale <= 1.0 ignores negative prompt");
}

void update_generation_config(VideoGenerationConfig& config, const ov::AnyMap& properties) {
    using ov::genai::utils::read_anymap_param;

    // override whole generation config first
    read_anymap_param(properties, VIDEO_GENERATION_CONFIG, config);

    read_anymap_param(properties, "guidance_rescale", config.guidance_rescale);
    read_anymap_param(properties, "num_frames", config.num_frames);
    read_anymap_param(properties, "frame_rate", config.frame_rate);
    read_anymap_param(properties, "num_videos_per_prompt", config.num_videos_per_prompt);

    read_anymap_param(properties, "negative_prompt", config.negative_prompt);
    read_anymap_param(properties, "guidance_scale", config.guidance_scale);
    read_anymap_param(properties, "height", config.height);
    read_anymap_param(properties, "width", config.width);
    read_anymap_param(properties, "num_inference_steps", config.num_inference_steps);
    read_anymap_param(properties, "max_sequence_length", config.max_sequence_length);

    read_anymap_param(properties, "adapters", config.adapters);

    // 'generator' has higher priority than 'seed' parameter
    const bool have_generator_param =
        properties.find(ov::genai::generator.name()) != properties.end();

    if (have_generator_param) {
        read_anymap_param(properties, "generator", config.generator);
    } else {
        // initialize random generator with a default seed value
        if (!config.generator) {
            config.generator = std::make_shared<CppStdGenerator>(42);
        }
    }

    validate_generation_config(config);
}

std::pair<std::string, ov::Any> generation_config(const VideoGenerationConfig& generation_config) {
    return {VIDEO_GENERATION_CONFIG, ov::Any::make<VideoGenerationConfig>(generation_config)};
}

} // namespace ov::genai::utils
