// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/video_generation/generation_config.hpp"
#include "utils.hpp"

using namespace ov::genai;

void VideoGenerationConfig::update_generation_config(const ov::AnyMap& properties) {
    using ov::genai::utils::read_anymap_param;
    read_anymap_param(properties, "guidance_rescale", guidance_rescale);
    read_anymap_param(properties, "num_frames", num_frames);
    read_anymap_param(properties, "frame_rate", frame_rate);
    read_anymap_param(properties, "num_videos_per_prompt", num_videos_per_prompt);

    read_anymap_param(properties, "negative_prompt", negative_prompt);
    read_anymap_param(properties, "guidance_scale", guidance_scale);
    read_anymap_param(properties, "height", height);
    read_anymap_param(properties, "width", width);
    read_anymap_param(properties, "num_inference_steps", num_inference_steps);
    read_anymap_param(properties, "max_sequence_length", max_sequence_length);

    // 'generator' has higher priority than 'seed' parameter
    const bool have_generator_param = properties.find(ov::genai::generator.name()) != properties.end();
    if (have_generator_param) {
        read_anymap_param(properties, "generator", generator);
    } else {
        read_anymap_param(properties, "rng_seed", rng_seed);

        // initialize random generator with a given seed value
        if (!generator) {
            generator = std::make_shared<CppStdGenerator>(rng_seed);
        }

        const bool have_rng_seed = properties.find(ov::genai::rng_seed.name()) != properties.end();
        if (have_rng_seed) {
            // we need to change seed as an user have specified it manually
            generator->seed(rng_seed);
        }
    }

    validate();
}

void VideoGenerationConfig::validate() const {
    OPENVINO_ASSERT(guidance_scale > 1.0f || negative_prompt == std::nullopt, "Guidance scale <= 1.0 ignores negative prompt");
}
