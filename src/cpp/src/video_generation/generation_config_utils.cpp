// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "generation_config_utils.hpp"
#include <cmath>
#include "logger.hpp"
#include "utils.hpp"

namespace ov::genai::utils {

bool requests_classifier_free_guidance(const VideoGenerationConfig& config) {
    return config.guidance_scale > 1.0f || config.audio_guidance_scale.value_or(config.guidance_scale) > 1.0f;
}

void validate_generation_config(const VideoGenerationConfig& config) {
    if (!requests_classifier_free_guidance(config) && config.negative_prompt != std::nullopt) {
        GENAI_WARN("Guidance scale <= 1.0 ignores negative prompt");
    }
    if (config.decode_timestep.has_value()) {
        std::visit([](const auto& values) {
            using Value = std::decay_t<decltype(values)>;
            if constexpr (std::is_same_v<Value, float>) {
                OPENVINO_ASSERT(std::isfinite(values), "decode_timestep must be finite");
            } else {
                OPENVINO_ASSERT(!values.empty(), "decode_timestep list must not be empty");
                for (const float value : values) {
                    OPENVINO_ASSERT(std::isfinite(value), "decode_timestep must be finite");
                }
            }
        }, *config.decode_timestep);
    }
    OPENVINO_ASSERT(!config.decode_noise_scale.has_value() || std::isfinite(*config.decode_noise_scale),
                    "decode_noise_scale must be finite");
}

void resolve_negative_prompt(VideoGenerationConfig& config) {
    if (!requests_classifier_free_guidance(config)) {
        config.negative_prompt = std::nullopt;
    }
}

void update_generation_config(VideoGenerationConfig& config, const ov::AnyMap& properties) {
    using ov::genai::utils::read_anymap_param;

    // override whole generation config first
    read_anymap_param(properties, VIDEO_GENERATION_CONFIG, config);

    read_anymap_param(properties, "guidance_rescale", config.guidance_rescale);
    read_anymap_param(properties, "num_frames", config.num_frames);
    read_anymap_param(properties, "frame_rate", config.frame_rate);
    read_anymap_param(properties, "audio_guidance_scale", config.audio_guidance_scale);
    if (const auto it = properties.find("decode_timestep"); it != properties.end()) {
        OPENVINO_ASSERT(!it->second.empty(), "decode_timestep must not be empty");
        if (it->second.is<std::variant<float, std::vector<float>>>()) {
            config.decode_timestep = it->second.as<std::variant<float, std::vector<float>>>();
        } else if (it->second.is<float>()) {
            config.decode_timestep = it->second.as<float>();
        } else if (it->second.is<int64_t>()) {
            config.decode_timestep = static_cast<float>(it->second.as<int64_t>());
        } else if (it->second.is<std::vector<double>>()) {
            const auto& values = it->second.as<std::vector<double>>();
            std::vector<float> timesteps;
            timesteps.reserve(values.size());
            for (const double value : values) {
                timesteps.push_back(static_cast<float>(value));
            }
            config.decode_timestep = std::move(timesteps);
        } else {
            config.decode_timestep = it->second.as<std::vector<float>>();
        }
    }
    read_anymap_param(properties, "decode_noise_scale", config.decode_noise_scale);
    read_anymap_param(properties, "stg_scale", config.stg_scale);
    read_anymap_param(properties, "audio_stg_scale", config.audio_stg_scale);
    read_anymap_param(properties, "modality_scale", config.modality_scale);
    read_anymap_param(properties, "audio_modality_scale", config.audio_modality_scale);
    read_anymap_param(properties, "audio_guidance_rescale", config.audio_guidance_rescale);
    read_anymap_param(properties, "spatio_temporal_guidance_blocks", config.spatio_temporal_guidance_blocks);
    read_anymap_param(properties, "num_videos_per_prompt", config.num_videos_per_prompt);

    read_anymap_param(properties, "negative_prompt", config.negative_prompt);
    read_anymap_param(properties, "guidance_scale", config.guidance_scale);
    read_anymap_param(properties, "height", config.height);
    read_anymap_param(properties, "width", config.width);
    read_anymap_param(properties, "num_inference_steps", config.num_inference_steps);
    read_anymap_param(properties, "max_sequence_length", config.max_sequence_length);
    read_anymap_param(properties, "taylorseer_config", config.taylorseer_config);

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

    // Negative-prompt handling needs the resolved defaults, so it happens in 'merge_generation_config'
}

std::pair<std::string, ov::Any> generation_config(const VideoGenerationConfig& generation_config) {
    return {VIDEO_GENERATION_CONFIG, ov::Any::make<VideoGenerationConfig>(generation_config)};
}

} // namespace ov::genai::utils
