// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/visual_language/pipeline.hpp"
#include "utils.hpp"

using namespace ov::genai;

namespace ov::genai {
class ov::genai::VLMPipeline::VLMPipelineBase {
    // Load pipeline time
    float m_load_time_ms = 0;

    struct VisionProperties {
        std::vector<ov::Tensor> images;
        std::vector<ov::Tensor> videos;
        std::vector<VideoMetadata> videos_metadata;
    };

    GenerationConfig resolve_generation_config(const ov::AnyMap& config_map) {
        ov::genai::OptionalGenerationConfig optional_config = utils::get_config_from_map(config_map);
        GenerationConfig config = optional_config.value_or(get_generation_config());
        config.update_generation_config(config_map);
        return config;
    }

    static VisionProperties extract_vision_properties(const ov::AnyMap& config_map) {
        VisionProperties vision_properties;

        auto image = config_map.find(ov::genai::image.name());
        auto images = config_map.find(ov::genai::images.name());
        auto videos = config_map.find(ov::genai::videos.name());
        auto videos_metadata = config_map.find(ov::genai::videos_metadata.name());

        if (config_map.end() != image) {
            vision_properties.images = {image->second.as<ov::Tensor>()};
        }

        if (config_map.end() != images) {
            if (images->second.is<std::vector<ov::Tensor>>()) {
                auto imgs = images->second.as<std::vector<ov::Tensor>>();
                vision_properties.images.insert(vision_properties.images.end(), imgs.begin(), imgs.end());
            } else if (images->second.is<ov::Tensor>()) {
                vision_properties.images.push_back(std::move(images->second.as<ov::Tensor>()));
            } else if (!images->second.empty()) {
                OPENVINO_THROW("Unknown images type.");
            }
        }

        if (config_map.end() != videos) {
            if (videos->second.is<std::vector<ov::Tensor>>()) {
                vision_properties.videos = videos->second.as<std::vector<ov::Tensor>>();
            } else if (videos->second.is<ov::Tensor>()) {
                vision_properties.videos = {videos->second.as<ov::Tensor>()};
            } else if (!videos->second.empty()) {
                OPENVINO_THROW("Unknown videos type.");
            }
        }

        if (config_map.end() != videos_metadata) {
            if (videos_metadata->second.is<std::vector<VideoMetadata>>()) {
                vision_properties.videos_metadata = videos_metadata->second.as<std::vector<VideoMetadata>>();
            } else if (videos_metadata->second.is<VideoMetadata>()) {
                vision_properties.videos_metadata = {videos_metadata->second.as<VideoMetadata>()};
            } else if (!videos_metadata->second.empty()) {
                OPENVINO_THROW("Unknown videos metadata type.");
            }
        }

        return vision_properties;
    }

public:

    virtual ~VLMPipelineBase() = default;

    virtual VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) = 0;

    virtual VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) = 0;

    virtual VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const std::vector<VideoMetadata>& videos_metadata,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) = 0;

    VLMDecodedResults generate(
        const std::string& prompt,
        const ov::AnyMap& config_map
    ) {
        auto vision_properties = extract_vision_properties(config_map);
        GenerationConfig config = resolve_generation_config(config_map);
        return generate(
            prompt,
            vision_properties.images,
            vision_properties.videos,
            vision_properties.videos_metadata,
            config,
            utils::get_streamer_from_map(config_map)
        );
    }

    virtual VLMDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) = 0;

    virtual VLMDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) = 0;

    virtual VLMDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const std::vector<VideoMetadata>& videos_metadata,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) = 0;

    VLMDecodedResults generate(
        const ChatHistory& history,
        const ov::AnyMap& config_map
    ) {
        auto vision_properties = extract_vision_properties(config_map);
        GenerationConfig config = resolve_generation_config(config_map);
        return generate(
            history,
            vision_properties.images,
            vision_properties.videos,
            vision_properties.videos_metadata,
            config,
            utils::get_streamer_from_map(config_map)
        );
    }

    virtual void start_chat(const std::string& system_message) = 0;

    virtual void finish_chat() = 0;

    virtual Tokenizer get_tokenizer() const = 0;

    virtual void set_chat_template(const std::string& new_template) = 0;

    virtual GenerationConfig get_generation_config() const = 0;

    virtual void set_generation_config(const GenerationConfig& new_config) = 0;

    void set_load_time(float load_time_ms) {
        m_load_time_ms = load_time_ms;
    }

    float get_load_time() {
        return m_load_time_ms;
    }
};
}
