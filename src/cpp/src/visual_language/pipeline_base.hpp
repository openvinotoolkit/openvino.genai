// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/visual_language/pipeline.hpp"
#include "visual_language/vision_properties.hpp"
#include "utils.hpp"

using namespace ov::genai;

namespace ov::genai {
class ov::genai::VLMPipeline::VLMPipelineBase {
    // Load pipeline time
    float m_load_time_ms = 0;

    GenerationConfig resolve_generation_config(const ov::AnyMap& config_map) {
        ov::genai::OptionalGenerationConfig optional_config = utils::get_config_from_map(config_map);
        GenerationConfig config = optional_config.value_or(get_generation_config());
        config.update_generation_config(config_map);
        return config;
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
        const auto vision_properties = extract_vision_properties(config_map);
        GenerationConfig config = resolve_generation_config(config_map);
        return generate(
            prompt,
            vision_properties.images.value_or(std::vector<ov::Tensor>{}),
            vision_properties.videos.value_or(std::vector<ov::Tensor>{}),
            vision_properties.videos_metadata.value_or(std::vector<VideoMetadata>{}),
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
        const auto vision_properties = extract_vision_properties(config_map);
        GenerationConfig config = resolve_generation_config(config_map);
        return generate(
            history,
            vision_properties.images.value_or(std::vector<ov::Tensor>{}),
            vision_properties.videos.value_or(std::vector<ov::Tensor>{}),
            vision_properties.videos_metadata.value_or(std::vector<VideoMetadata>{}),
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
}  // namespace ov::genai
