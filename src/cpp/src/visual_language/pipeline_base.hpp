// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/visual_language/pipeline.hpp"
#include "utils.hpp"

using namespace ov::genai;

namespace {

} // namespace

namespace ov::genai {
class ov::genai::VLMPipeline::VLMPipelineBase {
    // Load pipeline time
    float m_load_time_ms = 0;
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

    VLMDecodedResults generate(
        const std::string& prompt,
        const ov::AnyMap& config_map
    ) {
        auto image = config_map.find(ov::genai::image.name());
        auto images = config_map.find(ov::genai::images.name());
        auto videos = config_map.find(ov::genai::videos.name());

        ov::genai::OptionalGenerationConfig config_arg = utils::get_config_from_map(config_map);
        GenerationConfig config = (config_arg.has_value()) ? *config_arg : get_generation_config();
        config.update_generation_config(config_map);

        std::vector<ov::Tensor> images_vector = {};
        std::vector<ov::Tensor> videos_vector = {};
        if (config_map.end() != image) {
            images_vector = {image->second.as<ov::Tensor>()};
        }

        if (config_map.end() != images) {
            if (images->second.is<std::vector<ov::Tensor>>()) {
                auto imgs = images->second.as<std::vector<ov::Tensor>>();
                images_vector.insert(images_vector.end(), imgs.begin(), imgs.end());
            } else if (images->second.is<ov::Tensor>()) {
                images_vector.push_back(std::move(images->second.as<ov::Tensor>()));
            } else {
                OPENVINO_THROW("Unknown images type.");
            }
        }

        if (config_map.end() != videos) {
            if (videos->second.is<std::vector<ov::Tensor>>()) {
                videos_vector = videos->second.as<std::vector<ov::Tensor>>();
            } else if (videos->second.is<ov::Tensor>()) {
                videos_vector = {videos->second.as<ov::Tensor>()};
            } else {
                OPENVINO_THROW("Unknown videos type.");
            }
        }

        return generate(prompt, images_vector, videos_vector, config, utils::get_streamer_from_map(config_map));
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
