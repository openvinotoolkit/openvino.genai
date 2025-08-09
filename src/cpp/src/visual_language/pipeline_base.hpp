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
        const std::vector<ov::Tensor>& rgbs,
        const std::vector<ov::Tensor>& video,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) = 0;

    VLMDecodedResults generate(
        const std::string& prompt,
        const ov::AnyMap& config_map
    ) {
        auto image = config_map.find(ov::genai::image.name());
        auto images = config_map.find(ov::genai::images.name());
        auto video = config_map.find(ov::genai::video.name());
        OPENVINO_ASSERT(
            config_map.end() == image || config_map.end() == images,
            "Only one property can be set: image of images."
        );
        std::vector<ov::Tensor> rgbs;
        if (config_map.end() != image) {
            rgbs = {image->second.as<ov::Tensor>()};
        } if (config_map.end() != images) {
            if (images->second.is<std::vector<ov::Tensor>>()) {
                rgbs = images->second.as<std::vector<ov::Tensor>>();
            }
            else if (images->second.is<ov::Tensor>()){
                rgbs = {images->second.as<ov::Tensor>()};
            }
            else {
                OPENVINO_THROW("Unknown images type.");
            }
        }

        std::vector<ov::Tensor> video_rgbs;
        if (config_map.end() != video) {
            if (video->second.is<std::vector<ov::Tensor>>()) {
                video_rgbs = video->second.as<std::vector<ov::Tensor>>();
            }
            else if (video->second.is<ov::Tensor>()){
                video_rgbs = {video->second.as<ov::Tensor>()};
            }
            else {
                OPENVINO_THROW("Unknown video type.");
            }
        }

        ov::genai::OptionalGenerationConfig config_arg = utils::get_config_from_map(config_map);
        GenerationConfig config = (config_arg.has_value()) ? *config_arg : get_generation_config();
        config.update_generation_config(config_map);

        return generate(
            prompt,
            rgbs,
            video_rgbs,
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
