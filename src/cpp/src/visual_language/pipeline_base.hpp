// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/visual_language/pipeline.hpp"
#include "utils.hpp"

using namespace ov::genai;

namespace {

} // namespace

namespace ov::genai {

const ModelsMap::mapped_type& get_model_weights_pair(const ModelsMap& models_map, const std::string& key) {
    auto it = models_map.find(key);
    if (it != models_map.end()) {
        return it->second;
    }
    OPENVINO_THROW("Model with key '", key, "' not found in models map.");
}

// TODO: move to utils
std::pair<ov::AnyMap, SchedulerConfig> extract_scheduler_config(const ov::AnyMap& properties, std::optional<SchedulerConfig> default_config = std::nullopt) {
    ov::AnyMap plugin_config = properties;
    auto it = plugin_config.find(ov::genai::scheduler_config.name());
    SchedulerConfig scheduler_config;
    if (it != plugin_config.end()) {
        scheduler_config = it->second.as<SchedulerConfig>();
        plugin_config.erase(it);
    } else if (default_config.has_value()) {
        scheduler_config = *default_config;
    }
    return {plugin_config, scheduler_config};
};

class ov::genai::VLMPipeline::VLMPipelineBase {
    // Load pipeline time
    float m_load_time_ms = 0;
public:

    virtual ~VLMPipelineBase() = default;

    virtual VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& rgbs,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) = 0;

    VLMDecodedResults generate(
        const std::string& prompt,
        const ov::AnyMap& config_map
    ) {
        auto image = config_map.find(ov::genai::image.name());
        auto images = config_map.find(ov::genai::images.name());
        OPENVINO_ASSERT(
            config_map.end() == image || config_map.end() == images,
            "Only one property can be set: image of images."
        );
        std::vector<ov::Tensor> rgbs;
        if (config_map.end() != image) {
            rgbs = {image->second.as<ov::Tensor>()};
        } if (config_map.end() != images) {
            rgbs = images->second.as<std::vector<ov::Tensor>>();
        }

        ov::genai::OptionalGenerationConfig config_arg = utils::get_config_from_map(config_map);
        GenerationConfig config = (config_arg.has_value()) ? *config_arg : get_generation_config();
        config.update_generation_config(config_map);

        return generate(
            prompt,
            rgbs,
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