// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/vision_encoder.hpp"
#include "openvino/genai/vlm_config.hpp"

namespace ov::genai {
struct PromptImage {
    std::string prompt;
    ov::Tensor image;
};

class OPENVINO_GENAI_EXPORTS VLMPipeline {
public:
    VLMConfig m_vlm_config;
    Tokenizer tokenizer;
    VisionEncoder m_vision_encoder;
    ov::InferRequest resampler, ireq_embed, ireq;
    std::vector<float> language_embeddings_history;
    size_t history_length = 0;
    ov::Tensor pos_embed_cache;

    VLMPipeline(
        const ov::genai::Tokenizer& tokenizer,
        const VisionEncoder& vision_encoder,
        const ov::InferRequest& resampler,
        const ov::InferRequest& embedding,
        const ov::InferRequest& language_model,
        const VLMConfig& vlm_config=VLMConfig{}
    );

    explicit VLMPipeline(
        const std::filesystem::path& model_dir,
        const std::string& device="CPU",
        const ov::AnyMap device_config={},
        ov::Core core=ov::Core{}
    );

    std::string generate(
        const PromptImage& pair,
        const ProcessorConfig& processor_config,
        const VLMConfig& vlm_config,
        const std::function<bool(std::string&&)>& callback
    );
    std::string generate(
        const PromptImage& pair,
        const ProcessorConfig& processor_config,
        const VLMConfig& vlm_config,
        const std::shared_ptr<ov::genai::StreamerBase>& streamer=nullptr
    );
    std::string generate(
        const PromptImage& pair,
        const std::function<bool(std::string&&)>& callback
    ) {
        return generate(
            pair,
            m_vision_encoder.m_processor_config,
            m_vlm_config,
            callback
        );
    }
    std::string generate(
        const PromptImage& pair,
        const std::shared_ptr<ov::genai::StreamerBase>& streamer=nullptr
    ) {
        return generate(
            pair,
            m_vision_encoder.m_processor_config,
            m_vlm_config,
            streamer
        );
    }
    std::string generate(
        const PromptImage& pair,
        const ov::AnyMap& config_map
    );
    template <typename... Properties>
    util::EnableIfAllStringAny<PromptImage, Properties...> generate(
        const PromptImage& pair,
        Properties&&... properties
    ) {
        return generate(pair, AnyMap{
            std::forward<Properties>(properties)...
        });
    }
    void start_chat() {}
    void finish_chat() {}
    void set_2d_pos_cache(const HeightWidth& max_size);
    void adjust_pos_cache(const std::vector<HeightWidth>& target_sizes);
};
}
