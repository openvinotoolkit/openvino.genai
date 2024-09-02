// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/vision_encoder.hpp"
#include "openvino/genai/vlm_config.hpp"

namespace ov::genai {
/// @brief Only batch size one is supported.
struct EncodedPromptImage {
    EncodedInputs prompt;
    EncodedImage image;
};

/// @brief Only batch size one is supported.
struct PromptImage {
    StringInputs prompt;
    ov::Tensor image;
};

class OPENVINO_GENAI_EXPORTS VLMPipeline {
public:
    VLMConfig m_vlm_config;
    Tokenizer m_tokenizer;
    VisionEncoder m_vision_encoder;
    ov::InferRequest m_resampler, m_embedding, m_language;
    std::vector<float> m_language_embeddings_history;
    size_t m_history_length;
    ov::Tensor m_pos_embed_cache;
    bool is_chat_conversation;

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

    EncodedResults generate(
        const EncodedPromptImage& pair,
        const ProcessorConfig& processor_config,
        const VLMConfig& vlm_config,
        const std::function<bool(std::string&&)>& callback
    );
    EncodedResults generate(
        const EncodedPromptImage& pair,
        const ProcessorConfig& processor_config,
        const VLMConfig& vlm_config,
        const std::shared_ptr<ov::genai::StreamerBase>& streamer=nullptr
    );
    EncodedResults generate(
        const EncodedPromptImage& pair,
        const std::function<bool(std::string&&)>& callback
    ) {
        return generate(
            pair,
            m_vision_encoder.m_processor_config,
            m_vlm_config,
            callback
        );
    }
    EncodedResults generate(
        const EncodedPromptImage& pair,
        const std::shared_ptr<ov::genai::StreamerBase>& streamer=nullptr
    ) {
        return generate(
            pair,
            m_vision_encoder.m_processor_config,
            m_vlm_config,
            streamer
        );
    }
    EncodedResults generate(
        const EncodedPromptImage& pair,
        const ov::AnyMap& config_map
    );
    template <typename... Properties>
    util::EnableIfAllStringAny<EncodedResults, Properties...> generate(
        const EncodedPromptImage& pair,
        Properties&&... properties
    ) {
        return generate(pair, AnyMap{
            std::forward<Properties>(properties)...
        });
    }

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
    util::EnableIfAllStringAny<std::string, Properties...> generate(
        const PromptImage& pair,
        Properties&&... properties
    ) {
        return generate(pair, AnyMap{
            std::forward<Properties>(properties)...
        });
    }
    void start_chat() {is_chat_conversation = true;}
    void finish_chat() {is_chat_conversation = false;}
};
}
