// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/vision_encoder.hpp"
#include "openvino/genai/vlm_config.hpp"

namespace ov::genai {
/// @brief A string prompt and source image.
struct PromptImages {
    /// @brief A prompt represented as std::string.
    std::string prompt;
    /// @brief An image represented as ov::Tensor.
    std::vector<ov::Tensor> images;
};

/// @brief A Visual language modeling pipeline class used to generate a
/// response or run a chat given a prompt and an image.
class OPENVINO_GENAI_EXPORTS VLMPipeline {
public:
    // A config to follow for LLM input construction.
    VLMConfig m_vlm_config;
    // A config to follow for text generation.
    GenerationConfig m_generation_config;
    // A tokenizer encoding a prompt.
    Tokenizer m_tokenizer;
    // An encoder to infer embeddings of an image.
    VisionEncoder m_vision_encoder;
    // A resampler model to resample image embeddings.
    // [N, H*W, old_hidden_size] is the input shape.
    // [N, query_num, hidden_size] is the output shape.
    ov::InferRequest m_resampler;
    // A model to compute token embeddings.
    // Input shape: [N, conversation length].
    // Output shape: [1, conversation length, hidden_size].
    ov::InferRequest m_embedding;
    // A language model used to generate a response.
    // Input shapes: inputs_embeds[N, conversation length, hidden_size],
    // position_ids[N, conversation length], beam_idx[N].
    // Output shape: logits[N, conversation length, vocab_size].
    ov::InferRequest m_language;
    // Precomputed positional embeddings for the resampler.
    // [70, 70, hidden_size]. 70 is the initial guess of the image
    // height and width after dividing by patch_size.
    ov::Tensor m_pos_embed_cache;
    // True if chat mode is activated to save conversation
    // history between generate() calls.
    bool m_is_chat_conversation;
    ChatHistory m_history;
    std::string m_templated_chat_history;
    size_t image_id = 0;  // Used to insert <image_id>i</image_id> per image (not a slice).

    explicit VLMPipeline(
        const std::filesystem::path& model_dir,
        const std::string& device="CPU",
        const ov::AnyMap device_config={},
        ov::Core core=ov::Core{}
    ) : VLMPipeline{
        model_dir,
        Tokenizer(model_dir.string(), device_config),
        device,
        device_config,
        core
    } {}

    VLMPipeline(
        const std::filesystem::path& model_dir,
        const ov::genai::Tokenizer& tokenizer,
        const std::string& device="CPU",
        const ov::AnyMap device_config={},
        ov::Core core=ov::Core{}
    );

    /// @brief Default destructor.
    ~VLMPipeline();

    DecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const GenerationConfig& generation_config,
        const StreamerVariant& streamer
    );
    DecodedResults generate(
        const std::string& prompt,
        const ov::AnyMap& config_map
    );
    template <typename... Properties>
    util::EnableIfAllStringAny<DecodedResults, Properties...> generate(
        const std::string& prompt,
        Properties&&... properties
    ) {
        return generate(
            prompt, AnyMap{std::forward<Properties>(properties)...}
        );
    }
    void start_chat(const std::string& system_message="");
    void finish_chat() {m_is_chat_conversation = false;}
    GenerationConfig get_generation_config() const;
    void set_generation_config(const GenerationConfig& new_config);
private:
    class VLMPipelineImpl;
    std::unique_ptr<VLMPipelineImpl> m_pimpl;
};

/*
 * utils that allow to use generate() in the following way:
 * pipe.generate(prompt, ov::genai::image(std::move(image_tensor))).
*/
static constexpr ov::Property<ov::Tensor> image{"image"};
static constexpr ov::Property<std::vector<ov::Tensor>> images{"images"};
}
