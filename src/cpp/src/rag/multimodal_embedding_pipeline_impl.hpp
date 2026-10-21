// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "embedding_pipeline_impl.hpp"

#include <optional>
#include <unordered_set>

#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

class MultimodalEmbeddingPipelineImpl final : public EmbeddingPipelineImpl {
public:
    MultimodalEmbeddingPipelineImpl(const std::filesystem::path& models_path,
                                    const std::string& device,
                                    const ov::AnyMap& properties);

    EmbedResult embed(const StringInputs& text, const ov::AnyMap& properties) override;

    EmbedResult embed(const StringInputs& text,
                      const std::vector<ov::Tensor>& images,
                      const std::vector<ov::Tensor>& videos,
                      const std::vector<VideoMetadata>& videos_metadata,
                      const ov::AnyMap& properties) override;

private:
    void init_multimodal(const std::filesystem::path& models_path,
                         const std::string& device,
                         const ov::AnyMap& properties);

    EmbedResult multimodal_embed(const std::vector<std::string>& texts,
                                 const std::vector<EncodedImage>& encoded_images,
                                 const std::vector<EncodedVideo>& encoded_videos,
                                 const std::optional<std::string>& prompt);

    bool has_lm_input(const std::string& input_name) const;
    bool has_lm_output(const std::string& output_name) const;
    std::string format_prompt(const std::string& text, const std::optional<std::string>& prompt) const;
    std::string append_added_special_tokens(Tokenizer& tokenizer, const std::string& text) const;

    std::shared_ptr<InputsEmbedder> m_inputs_embedder;
    TextEmbeddingPipeline::Config m_config;
    ov::CompiledModel m_compiled_language_model;
    ov::InferRequest m_language_model_request;
    std::unordered_set<std::string> m_language_model_input_names;
    std::unordered_set<std::string> m_language_model_output_names;
    std::string m_embedding_output_name;
};

}  // namespace ov::genai
