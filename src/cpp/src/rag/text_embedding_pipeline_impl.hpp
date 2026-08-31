// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "embedding_pipeline_impl.hpp"

namespace ov::genai {

class TextEmbeddingPipelineImpl final : public EmbeddingPipelineImpl {
public:
    TextEmbeddingPipelineImpl(const std::filesystem::path& models_path,
                              const std::string& device,
                              const TextEmbeddingPipeline::Config& config,
                              const ov::AnyMap& properties);

    EmbedResult embed(const StringInputs& text, const ov::AnyMap& properties) override;

    EmbedResult embed(const StringInputs& text,
                      const std::vector<ov::Tensor>& images,
                      const std::vector<ov::Tensor>& videos,
                      const std::vector<VideoMetadata>& videos_metadata,
                      const ov::AnyMap& properties) override;

private:
    std::unique_ptr<TextEmbeddingPipeline> m_text_embedding_pipeline;
};

}  // namespace ov::genai
