// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "text_embedding_pipeline_impl.hpp"

#include <optional>

#include "text_embedding_utils.hpp"
#include "utils.hpp"

namespace ov::genai {
namespace {

ov::Tensor embedding_results_to_tensor(const EmbeddingResults& embedding_results) {
    return std::visit([](const auto& values) {
        if (values.empty()) {
            return ov::Tensor(ov::element::f32, {0, 0});
        }

        const size_t embedding_size = values.front().size();
        ov::Tensor result(ov::element::f32, {values.size(), embedding_size});
        float* result_data = result.data<float>();

        for (size_t row_idx = 0; row_idx < values.size(); ++row_idx) {
            OPENVINO_ASSERT(values[row_idx].size() == embedding_size,
                            "All embedding vectors must have the same size");
            for (size_t column_idx = 0; column_idx < embedding_size; ++column_idx) {
                result_data[row_idx * embedding_size + column_idx] = static_cast<float>(values[row_idx][column_idx]);
            }
        }

        return result;
    }, embedding_results);
}

}  // namespace

TextEmbeddingPipelineImpl::TextEmbeddingPipelineImpl(const std::filesystem::path& models_path,
                                                     const std::string& device,
                                                     const ov::AnyMap& properties)
    : m_text_embedding_pipeline(std::make_unique<TextEmbeddingPipeline>(
          models_path,
          device,
          utils::get_text_embedding_config(properties),
          utils::remove_config_properties(properties))) {}

EmbedResult TextEmbeddingPipelineImpl::embed(const StringInputs& text, const ov::AnyMap& properties) {
    std::optional<std::string> prompt;
    utils::read_anymap_param(properties, embedding_prompt.name(), prompt);
    const std::vector<std::string> texts = std::holds_alternative<std::string>(text)
        ? std::vector<std::string>{std::get<std::string>(text)}
        : std::get<std::vector<std::string>>(text);

    if (prompt.has_value()) {
        return EmbedResult{embedding_results_to_tensor(m_text_embedding_pipeline->embed(texts, *prompt))};
    }
    return EmbedResult{embedding_results_to_tensor(m_text_embedding_pipeline->embed_documents(texts))};
}

EmbedResult TextEmbeddingPipelineImpl::embed(const StringInputs& text,
                                             const std::vector<ov::Tensor>& images,
                                             const std::vector<ov::Tensor>& videos,
                                             const std::vector<VideoMetadata>& videos_metadata,
                                             const ov::AnyMap& properties) {
    OPENVINO_ASSERT(images.empty() && videos.empty() && videos_metadata.empty(),
                    "This model does not support image/video input");
    return embed(text, properties);
}

}  // namespace ov::genai
