// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/rag/embedding_pipeline.hpp"

#include "embedding_pipeline_impl.hpp"
#include "multimodal_embedding_pipeline_impl.hpp"
#include "text_embedding_utils.hpp"
#include "text_embedding_pipeline_impl.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

EmbeddingPipeline::EmbeddingPipeline(const std::filesystem::path& models_path,
                                     const std::string& device,
                                     const ov::AnyMap& properties) {
    if (std::filesystem::exists(models_path / "openvino_language_model.xml")) {
        m_impl = std::make_unique<MultimodalEmbeddingPipelineImpl>(models_path, device, properties);
    } else {
        m_impl = std::make_unique<TextEmbeddingPipelineImpl>(models_path, device, properties);
    }
}

EmbedResult EmbeddingPipeline::embed(const StringInputs& text,
                                    const std::vector<ov::Tensor>& images,
                                    const std::vector<ov::Tensor>& videos,
                                    const std::vector<VideoMetadata>& videos_metadata,
                                    const ov::AnyMap& properties) {
    return m_impl->embed(text, images, videos, videos_metadata, properties);
}

EmbedResult EmbeddingPipeline::embed(const ov::AnyMap& properties) {
    std::vector<ov::Tensor> images_vec;
    std::vector<ov::Tensor> videos_vec;
    std::vector<VideoMetadata> videos_metadata_vec;
    std::variant<std::string, std::vector<std::string>> text_variant{std::string{}};

    utils::read_anymap_param(properties, ov::genai::images.name(), images_vec);
    utils::read_anymap_param(properties, ov::genai::videos.name(), videos_vec);
    utils::read_anymap_param(properties, ov::genai::videos_metadata.name(), videos_metadata_vec);
    
    const auto text_it = properties.find(ov::genai::text.name());
    if (text_it != properties.end() && !text_it->second.empty()) {
        if (text_it->second.is<std::string>()) {
            text_variant = text_it->second.as<std::string>();
        } else if (text_it->second.is<std::vector<std::string>>()) {
            text_variant = text_it->second.as<std::vector<std::string>>();
        } else if (text_it->second.is<std::variant<std::string, std::vector<std::string>>>()) {
            text_variant = text_it->second.as<std::variant<std::string, std::vector<std::string>>>();
        } else {
            OPENVINO_THROW("Unsupported type for 'text' property. Expected std::string or std::vector<std::string>.");
        }
    }

    StringInputs text_input = text_variant;

    return m_impl->embed(text_input, images_vec, videos_vec, videos_metadata_vec, properties);
}

EmbeddingPipeline::~EmbeddingPipeline() = default;

}  // namespace genai
}  // namespace ov
