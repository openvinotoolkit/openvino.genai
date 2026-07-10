// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <variant>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/rag/text_embedding_pipeline.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/genai/visual_language/video_metadata.hpp"

namespace ov {
namespace genai {

/**
 * @brief Result of an embedding computation.
 */
struct OPENVINO_GENAI_EXPORTS EmbedResult {
    ov::Tensor embeddings;
};

/**
 * @brief Embedding pipeline for multimodal and text-only inputs.
 *
 * Computes embedding vectors for a single prompt or a batch of prompts, with optional images and videos.
 */
class OPENVINO_GENAI_EXPORTS EmbeddingPipeline {
public:

    /**
     * @brief Constructs a pipeline from a folder containing tokenizer and VLM IRs.
     *
     * @param models_path Path to the directory containing model xml/bin files and tokenizer.
     * @param device Device.
     * @param properties Optional plugin and pipeline properties to pass to ov::Core::compile_model().
     * e.g ov::AnyMap{{"CACHE_DIR", "/path/to/cache"}, {ov::genai::text_embedding_config.name(), config}}.
     */
    EmbeddingPipeline(const std::filesystem::path& models_path,
                      const std::string& device,
                      const ov::AnyMap& properties = {});

    /**
    * @brief Computes embedding vectors for text or a batch of texts with images and videos.
    *
    * @param text Text or a batch of texts for which embedding is computed.
    * @param images Images for which embedding is computed. Each image is represented as a tensor of shape [H, W, C] (uint8) or a batch [N, H, W, C].
    * @param videos Videos for which embedding is computed. Each video is represented as a tensor of shape [F, H, W, C] (uint8), where F is the number of frames.
    * @param videos_metadata Video metadata for the videos provided.
    * @param properties Generation arguments (e.g. ov::genai::embedding_prompt("Represent this text")).
    * @return EmbedResult.
    */
    EmbedResult embed(const StringInputs& text,
                      const std::vector<ov::Tensor>& images = {},
                      const std::vector<ov::Tensor>& videos = {},
                      const std::vector<VideoMetadata>& videos_metadata = {},
                      const ov::AnyMap& properties = {});

    /**
    * @brief Computes embedding vectors using properties.
    *
    * @param properties Generation arguments and inputs (e.g. ov::genai::text(...), ov::genai::images(...),
    *                   ov::genai::videos(...), ov::genai::videos_metadata(...), ov::genai::embedding_prompt(...)).
    * @return EmbedResult.
    */
    EmbedResult embed(const ov::AnyMap& properties);

    /**
    * @brief Computes embedding vectors using keyword properties.
    *
    * @param properties Generation arguments and inputs (e.g. ov::genai::text(...), ov::genai::images(...),
    *                   ov::genai::embedding_prompt(...)).
    * @return EmbedResult.
    */
    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    EmbedResult embed(Properties&&... properties) {
        return embed(ov::AnyMap{std::forward<Properties>(properties)...});
    }

    ~EmbeddingPipeline();

private:
    class EmbeddingPipelineImpl;
    std::unique_ptr<EmbeddingPipelineImpl> m_impl;
};

/**
 * @brief Text or batch of texts to embed via EmbeddingPipeline::embed(AnyMap).
 */
static constexpr ov::Property<std::variant<std::string, std::vector<std::string>>> text{"text"};

/**
 * @brief Instruction for encoding a document or query.
 * If the model has a chat template, the prompt is added to the system message; otherwise it is prepended to the text.
 */
static constexpr ov::Property<std::string> embedding_prompt{"embedding_prompt"};

/**
 * @brief Configuration for the text embedding pipeline.
 */
static constexpr ov::Property<TextEmbeddingPipeline::Config> text_embedding_config{"text_embedding_config"};
}  // namespace genai
}  // namespace ov
