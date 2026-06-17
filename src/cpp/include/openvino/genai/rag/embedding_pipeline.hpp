// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/visual_language/video_metadata.hpp"

namespace ov {
namespace genai {

/**
 * @brief Embedding pipeline for multimodal and text-only inputs.
 *
 * Computes a single embedding vector for a prompt with optional images and videos.
 */
class OPENVINO_GENAI_EXPORTS EmbeddingPipeline {
public:
    /**
     * @brief Constructs a pipeline from a folder containing tokenizer and VLM IRs.
     *
     * @param models_path Path to the directory containing model xml/bin files and tokenizer.
     * @param device Device.
     * @param properties Optional plugin properties to pass to ov::Core::compile_model().
     */
    EmbeddingPipeline(const std::filesystem::path& models_path,
                      const std::string& device,
                      const ov::AnyMap& properties = {});

    /**
     * @brief Constructs a pipeline from a folder containing tokenizer and VLM IRs.
     *
     * @param models_path Path to the directory containing model xml/bin files and tokenizer.
     * @param device Device.
     * @param properties Plugin and/or config properties.
     */
    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    EmbeddingPipeline(const std::filesystem::path& models_path,
                      const std::string& device,
                      Properties&&... properties)
        : EmbeddingPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    /**
     * @brief Computes an embedding vector for text.
     */
    std::vector<float> embed(const std::string& text);

    /**
     * @brief Computes a document embedding vector for text.
     */
    std::vector<float> embed_document(const std::string& text);

    /**
     * @brief Computes document embedding vectors for a batch of texts.
     */
    std::vector<std::vector<float>> embed_documents(const std::vector<std::string>& texts);

    /**
     * @brief Starts asynchronous embedding computation for a batch of texts.
     */
    void start_embed_documents_async(const std::vector<std::string>& texts);

    /**
     * @brief Waits for asynchronous batch embedding computation and returns results.
     */
    std::vector<std::vector<float>> wait_embed_documents();

    /**
     * @brief Starts asynchronous embedding computation for text.
     */
    void start_embed_async(const std::string& text);

    /**
     * @brief Waits for asynchronous embedding computation and returns result.
     */
    std::vector<float> wait_embed();

    /**
     * @brief Computes an embedding vector for text and images.
     */
    std::vector<float> embed(const std::string& text, const std::vector<ov::Tensor>& images);

    /**
     * @brief Computes an embedding vector for text, images and videos.
     */
    std::vector<float> embed(const std::string& text,
                             const std::vector<ov::Tensor>& images,
                             const std::vector<ov::Tensor>& videos,
                             const std::vector<VideoMetadata>& videos_metadata = {});

    ~EmbeddingPipeline();

private:
    class EmbeddingPipelineImpl;
    std::unique_ptr<EmbeddingPipelineImpl> m_impl;
};

}  // namespace genai
}  // namespace ov
