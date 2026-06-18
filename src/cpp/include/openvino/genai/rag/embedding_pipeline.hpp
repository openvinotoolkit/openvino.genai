// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include "openvino/genai/visibility.hpp"
#include "openvino/runtime/tensor.hpp"
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
     * @brief Computes an embedding vector for text.
     */
    ov::Tensor embed(const std::string& text, const std::optional<std::string>& prompt = std::nullopt);

    /**
     * @brief Computes embedding vectors for a batch of texts.
     */
    ov::Tensor embed(const std::vector<std::string>& texts, const std::optional<std::string>& prompt = std::nullopt);

    /**
     * @brief Starts asynchronous embedding computation for text.
     */
    void start_embed_async(const std::string& text, const std::optional<std::string>& prompt = std::nullopt);

    /**
     * @brief Starts asynchronous embedding computation for a batch of texts.
     */
    void start_embed_async(const std::vector<std::string>& texts, const std::optional<std::string>& prompt = std::nullopt);

    /**
     * @brief Waits for asynchronous embedding computation and returns result.
     */
    ov::Tensor wait_embed();

    /**
     * @brief Computes an embedding vector for text and images.
     */
    ov::Tensor embed(const std::string& text,
                     const std::vector<ov::Tensor>& images,
                     const std::optional<std::string>& prompt = std::nullopt);

    /**
     * @brief Computes an embedding vector for text, images and videos.
     */
    ov::Tensor embed(const std::string& text,
                     const std::vector<ov::Tensor>& images,
                     const std::vector<ov::Tensor>& videos,
                     const std::vector<VideoMetadata>& videos_metadata = {},
                     const std::optional<std::string>& prompt = std::nullopt);

    ~EmbeddingPipeline();

private:
    class EmbeddingPipelineImpl;
    std::unique_ptr<EmbeddingPipelineImpl> m_impl;
};

}  // namespace genai
}  // namespace ov
