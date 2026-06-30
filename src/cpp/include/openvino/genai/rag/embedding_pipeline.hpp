// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/rag/text_embedding_pipeline.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/genai/visual_language/video_metadata.hpp"

namespace ov {
namespace genai {

/**
 * @brief Embedding pipeline for multimodal and text-only inputs.
 *
 * Computes embedding vectors for a single prompt or a batch of prompts, with optional images and videos.
 */
class OPENVINO_GENAI_EXPORTS EmbeddingPipeline {
public:
    using TextInput = std::variant<std::string, std::vector<std::string>>;

    /**
     * @brief Constructs a pipeline from a folder containing tokenizer and VLM IRs.
     *
     * @param models_path Path to the directory containing model xml/bin files and tokenizer.
     * @param device Device.
     * @param config Pipeline configuration.
     * @param properties Optional plugin properties to pass to ov::Core::compile_model().
     */
    EmbeddingPipeline(const std::filesystem::path& models_path,
                      const std::string& device,
                      const TextEmbeddingPipeline::Config& config,
                      const ov::AnyMap& properties = {});

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
    * @brief Computes embedding vectors for text or a batch of texts with images and videos.
    *
    * @param text Text or a batch of texts for which embedding is computed.
    * @param images Images for which embedding is computed. Each image is represented as a tensor of shape [channels, height, width].
    * @param videos Videos for which embedding is computed. Each video is represented as a tensor of shape [num_frames, channels, height, width].
    * @param videos_metadata Video metadata for the videos provided.
    * @param prompt Prompt to use for embedding computation.
    * @return Embedding tensor.
    */
    ov::Tensor embed(const TextInput& text,
                     const std::vector<ov::Tensor>& images = {},
                     const std::vector<ov::Tensor>& videos = {},
                     const std::vector<VideoMetadata>& videos_metadata = {},
                     const std::optional<std::string>& prompt = std::nullopt);

    ov::Tensor embed(const ov::AnyMap& properties);

    /**
    * @brief Computes embedding vectors for text or a batch of texts with images and videos.
    *
    * @param text Text or a batch of texts for which embedding is computed.
    * @param images Images for which embedding is computed. Each image is represented as a tensor of shape [channels, height, width].
    * @param videos Videos for which embedding is computed. Each video is represented as a tensor of shape [num_frames, channels, height, width].
    * @param videos_metadata Video metadata for the videos provided.
    * @param prompt Prompt to use for embedding computation.
    * @return Embedding tensor.
    */
     template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    ov::Tensor embed(Properties&&... properties) {
        return embed(ov::AnyMap{std::forward<Properties>(properties)...});
    }

    /**
    * @brief Starts asynchronous embedding computation for text or a batch of texts with images and videos.
    * 
    * @param text Text or a batch of texts for which embedding is computed.
    * @param images Images for which embedding is computed. Each image is represented as a tensor of shape [channels, height, width].
    * @param videos Videos for which embedding is computed. Each video is represented as a tensor of shape [num_frames, channels, height, width].
    * @param videos_metadata Video metadata for the videos provided.
    * @param prompt Prompt to use for embedding computation.
    * @return Embedding tensor.
    */
    void start_embed_async(const TextInput& text,
                           const std::vector<ov::Tensor>& images = {},
                           const std::vector<ov::Tensor>& videos = {},
                           const std::vector<VideoMetadata>& videos_metadata = {},
                           const std::optional<std::string>& prompt = std::nullopt);

    void start_embed_async(const ov::AnyMap& properties);

    /**
    * @brief Starts asynchronous embedding computation for images and videos.
    * 
    * @param images Images for which embedding is computed. Each image is represented as a tensor of shape [channels, height, width].
    * @param videos Videos for which embedding is computed. Each video is represented as a tensor of shape [num_frames, channels, height, width].
    * @param videos_metadata Video metadata for the videos provided.
    * @param prompt Prompt to use for embedding computation.
    */
    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    void start_embed_async(Properties&&... properties) {
        start_embed_async(ov::AnyMap{std::forward<Properties>(properties)...});
    }

    /**
    * @brief Waits for asynchronous embedding computation and returns result.
     */
    ov::Tensor wait();

    ~EmbeddingPipeline();

private:
    class EmbeddingPipelineImpl;
    std::unique_ptr<EmbeddingPipelineImpl> m_impl;
};

/**
 * @brief Prompt to use for embedding computation.
 */
static constexpr ov::Property<std::string> prompt{"prompt"};

}  // namespace genai
}  // namespace ov
