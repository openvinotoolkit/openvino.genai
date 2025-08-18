// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>
#include <iostream>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "visual_language/cdpruner/cdpruner.hpp"
#include "visual_language/embedding_model.hpp"
#include "openvino/genai/tokenizer.hpp"

namespace ov::genai {

class VisionEncoderLLaVA : public VisionEncoder {
private:
    // Text processing components for relevance calculation
    std::shared_ptr<EmbeddingsModel> m_text_embedding_model;
    std::optional<Tokenizer> m_tokenizer;
    
    // VLM configuration
    VLMConfig m_vlm_config;
    
    // Helper method to extract text features
    ov::Tensor extract_text_features(const std::string& text_prompt);
    
    // Helper method to initialize text processing components
    void initialize_text_components(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap& properties);
    
    void initialize_text_components(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap& properties);

    bool is_pruning_available() override;

public:
    VisionEncoderLLaVA(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap properties);

    VisionEncoderLLaVA(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap properties);

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
    
    EncodedImage encode_with_pruning(
        const ov::Tensor& image,
        const std::string& text_prompt,
        const size_t viusal_tokens_retain_percentage,
        const ov::AnyMap& config_map = {}) override;
};

class InputsEmbedderLLaVA : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderLLaVA(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderLLaVA(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {}) override;

    std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;

    std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images, const ov::AnyMap& config_map) override;

    std::pair<std::string, std::vector<size_t>> normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images
    ) const override;

    /// @brief Enhanced version of normalize_prompt that supports pruned features
    /// @param prompt The input prompt with image placeholders
    /// @param base_id Base ID for image indexing
    /// @param images Vector of encoded images (may contain pruned features)
    /// @return Pair of normalized prompt and image sequence
    std::pair<std::string, std::vector<size_t>> normalize_prompt_with_pruning_support(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images
    ) const;

protected:
    ov::Tensor merge_text_and_image_embeddings_llava(
        const ov::Tensor& input_ids,
        ov::Tensor& text_embeds,
        const std::vector<ov::Tensor>& image_embeds,
        int64_t image_token_id);

    /// @brief Check if the image features appear to be pruned
    /// @param image_features The image features tensor
    /// @param resized_source_size The reported grid size
    /// @return true if features appear to be pruned
    bool is_image_features_pruned(
        const ov::Tensor& image_features,
        const ImageSize& resized_source_size
    ) const;

    /// @brief Get the actual number of visual tokens from image features
    /// @param image_features The image features tensor
    /// @return Number of visual tokens
    size_t get_actual_visual_token_count(const ov::Tensor& image_features) const;
};

} // namespace ov::genai
