// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "circular_buffer_queue.hpp"

namespace ov::genai {

class VisionEncoderMistral3 : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

class InputsEmbedderMistral3 : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderMistral3(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderMistral3(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    ov::Tensor get_inputs_embeds(
        const std::string& prompt,
        const std::vector<ov::genai::EncodedImage>& images,
        ov::genai::VLMPerfMetrics& metrics,
        bool recalculate_merged_embeddings = true,
        const std::vector<size_t>& image_sequence = {}) override;

    std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images) const override;

private:
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_multi_modal_projector;

    /// @brief Apply spatial merge (unfold) to vision encoder output.
    /// Groups adjacent spatial_merge_size x spatial_merge_size patches and concatenates their features.
    /// Input: [num_patches, hidden_size] with grid dims (h_patches, w_patches).
    /// Output: [h_merged * w_merged, hidden_size * spatial_merge_size^2].
    ov::Tensor spatial_merge(const ov::Tensor& features, size_t h_patches, size_t w_patches) const;

    /// @brief Merge text and image embeddings using masked scatter.
    /// Replaces every position in text_embeds where input_ids == image_token_id
    /// with the next image embedding in sequence. Handles non-contiguous image tokens
    /// separated by [IMG_BREAK]/[IMG_END].
    ov::Tensor merge_image_embeddings(
        const ov::Tensor& input_ids,
        const ov::Tensor& text_embeds,
        const std::vector<ov::Tensor>& image_embeds,
        int64_t image_token_id) const;
};

}  // namespace ov::genai
