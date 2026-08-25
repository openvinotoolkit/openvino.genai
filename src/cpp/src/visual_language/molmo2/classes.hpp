// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

/// @brief Vision encoder for the Molmo2 (allenai/MolmoWeb-4B) architecture.
///
/// Molmo2 preprocessing is data-dependent: the input image is decomposed into a low-resolution
/// (global) crop plus a set of overlapping high-resolution crops, each split into 14x14 patches.
/// A pooling-index tensor describes which patches are averaged into a single image token. Both the
/// patchified crops ("images") and the pooling indices ("pooled_patches_idx") are consumed by the
/// exported vision backbone, which returns one embedding row per image token.
class VisionEncoderMolmo2 : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

/// @brief Inputs embedder for the Molmo2 architecture.
///
/// Molmo2 expands the "<|image|>" placeholder into a run of image special tokens whose layout
/// depends on the image grid (a low-resolution section followed by a high-resolution section).
/// Vision embeddings are *added* to the learned "<im_patch>" placeholder embeddings (Molmo2 uses an
/// additive merge rather than a scatter-replace). Image placeholder tokens receive token_type_ids == 1
/// so the language model can apply bidirectional attention between them at prefill.
class InputsEmbedderMolmo2 : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderMolmo2(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderMolmo2(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {}) override;

    std::pair<ov::Tensor, ov::Tensor> get_inputs_embeds_with_token_type_ids(const std::string& prompt, const std::vector<EncodedImage>& images, VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {}) override;

    bool has_token_type_ids() const override;

    std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;

    NormalizedPrompt normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const override;

private:
    /// @brief Build the run of image special tokens for a single image grid, mirroring
    /// Molmo2Processor.get_image_tokens (low-resolution section followed by high-resolution section).
    std::string build_image_tokens(size_t resized_h, size_t resized_w, size_t high_h, size_t high_w) const;

    /// @brief Replace the tokenizer chat template with a Molmo2 template that keeps image tokens
    /// (already embedded in the normalized prompt) in front of the "User:" role prefix.
    void patch_chat_template();
};

} // namespace ov::genai
