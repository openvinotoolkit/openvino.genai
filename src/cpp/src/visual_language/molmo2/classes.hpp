// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

/// @brief Vision encoder for Molmo2 (AI2 "MolmoWeb") models.
///
/// Molmo2 preprocesses an image into a set of overlapping high-resolution crops plus one
/// low-resolution global thumbnail (the original Molmo multi-crop tiling recipe), feeds all
/// crops through a shared ViT, then pools patch features in 2x2 groups via a learned
/// cross-attention pooling head. The vision embeddings OpenVINO model therefore takes two
/// inputs (`pixel_values`, `image_token_pooling`) instead of the single `pixel_values` input
/// used by most other VLMs in this codebase.
class VisionEncoderMolmo2 : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

/// @brief Inputs embedder for Molmo2 models.
///
/// Unlike llava-style VLMs that *replace* placeholder token embeddings with vision features,
/// Molmo2 *adds* the pooled vision feature onto the `<im_patch>` placeholder token's own
/// (learned) text embedding. Bidirectional attention over image tokens is enabled via
/// `token_type_ids`, following the same dispatch mechanism as Gemma3/Gemma4.
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
    // Shared implementation behind both get_inputs_embeds() and
    // get_inputs_embeds_with_token_type_ids(): computes the merged (text + additive vision)
    // embeddings and, in the same pass, the per-token image/text type ids. The exported
    // language model currently exposed by optimum-intel's Molmo2 config (upstream PR #1812)
    // does not have a `token_type_ids` input port, so has_token_type_ids() reports false and
    // only the embeddings are consumed today -- the type ids are still computed and returned
    // here for forward-compatibility, so this method keeps working transparently if a future
    // export adds bidirectional image-token attention support.
    std::pair<ov::Tensor, ov::Tensor> compute_merged_embeds_and_token_type_ids(
        const std::string& unified_prompt,
        const std::vector<EncodedImage>& images,
        VLMPerfMetrics& metrics,
        const std::vector<size_t>& images_sequence);
};

} // namespace ov::genai
