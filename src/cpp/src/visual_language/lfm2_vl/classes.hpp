// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

// Vision encoder for LiquidAI LFM2-VL (model_type "lfm2_vl").
//
// The naflex Siglip2 tower + multimodal projector is exported as a single IR
// consuming one image's flattened valid patches. This encoder reproduces the
// HuggingFace/optimum-intel `Lfm2VlImageProcessor` naflex preprocessing
// (smart-resize, optional tiling with a thumbnail, antialias bilinear resize,
// patch packing and the per-image positional resample kernel), runs the IR for
// every tile/thumbnail and concatenates the projected token features.
class VisionEncoderLFM2VL : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

class InputsEmbedderLFM2VL : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderLFM2VL(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderLFM2VL(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {}) override;

    std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images
    ) const override;

private:
    // Builds the placeholder expansion string for a single image and appends the
    // per-run (per-tile / thumbnail / single) image-token counts to run_sizes.
    std::string build_image_placeholders(const EncodedImage& image, std::vector<size_t>& run_sizes) const;
};

} // namespace ov::genai
