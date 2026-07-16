// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

/// @brief Vision encoder for GLM-Edge-V (model_type "glm" with a nested
/// SigLIP vision_config). Preprocessing mirrors the HF MllamaImageProcessor
/// with max_image_tiles == 1: aspect-ratio-preserving resize to fit within a
/// single size.height x size.width canvas, followed by bottom/right padding.
class VisionEncoderGLMEdgeV : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

/// @brief Inputs embedder for GLM-Edge-V. The image placeholder token is
/// <|begin_of_image|> (boi_token_id). The prompt is expanded so that each image
/// contributes N copies of that token (N == number of vision embedding rows),
/// and vision features are scattered into those positions like InternVL.
class InputsEmbedderGLMEdgeV : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderGLMEdgeV(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderGLMEdgeV(
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
};

} // namespace ov::genai
