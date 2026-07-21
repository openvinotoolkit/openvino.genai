// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

/// @brief Vision encoder for GLM-Edge-V (model_type "glm").
/// Reproduces the MllamaImageProcessor preprocessing with a single tile:
/// aspect-preserving fit-to-canvas resize into a fixed square, pad to the
/// square size, rescale and normalize.
class VisionEncoderGlmEdgeV : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

/// @brief Inputs embedder for GLM-Edge-V.
/// The chat/normalize step expands each image into as many
/// `<|begin_of_image|>` placeholder tokens as the vision tower emits, and the
/// vision embeddings replace that contiguous placeholder run 1:1 (LLaVA-style).
class InputsEmbedderGlmEdgeV : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderGlmEdgeV(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderGlmEdgeV(
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
