// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

/// @brief Vision encoder for GLM-Edge-V (zai-org/glm-edge-v-2b, config.model_type == "glm").
/// Uses Mllama-style preprocessing: aspect-preserving bicubic resize to fit a fixed square
/// canvas followed by bottom-right zero padding. The exported vision model already projects
/// each image into the language-model hidden dimension, so no resampler/projector is applied
/// here. The exported vision output is float16 and is converted to float32 for embedding merge.
class VisionEncoderGLMEdgeV : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

/// @brief Inputs embedder for GLM-Edge-V. Mirrors the LLaVA placeholder-replacement scheme:
/// the single image tag is expanded into a run of `<|begin_of_image|>` (boi) placeholder tokens
/// (one per image embedding), and vision embeddings replace those positions via
/// utils::merge_text_and_image_embeddings_llava.
class InputsEmbedderGLMEdgeV : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderGLMEdgeV(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const Tokenizer& tokenizer,
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
