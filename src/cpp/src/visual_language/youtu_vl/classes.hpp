// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <unordered_map>
#include <vector>

#include "visual_language/vlm_config.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

/// @brief Vision encoder for YoutuVL model (Siglip2-based vision encoder).
/// Preprocesses images using Siglip2 patch-based approach and runs the
/// openvino_vision_embeddings_model which internally applies the VLPatchMerger.
class VisionEncoderYoutuVL : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

/// @brief Inputs embedder for YoutuVL model.
/// Replaces <|image_pad|> tokens with vision embeddings produced by the
/// Siglip2-based vision encoder.
class InputsEmbedderYoutuVL : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderYoutuVL(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderYoutuVL(
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

    std::vector<ov::genai::EncodedImage> encode_images(
        const std::vector<ov::Tensor>& images) override;

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images) const override;

protected:
    // Native image tag produced by the YoutuVL chat template:
    //   <|vision_start|><|image_pad|><|vision_end|>
    inline static const std::string NATIVE_TAG = "<|vision_start|><|image_pad|><|vision_end|>";

private:
    std::unordered_map<std::string, int64_t> m_special_token_ids;

    void load_special_token_ids(const std::filesystem::path& config_dir_path);
    ov::Tensor encode_prompt_with_special_token_ids(
        const std::string& prompt,
        ov::genai::VLMPerfMetrics& metrics);
};

} // namespace ov::genai
