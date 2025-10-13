// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/llava/classes.hpp"

namespace ov::genai {

class VisionEncoderLLaVANext : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;

protected:
    ov::Tensor get_pixel_values_llava_next(const ov::Tensor& image, const ProcessorConfig& config);

};

class InputsEmbedderLLaVANext : public InputsEmbedderLLaVA {
public:
    using InputsEmbedderLLaVA::InputsEmbedderLLaVA;

    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {}) override;

    std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;

    NormlizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images
    ) const override;

    ov::Tensor pack_image_features_llava_next(const EncodedImage& encoded_image, const ov::Tensor& image_newline) const;
};

} // namespace ov::genai
