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
};

class InputsEmbedderLLaVANext : public InputsEmbedderLLaVA {
public:
    using InputsEmbedderLLaVA::InputsEmbedderLLaVA;

    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, std::optional<ov::Tensor> merged_image_embeddings = std::nullopt) override;

    std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;
};

} // namespace ov::genai
