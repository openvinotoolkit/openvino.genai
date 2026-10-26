// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

class VisionEncoderLFM3VL : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

class InputsEmbedderLFM3VL : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderLFM3VL(const VLMConfig& vlm_config,
                        const std::filesystem::path& model_dir,
                        const Tokenizer& tokenizer,
                        const std::string& device,
                        const ov::AnyMap device_config);

    InputsEmbedderLFM3VL(const VLMConfig& vlm_config,
                        const ModelsMap& models_map,
                        const Tokenizer& tokenizer,
                        const std::filesystem::path& config_dir_path,
                        const std::string& device,
                        const ov::AnyMap device_config);

    std::vector<EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;

    NormalizedPrompt normalize_prompt(const std::string& prompt,
                                      size_t base_id,
                                      const std::vector<EncodedImage>& images) const override;

    ov::Tensor get_inputs_embeds(const std::string& prompt,
                                const std::vector<EncodedImage>& images,
                                VLMPerfMetrics& metrics,
                                bool recalculate_merged_embeddings = true,
                                const std::vector<size_t>& image_sequence = {}) override;
};

}  // namespace ov::genai
