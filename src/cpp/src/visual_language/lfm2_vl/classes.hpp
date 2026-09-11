// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/inputs_embedder.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai {

// SigLIP2-NaFlex vision tower with a fused projector, single-tile only. LFM2-VL's image processor
// (`do_image_splitting=True`, `min_tiles=2`) additionally supports multi-tile grid splitting with a
// thumbnail tile for images that don't fit the single-tile pixel budget; that path isn't implemented
// here yet, see `encode()`.
class VisionEncoderLfm2Vl : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

class InputsEmbedderLfm2Vl : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderLfm2Vl(const VLMConfig& vlm_config,
                         const std::filesystem::path& model_dir,
                         const Tokenizer& tokenizer,
                         const std::string& device,
                         const ov::AnyMap device_config);

    InputsEmbedderLfm2Vl(const VLMConfig& vlm_config,
                         const ModelsMap& models_map,
                         const Tokenizer& tokenizer,
                         const std::filesystem::path& config_dir_path,
                         const std::string& device,
                         const ov::AnyMap device_config);

    ov::Tensor get_inputs_embeds(const std::string& prompt,
                                 const std::vector<ov::genai::EncodedImage>& images,
                                 ov::genai::VLMPerfMetrics& metrics,
                                 bool recalculate_merged_embeddings = true,
                                 const std::vector<size_t>& image_sequence = {}) override;

    NormalizedPrompt normalize_prompt(const std::string& prompt,
                                      size_t base_id,
                                      const std::vector<EncodedImage>& images) const override;
};

}  // namespace ov::genai
