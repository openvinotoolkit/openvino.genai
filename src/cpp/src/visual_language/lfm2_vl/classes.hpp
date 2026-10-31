// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

/// @brief Vision encoder for LiquidAI LFM2-VL (model_type "lfm2_vl").
///
/// The model combines a Siglip2 NaFlex vision tower with SmolVLM-style image
/// tiling. Large images are split into a grid of @c tile_size tiles plus a
/// downscaled thumbnail; small images are resized to a single variable-length
/// patch sequence. Each sub-image is patchified ([num_patches, patch*patch*3])
/// and fed to the exported vision model together with an out-of-graph
/// positional-embedding interpolation matrix, matching optimum-intel's
/// _OVLfm2VlForCausalLM runtime preprocessing.
class VisionEncoderLFM2VL : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

class InputsEmbedderLFM2VL : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderLFM2VL(const VLMConfig& vlm_config,
                         const std::filesystem::path& model_dir,
                         const Tokenizer& tokenizer,
                         const std::string& device,
                         const ov::AnyMap device_config);

    InputsEmbedderLFM2VL(const VLMConfig& vlm_config,
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
