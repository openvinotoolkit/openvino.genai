// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

/// @brief Vision encoder for LiquidAI LFM2-VL (model_type "lfm2_vl").
///
/// LFM2-VL uses a SigLIP2 NaFlex packed-patch vision tower followed by a
/// pixel-unshuffle multimodal projector. The exported
/// openvino_vision_embeddings_model consumes a single image (or tile) worth of
/// packed patches together with its spatial shape and returns the already
/// projected image features [num_downsampled_tokens, text_hidden_size].
///
/// The encoder reproduces Lfm2VlImageProcessor: smart-resize, optional tiling
/// with a thumbnail, rescale/normalize, and patchification. Each tile (and the
/// thumbnail) is inferred independently and the resulting features are
/// concatenated in the exact order the HF processor lays out image tokens.
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

    NormalizedPrompt normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const override;
};

} // namespace ov::genai
