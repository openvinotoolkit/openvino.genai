// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

// Vision encoder for MiniCPM-V-4.6.
//
// The exported openvino_vision_embeddings_model performs the entire image
// feature extraction (NaViT SigLIP encoder, ViT window-attention merger,
// downsample merger) but expects all data-dependent index/mask tensors to be
// provided by the caller. This class reproduces the MiniCPMV4_6ImageProcessor
// preprocessing (slice grid, resize, patchify) and the NaViT packing index/mask
// computation from optimum-intel's _OVMiniCPMV4_6ForCausalLM in C++.
class VisionEncoderMiniCPMV4_6 : public VisionEncoder {
    size_t m_patch_size = 14;
    size_t m_num_patches_per_side = 70;  // image_size / patch_size
    std::array<size_t, 2> m_window_kernel_size{2, 2};
    std::array<size_t, 2> m_merge_kernel_size{2, 2};
    // downsample factor per spatial dimension used to derive image-token count
    // from a patch grid ("16x" -> total /16 -> /4 per side).
    size_t m_token_divisor = 16;

public:
    VisionEncoderMiniCPMV4_6(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap properties);

    VisionEncoderMiniCPMV4_6(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;

private:
    void read_vision_params(const std::filesystem::path& config_dir_path);
};

class InputsEmbedderMiniCPMV4_6 : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderMiniCPMV4_6(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderMiniCPMV4_6(
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

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images) const override;
};

} // namespace ov::genai
