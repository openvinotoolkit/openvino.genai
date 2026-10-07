// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

class VisionEncoderMiniCPM : public VisionEncoder {
    // A resampler model to resample image embeddings.
    // [N, H*W, old_hidden_size] is the input shape.
    // [N, query_num, hidden_size] is the output shape.
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_resampler;
    // Precomputed positional embeddings for the resampler.
    // [70, 70, hidden_size]. 70 is the initial guess of the image
    // height and width after dividing by patch_size.
    ov::Tensor m_pos_embed_cache;
    // VLM config
    VLMConfig m_vlm_config;

    ov::Tensor resample(const ov::Tensor& encoded_image, const ImageSize& target_size, size_t pad_to_max);

    ResampledImage resample_encoded_image(const EncodedImage& image, const ov::Tensor& slices, const ImageSize& target_sizes);
public:
    VisionEncoderMiniCPM(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap properties);


    VisionEncoderMiniCPM(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);
    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

class InputsEmbedderMiniCPM : public InputsEmbedder::IInputsEmbedder {

public:
    InputsEmbedderMiniCPM(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap device_config);
    
    InputsEmbedderMiniCPM(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {}) override;

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images
    ) const override;

protected:
    // MiniCPM-V's chat template omits BOS, while some MiniCPM tokenizers are configured
    // to prepend it (add_bos_token=true, e.g. MiniCPM-V-4). For those, GenAI must keep the
    // tokenizer's special tokens enabled even when a chat template is applied so tokenization
    // matches the HF/optimum reference (leading <s>). Other MiniCPM variants (e.g. the
    // Qwen2-based 2.6 / o-2.6) do not prepend BOS and must stay unchanged. The decision is
    // derived from the tokenizer itself (probed once in the constructor), not hard-coded.
    bool add_special_tokens_when_chat_templated() const override { return m_tokenizer_prepends_special_token; }

private:
    // True when the tokenizer prepends a leading special token (BOS) under add_special_tokens.
    bool m_tokenizer_prepends_special_token = false;

    // Probe the tokenizer to detect whether it prepends a leading special (BOS) token.
    static bool probe_tokenizer_prepends_special_token(const Tokenizer& tokenizer);

};

} // namespace ov::genai
