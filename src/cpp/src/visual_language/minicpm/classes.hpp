// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

class VisionEncoderMiniCPM : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
};

class InputsEmbedderMiniCPM : public InputsEmbedder::IInputsEmbedder {
    // A resampler model to resample image embeddings.
    // [N, H*W, old_hidden_size] is the input shape.
    // [N, query_num, hidden_size] is the output shape.
    ov::InferRequest m_resampler;
    // Precomputed positional embeddings for the resampler.
    // [70, 70, hidden_size]. 70 is the initial guess of the image
    // height and width after dividing by patch_size.
    ov::Tensor m_pos_embed_cache;
    size_t m_prev_image_id = 0;

public:
    InputsEmbedderMiniCPM(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderMiniCPM(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics) override;

    void update_chat_history(const std::string& decoded_results, const ov::genai::GenerationStatus generation_finish_status) override;

    void start_chat(const std::string& system_message) override;

    void finish_chat() override;

    bool prompt_has_image_tag(const std::string& prompt) const override;

private:
    ov::Tensor resample(const ov::Tensor& encoded_image, const std::vector<ImageSize>& target_sizes);
};

} // namespace ov::genai
