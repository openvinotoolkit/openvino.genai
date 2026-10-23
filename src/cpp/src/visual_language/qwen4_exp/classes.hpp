// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "visual_language/qwen3_vl/classes.hpp"
#include "visual_language/qwen3_5/classes.hpp"

namespace ov::genai {

class VisionEncoderQwen4_Exp : public VisionEncoderQwen3VL {
public:
    using VisionEncoderQwen3VL::VisionEncoderQwen3VL;
};

class InputsEmbedderQwen4_Exp : public InputsEmbedderQwen3_5 {
public:
    InputsEmbedderQwen4_Exp(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap device_config
    );

    InputsEmbedderQwen4_Exp(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config
    );

    const std::unordered_map<std::string, ov::Tensor>& get_lm_extra_inputs() const override;

    using InputsEmbedderQwen3_5::get_inputs_embeds;

    ov::Tensor get_inputs_embeds(
        const std::string& prompt,
        const std::vector<EncodedImage>& images,
        const std::vector<EncodedVideo>& videos,
        VLMPerfMetrics& metrics,
        bool recalculate_merged_embeddings = true,
        const std::vector<size_t>& images_sequence = {},
        const std::vector<size_t>& videos_sequence = {},
        const std::vector<std::pair<size_t, size_t>>& history_vision_count = {}
    ) override;

    std::function<ov::Tensor(const ov::Tensor& new_input_ids, const std::vector<int32_t>& beam_indices)>
    get_ngram_embeddings_callback() override;

    std::pair<ov::Tensor, std::optional<int64_t>> get_position_ids(
        size_t inputs_embeds_size,
        size_t history_size
    ) override;

    void start_chat(const std::string& system_message) override;

    void finish_chat() override;

    std::pair<ov::Tensor, std::optional<int64_t>> get_generation_phase_position_ids(
        const size_t inputs_embeds_size,
        const size_t history_size,
        int64_t rope_delta
    ) override;

protected:
    std::pair<ov::Tensor, int64_t> create_position_ids(
        const ov::Tensor& input_ids_tensor,
        const std::vector<std::array<size_t, 3>>& images_grid_thw,
        const std::vector<size_t>& images_sequence,
        const size_t image_id,
        const std::vector<std::array<size_t, 3>>& videos_grid_thw,
        const std::vector<size_t>& videos_sequence,
        const size_t video_id,
        const int64_t vision_start_token_id,
        const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count
    ) override;

private:
    bool m_has_ngram_embeddings = false;
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ngram_embeddings_ireq_queue = nullptr;
    size_t m_ngram_context_length = 0;
    ov::Tensor m_ngram_history;

    void init_ngram_embeddings_model(
        const std::shared_ptr<ov::Model>& model,
        const std::string& device,
        const ov::AnyMap& device_config
    );

    ov::Tensor get_ngram_embeddings(const ov::Tensor& input_ids);
};

} // namespace ov::genai
