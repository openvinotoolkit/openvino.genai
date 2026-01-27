// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "visual_language/vlm_config.hpp"

#include "visual_language/vision_encoder.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "circular_buffer_queue.hpp"
#include "visual_language/cdpruner/cdpruner.hpp"

namespace ov::genai {

class VisionEncoderQwen2VL : public VisionEncoder {
public:
    explicit VisionEncoderQwen2VL(const std::filesystem::path& model_dir, const std::string& device, const ov::AnyMap properties);
    explicit VisionEncoderQwen2VL(const ModelsMap& models_map, const std::filesystem::path& config_dir_path, const std::string& device, const ov::AnyMap properties);

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map) override;
    EncodedVideo encode_frames(const std::vector<ov::Tensor>& frames, const ov::AnyMap& config_map) override;

private:
    void encode_with_imagepreprocess_cpp(const std::vector<ov::Tensor>& image,
                                                 const ov::AnyMap& config_map,
                                                 ov::Tensor& out_tensor,
                                                 ImageSize& out_rsz_size,
                                                 size_t frame_num = 1,
                                                 size_t frame_id = 0);
    void encode_with_imagepreprocess_ov(const std::vector<ov::Tensor>& image,
                                        const ov::AnyMap& config_map,
                                        ov::Tensor& out_tensor,
                                        ImageSize& out_rsz_size,
                                        size_t frame_num = 1,
                                        size_t frame_id = 0);

    bool use_ov_vision_preprocess = true; // default use ov vision preprocess, control by env VISION_PREPROCESS=CPP to use cpp vision preprocess
};

class InputsEmbedderQwen2VL : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderQwen2VL(
        const VLMConfig& vlm_config,
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config);

    InputsEmbedderQwen2VL(
        const VLMConfig& vlm_config,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer, 
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config);

    ov::Tensor get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings = true, const std::vector<size_t>& image_sequence = {}) override;
    ov::Tensor get_inputs_embeds(const std::string& prompt,
                                 const std::vector<ov::genai::EncodedImage>& images,
                                 const std::vector<ov::genai::EncodedVideo>& videos,
                                 ov::genai::VLMPerfMetrics& metrics,
                                 bool recalculate_merged_embeddings = true,
                                 const std::vector<size_t>& image_sequence = {},
                                 const std::vector<size_t>& videos_sequence = {},
                                 const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count = {}) override;

    std::vector<ov::genai::EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;

    std::vector<ov::genai::EncodedVideo> encode_videos(const std::vector<ov::Tensor>& videos) override;

    std::pair<ov::Tensor, std::optional<int64_t>> get_position_ids(const size_t inputs_embeds_size, const size_t history_size) override;

    std::pair<ov::Tensor, std::optional<int64_t>> get_generation_phase_position_ids(const size_t inputs_embeds_size, const size_t history_size, int64_t rope_delta) override;

    void start_chat(const std::string& system_message) override;

    void finish_chat() override;

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images) const override {
        auto norm_prompt = normalize_prompt(prompt, base_id, 0, images, {});
        return {norm_prompt.unified_prompt, norm_prompt.images_sequence};
    }

    NormalizedPrompt normalize_prompt(
        const std::string& prompt,
        size_t image_base_id,
        size_t video_base_id,
        const std::vector<EncodedImage>& images,
        const std::vector<EncodedVideo>& videos) const override;

protected:
    // A model for merging image embeddings (hidden states), rotary_pos_emb and attension_mask.
    // Inputs:
    //  - hidden_states: [N, embed_dim]
    //  - rotary_pos_emb: [?, 40]
    //  - attention_mask: [1, ?, ?]
    // Output: [N, hidden_size]
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_ireq_queue_vision_embeddings_merger;
    ov::Tensor m_merged_image_embeddings;
    ov::Tensor m_merged_video_embeddings;
    std::map<std::string, int64_t> m_vision_token_ids;
    size_t m_merge_length;

    bool m_with_cu_seqlens_input = false;

    virtual std::pair<ov::Tensor, ov::Tensor> run_video_image_embeddings_merger(
        const std::vector<EncodedImage>& images, 
        const std::vector<size_t>& images_sequence,
        const std::vector<EncodedVideo>& videos,
        const std::vector<size_t>& videos_sequence);

    ov::Tensor get_rotary_pos_emb(const std::vector<std::array<size_t, 3>>& grids_thw);

    ov::Tensor create_position_ids(
        const ov::Tensor& input_ids_tensor,
        const std::vector<std::array<size_t, 3>>& images_grid_thw,
        const std::vector<size_t>& images_sequence,
        const size_t image_id,
        const std::vector<std::array<size_t, 3>>& videos_grid_thw,
        const std::vector<size_t>& videos_sequence,
        const size_t video_id,
        const int64_t vision_start_token_id,
        const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count
    );

    void encode_vision_placeholder_tokens();

    void cvt_to_3_chn_image(ov::Tensor& image);

    size_t calc_tokens_num(size_t grid_t, size_t grid_h, size_t grid_w) const;

    size_t calc_vec_tokens_num(const std::vector<std::array<size_t, 3UL>>& vec_grid_thw) const;
};

namespace qwen2_vl_utils {

std::pair<std::vector<ov::Tensor>, std::vector<std::array<size_t, 3>>> reorder_image_embeds_and_grid_thw(
    const std::vector<EncodedImage>& encoded_images,
    const std::vector<size_t>& images_sequence);
std::pair<std::vector<ov::Tensor>, std::vector<std::array<size_t, 3>>> reorder_video_embeds_and_grid_thw(
    const std::vector<EncodedVideo>& videos,
    const std::vector<size_t>& videos_sequence);

ov::Tensor get_attention_mask(const std::vector<std::array<size_t, 3>>& reordered_images_grid_thw, const std::vector<std::array<size_t, 3>>& reordered_videos_grid_thw);
ov::Tensor get_cu_seqlens(const std::vector<std::array<size_t, 3>>& reordered_images_grid_thw, const std::vector<std::array<size_t, 3>>& reordered_videos_grid_thw);

ov::Tensor concatenate_video_image_embeds(const std::vector<ov::Tensor>& reordered_video_embeds, const std::vector<ov::Tensor>& reordered_image_embeds);

} // namespace qwen2vl_utils

} // namespace ov::genai
