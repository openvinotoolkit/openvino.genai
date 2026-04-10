// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/qwen3_5/classes.hpp"

#include "visual_language/qwen2vl/classes.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {

const std::unordered_map<std::string, ov::Tensor> g_empty_extra_inputs;

} // namespace

InputsEmbedderQwen3_5::InputsEmbedderQwen3_5(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config)
    : InputsEmbedderQwen3VL(vlm_config, model_dir, device, device_config) {}

InputsEmbedderQwen3_5::InputsEmbedderQwen3_5(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config)
    : InputsEmbedderQwen3VL(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderQwen3_5::run_video_image_embeddings_merger(
    const std::vector<EncodedImage>& images,
    const std::vector<size_t>& images_sequence,
    const std::vector<EncodedVideo>& videos,
    const std::vector<size_t>& videos_sequence
) {
    auto [reordered_image_embeds, reordered_images_grid_thw] =
        qwen2_vl_utils::reorder_image_embeds_and_grid_thw(images, images_sequence);
    auto [reordered_video_embeds, reordered_videos_grid_thw] =
        qwen2_vl_utils::reorder_video_embeds_and_grid_thw(videos, videos_sequence);

    ov::Tensor concatenated_embeds =
        qwen2_vl_utils::concatenate_video_image_embeds(reordered_video_embeds, reordered_image_embeds);

    std::vector<std::array<size_t, 3>> combined_grid_thw;
    combined_grid_thw.insert(combined_grid_thw.end(),
        reordered_videos_grid_thw.begin(), reordered_videos_grid_thw.end());
    combined_grid_thw.insert(combined_grid_thw.end(),
        reordered_images_grid_thw.begin(), reordered_images_grid_thw.end());

    // Add interpolated position embeddings (reused from Qwen3-VL parent)
    if (!combined_grid_thw.empty()) {
        ov::Tensor pos_embeds = get_interpolated_pos_embeds(combined_grid_thw);

        float* concat_data = concatenated_embeds.data<float>();
        const float* pos_data = pos_embeds.data<const float>();
        for (size_t i = 0; i < concatenated_embeds.get_size(); ++i) {
            concat_data[i] += pos_data[i];
        }
    }

    ov::Tensor rotary_pos_emb = get_rotary_pos_emb(combined_grid_thw);

    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(
        this->m_ireq_queue_vision_embeddings_merger.get());
    ov::InferRequest& merger = infer_request_guard.get();

    merger.set_tensor("hidden_states", concatenated_embeds);

    if (m_with_cu_seqlens_input) {
        merger.set_tensor("cu_seq_lens",
            qwen2_vl_utils::get_cu_seqlens(reordered_images_grid_thw, reordered_videos_grid_thw));
    } else {
        merger.set_tensor("attention_mask",
            qwen2_vl_utils::get_attention_mask(reordered_images_grid_thw, reordered_videos_grid_thw));
    }

    merger.set_tensor("rotary_pos_emb", rotary_pos_emb);
    merger.infer();

    // Qwen3.5 merger outputs only "last_hidden_state" (no deepstack_feature_lists)
    ov::Tensor vision_embeds = merger.get_tensor("last_hidden_state");
    auto vision_embeds_shape = vision_embeds.get_shape();

    size_t video_tokens = calc_vec_tokens_num(reordered_videos_grid_thw);
    size_t image_tokens = calc_vec_tokens_num(reordered_images_grid_thw);
    size_t total_tokens = video_tokens + image_tokens;

    size_t video_token_count = 0;
    if (total_tokens > 0) {
        video_token_count = vision_embeds_shape[0] * video_tokens / total_tokens;
    }
    size_t image_token_count = vision_embeds_shape[0] - video_token_count;

    ov::Tensor video_embeds{vision_embeds.get_element_type(), {video_token_count, vision_embeds_shape[1]}};
    ov::Tensor image_embeds{vision_embeds.get_element_type(), {image_token_count, vision_embeds_shape[1]}};

    std::memcpy(video_embeds.data(), vision_embeds.data(), video_embeds.get_byte_size());
    std::memcpy(image_embeds.data(),
                static_cast<uint8_t*>(vision_embeds.data()) + video_embeds.get_byte_size(),
                image_embeds.get_byte_size());

    return {video_embeds, image_embeds};
}

ov::Tensor InputsEmbedderQwen3_5::get_inputs_embeds(
    const std::string& unified_prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    const std::vector<ov::genai::EncodedVideo>& videos,
    ov::genai::VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& images_sequence,
    const std::vector<size_t>& videos_sequence,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count
) {
    std::vector<std::array<size_t, 3>> images_grid_thw;
    images_grid_thw.reserve(images.size());
    for (const auto& encoded_image : images) {
        images_grid_thw.push_back({
            1,
            encoded_image.resized_source_size.height,
            encoded_image.resized_source_size.width
        });
    }

    std::vector<std::array<size_t, 3>> videos_grid_thw;
    videos_grid_thw.reserve(videos.size());
    for (const auto& encoded_video : videos) {
        videos_grid_thw.push_back({
            encoded_video.frame_num,
            encoded_video.resized_source_size.height,
            encoded_video.resized_source_size.width
        });
    }

    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(
        m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    int64_t vision_start_token_id = m_vision_token_ids.at("vision_start");
    int64_t image_pad_token_id = m_vision_token_ids.at("image_pad");
    int64_t video_pad_token_id = m_vision_token_ids.at("video_pad");

    m_position_ids = create_position_ids(input_ids, images_grid_thw, images_sequence, 0,
                                         videos_grid_thw, videos_sequence, 0,
                                         vision_start_token_id, history_vision_count);

    int64_t position_ids_max = *std::max_element(
        m_position_ids.data<int64_t>(),
        m_position_ids.data<int64_t>() + m_position_ids.get_size());
    m_rope_delta = position_ids_max + 1 - static_cast<int64_t>(input_ids.get_shape().at(1));

    if (images.empty() && videos.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    if (recalculate_merged_embeddings) {
        std::tie(m_merged_video_embeddings, m_merged_image_embeddings) =
            run_video_image_embeddings_merger(images, images_sequence, videos, videos_sequence);
    }

    return qwen2_vl_utils::merge_text_and_video_image_embeddings(
        input_ids, text_embeds, m_merged_image_embeddings, m_merged_video_embeddings,
        image_pad_token_id, video_pad_token_id);
}

const std::unordered_map<std::string, ov::Tensor>& InputsEmbedderQwen3_5::get_lm_extra_inputs() const {
    return g_empty_extra_inputs;
}

void InputsEmbedderQwen3_5::start_chat(const std::string& system_message) {
    InputsEmbedderQwen2VL::start_chat(system_message);
}

void InputsEmbedderQwen3_5::finish_chat() {
    InputsEmbedderQwen2VL::finish_chat();
}

} // namespace ov::genai
