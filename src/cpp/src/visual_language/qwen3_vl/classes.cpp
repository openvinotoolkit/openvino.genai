// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/qwen3_vl/classes.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace qwen3_vl_utils {

std::pair<ov::Tensor, ov::Tensor> get_position_interpolation_indices_and_weights(
    const std::vector<std::array<size_t, 3>>& grids_thw,
    size_t num_grid_per_side
) {
    std::vector<std::vector<int64_t>> indices_list(4);
    std::vector<std::vector<float>> weights_list(4);
    
    for (const auto& grid_thw : grids_thw) {
        size_t t = grid_thw[0];
        size_t h = grid_thw[1];
        size_t w = grid_thw[2];
        
        // Create linearly spaced indices for h and w
        std::vector<float> h_idxs(h), w_idxs(w);
        float h_scale = h > 1 ? static_cast<float>(num_grid_per_side - 1) / (h - 1) : 0.0f;
        float w_scale = w > 1 ? static_cast<float>(num_grid_per_side - 1) / (w - 1) : 0.0f;
        
        for (size_t i = 0; i < h; ++i) {
            h_idxs[i] = static_cast<float>(i) * h_scale;
        }
        for (size_t i = 0; i < w; ++i) {
            w_idxs[i] = static_cast<float>(i) * w_scale;
        }
        
        // Compute floor/ceil indices and interpolation weights
        for (size_t ti = 0; ti < t; ++ti) {
            for (size_t hi = 0; hi < h; ++hi) {
                int64_t h_floor = static_cast<int64_t>(h_idxs[hi]);
                int64_t h_ceil = std::min(h_floor + 1, static_cast<int64_t>(num_grid_per_side - 1));
                float dh = h_idxs[hi] - static_cast<float>(h_floor);
                
                for (size_t wi = 0; wi < w; ++wi) {
                    int64_t w_floor = static_cast<int64_t>(w_idxs[wi]);
                    int64_t w_ceil = std::min(w_floor + 1, static_cast<int64_t>(num_grid_per_side - 1));
                    float dw = w_idxs[wi] - static_cast<float>(w_floor);
                    
                    // 4 corners: (floor,floor), (floor,ceil), (ceil,floor), (ceil,ceil)
                    indices_list[0].push_back(h_floor * num_grid_per_side + w_floor);
                    indices_list[1].push_back(h_floor * num_grid_per_side + w_ceil);
                    indices_list[2].push_back(h_ceil * num_grid_per_side + w_floor);
                    indices_list[3].push_back(h_ceil * num_grid_per_side + w_ceil);
                    
                    // Bilinear weights
                    weights_list[0].push_back((1.0f - dh) * (1.0f - dw));
                    weights_list[1].push_back((1.0f - dh) * dw);
                    weights_list[2].push_back(dh * (1.0f - dw));
                    weights_list[3].push_back(dh * dw);
                }
            }
        }
    }
    
    size_t total_positions = indices_list[0].size();
    ov::Tensor indices{ov::element::i64, {4, total_positions}};
    ov::Tensor weights{ov::element::f32, {4, total_positions}};
    
    int64_t* indices_data = indices.data<int64_t>();
    float* weights_data = weights.data<float>();
    
    for (size_t corner = 0; corner < 4; ++corner) {
        std::memcpy(indices_data + corner * total_positions,
                    indices_list[corner].data(),
                    total_positions * sizeof(int64_t));
        std::memcpy(weights_data + corner * total_positions,
                    weights_list[corner].data(),
                    total_positions * sizeof(float));
    }
    
    return {indices, weights};
}

ov::Tensor permute_with_spatial_merge(
    const ov::Tensor& pos_embeds,
    const std::vector<std::array<size_t, 3>>& grids_thw,
    size_t spatial_merge_size
) {
    size_t num_positions = pos_embeds.get_shape()[0];
    size_t embed_dim = pos_embeds.get_shape()[1];
    const float* pos_embeds_data = pos_embeds.data<const float>();
    
    std::vector<float> permuted_data;
    permuted_data.reserve(num_positions * embed_dim);
    
    size_t offset = 0;
    for (const auto& grid_thw : grids_thw) {
        size_t t = grid_thw[0];
        size_t h = grid_thw[1];
        size_t w = grid_thw[2];
        size_t hw = h * w;
        
        size_t merge_h = h / spatial_merge_size;
        size_t merge_w = w / spatial_merge_size;
        
        for (size_t ti = 0; ti < t; ++ti) {
            for (size_t mhi = 0; mhi < merge_h; ++mhi) {
                for (size_t mwi = 0; mwi < merge_w; ++mwi) {
                    for (size_t shi = 0; shi < spatial_merge_size; ++shi) {
                        for (size_t swi = 0; swi < spatial_merge_size; ++swi) {
                            size_t src_h = mhi * spatial_merge_size + shi;
                            size_t src_w = mwi * spatial_merge_size + swi;
                            size_t src_idx = offset + ti * hw + src_h * w + src_w;
                            
                            const float* src = pos_embeds_data + src_idx * embed_dim;
                            permuted_data.insert(permuted_data.end(), src, src + embed_dim);
                        }
                    }
                }
            }
        }
        offset += t * hw;
    }
    
    size_t permuted_len = permuted_data.size() / embed_dim;
    ov::Tensor result{ov::element::f32, {permuted_len, embed_dim}};
    std::memcpy(result.data<float>(), permuted_data.data(), permuted_data.size() * sizeof(float));
    
    return result;
}

ov::Tensor create_visual_pos_masks(
    const ov::Tensor& input_ids,
    int64_t image_pad_token_id,
    int64_t video_pad_token_id
) {
    auto input_ids_shape = input_ids.get_shape();
    ov::Tensor result{ov::element::boolean, input_ids_shape};
    bool* result_data = result.data<bool>();
    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    for (size_t i = 0; i < ov::shape_size(input_ids_shape); ++i) {
        result_data[i] = (input_ids_data[i] == image_pad_token_id || input_ids_data[i] == video_pad_token_id);
    }
    return result;
}

} // namespace qwen3_vl_utils

InputsEmbedderQwen3VL::InputsEmbedderQwen3VL(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config
) : InputsEmbedderQwen2VL(vlm_config, model_dir, device, device_config) {
    auto pos_model = utils::singleton_core().read_model(
        model_dir / "openvino_vision_embeddings_pos_model.xml");
    auto pos_compiled = utils::singleton_core().compile_model(pos_model, device, device_config);
    
    m_ireq_queue_vision_embeddings_pos = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        pos_compiled.get_property(ov::optimal_number_of_infer_requests),
        [&pos_compiled]() -> ov::InferRequest {
            return pos_compiled.create_infer_request();
        }
    );
}

InputsEmbedderQwen3VL::InputsEmbedderQwen3VL(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config
) : InputsEmbedderQwen2VL(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    const auto& [pos_model_str, pos_weights] = 
        utils::get_model_weights_pair(models_map, "vision_embeddings_pos");
    auto pos_model = utils::singleton_core().read_model(pos_model_str, pos_weights);
    auto pos_compiled = utils::singleton_core().compile_model(pos_model, device, device_config);
    
    m_ireq_queue_vision_embeddings_pos = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        pos_compiled.get_property(ov::optimal_number_of_infer_requests),
        [&pos_compiled]() -> ov::InferRequest {
            return pos_compiled.create_infer_request();
        }
    );
}

ov::Tensor InputsEmbedderQwen3VL::get_interpolated_pos_embeds(
    const std::vector<std::array<size_t, 3>>& grids_thw
) {
    const size_t num_grid_per_side = static_cast<size_t>(
        std::sqrt(static_cast<double>(m_vlm_config.vision_config_num_position_embeddings)));
    
    auto [indices, weights] = qwen3_vl_utils::get_position_interpolation_indices_and_weights(grids_thw, num_grid_per_side);
    
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(m_ireq_queue_vision_embeddings_pos.get());
    ov::InferRequest& vision_embeddings_pos = infer_request_guard.get();

    vision_embeddings_pos.set_tensor("input", indices);
    vision_embeddings_pos.infer();
    ov::Tensor pos_embeds = vision_embeddings_pos.get_output_tensor();

    size_t num_positions = pos_embeds.get_shape()[1];
    size_t embed_dim = pos_embeds.get_shape()[2];

    // Weighted sum over 4 corners
    ov::Tensor weighted_sum{ov::element::f32, {num_positions, embed_dim}};
    float* weighted_sum_data = weighted_sum.data<float>();
    std::fill_n(weighted_sum_data, num_positions * embed_dim, 0.0f);

    const float* pos_embeds_data = pos_embeds.data<const float>();
    const float* weights_data = weights.data<const float>();
    
    // Apply weights and sum: pos_embeds * weights[:, :, None], then sum over dim 0
    for (size_t corner = 0; corner < 4; ++corner) {
        for (size_t pos = 0; pos < num_positions; ++pos) {
            float w = weights_data[corner * num_positions + pos];
            const float* src = pos_embeds_data + (corner * num_positions + pos) * embed_dim;
            float* dst = weighted_sum_data + pos * embed_dim;
            for (size_t d = 0; d < embed_dim; ++d) {
                dst[d] += w * src[d];
            }
        }
    }
    
    size_t spatial_merge_size = m_vision_encoder->get_processor_config().merge_size;
    return qwen3_vl_utils::permute_with_spatial_merge(weighted_sum, grids_thw, spatial_merge_size);
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderQwen3VL::run_video_image_embeddings_merger(
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
    
    // Combined grid for position computation
    std::vector<std::array<size_t, 3>> combined_grid_thw;
    combined_grid_thw.insert(combined_grid_thw.end(), 
        reordered_videos_grid_thw.begin(), reordered_videos_grid_thw.end());
    combined_grid_thw.insert(combined_grid_thw.end(), 
        reordered_images_grid_thw.begin(), reordered_images_grid_thw.end());
    
    if (!combined_grid_thw.empty()) {
        ov::Tensor pos_embeds = get_interpolated_pos_embeds(combined_grid_thw);
        
        float* concatenated_embeds_data = concatenated_embeds.data<float>();
        const float* pos_embeds_data = pos_embeds.data<const float>();
        for (size_t i = 0; i < concatenated_embeds.get_size(); ++i) {
            concatenated_embeds_data[i] += pos_embeds_data[i];
        }
    }
    
    ov::Tensor rotary_pos_emb = get_rotary_pos_emb(combined_grid_thw);
    
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(m_ireq_queue_vision_embeddings_merger.get());
    ov::InferRequest& vision_embeddings_merger = infer_request_guard.get();
    
    vision_embeddings_merger.set_tensor("hidden_states", concatenated_embeds);
    
    if (m_with_cu_seqlens_input) {
        vision_embeddings_merger.set_tensor("cu_seq_lens", 
            qwen2_vl_utils::get_cu_seqlens(reordered_images_grid_thw, reordered_videos_grid_thw));
    } else {
        vision_embeddings_merger.set_tensor("attention_mask",
            qwen2_vl_utils::get_attention_mask(reordered_images_grid_thw, reordered_videos_grid_thw));
    }
    
    vision_embeddings_merger.set_tensor("rotary_pos_emb", rotary_pos_emb);
    vision_embeddings_merger.infer();
    
    ov::Tensor vision_embeds = vision_embeddings_merger.get_tensor("last_hidden_state");
    m_lm_extra_inputs["deepstack_visual_embeds"] = vision_embeddings_merger.get_tensor("deepstack_feature_lists");
    
    auto vision_embeds_shape = vision_embeds.get_shape();
    
    // Split vision embeddings
    size_t video_tokens = calc_vec_tokens_num(reordered_videos_grid_thw);
    size_t image_tokens = calc_vec_tokens_num(reordered_images_grid_thw);
    size_t total_tokens = video_tokens + image_tokens;
    
    size_t video_count = 0;
    if (total_tokens > 0) {
        video_count = vision_embeds_shape[0] * video_tokens / total_tokens;
    }
    size_t image_count = vision_embeds_shape[0] - video_count;
    
    ov::Tensor video_embeds{vision_embeds.get_element_type(), {video_count, vision_embeds_shape[1]}};
    ov::Tensor image_embeds{vision_embeds.get_element_type(), {image_count, vision_embeds_shape[1]}};
    
    std::memcpy(video_embeds.data(), vision_embeds.data(), video_embeds.get_byte_size());
    std::memcpy(image_embeds.data(),
                static_cast<uint8_t*>(vision_embeds.data()) + video_embeds.get_byte_size(),
                image_embeds.get_byte_size());
    
    return {video_embeds, image_embeds};
}

ov::Tensor InputsEmbedderQwen3VL::get_inputs_embeds(
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
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    int64_t vision_start_token_id = m_vision_token_ids.at("vision_start");
    int64_t image_pad_token_id = m_vision_token_ids.at("image_pad");
    int64_t video_pad_token_id = m_vision_token_ids.at("video_pad");

    m_position_ids = create_position_ids(input_ids, images_grid_thw, images_sequence, 0, 
                                         videos_grid_thw, videos_sequence, 0, 
                                         vision_start_token_id, history_vision_count);

    int64_t position_ids_max = *std::max_element(m_position_ids.data<int64_t>(), 
                                                 m_position_ids.data<int64_t>() + m_position_ids.get_size());
    m_rope_delta = position_ids_max + 1 - static_cast<int64_t>(input_ids.get_shape().at(1));

    if (images.empty() && videos.empty()) {
        // visual_pos_masks extra input
        const size_t batch_size = input_ids.get_shape()[0];
        ov::Tensor visual_pos_masks(ov::element::boolean, {batch_size, 1});
        std::fill_n(visual_pos_masks.data<bool>(), visual_pos_masks.get_size(), false);
        m_lm_extra_inputs["visual_pos_masks"] = std::move(visual_pos_masks);

        // deepstack_visual_embeds extra input
        const size_t num_layers = m_vlm_config.vision_config_deepstack_visual_indexes.size();
        const size_t emd_dim = text_embeds.get_shape()[2];
        ov::Tensor deepstack_visual_embeds(ov::element::f32, {num_layers, 1, emd_dim});
        std::fill_n(deepstack_visual_embeds.data<float>(), deepstack_visual_embeds.get_size(), 0.0f);
        m_lm_extra_inputs["deepstack_visual_embeds"] = std::move(deepstack_visual_embeds);

        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    if (recalculate_merged_embeddings) {
        std::tie(m_merged_video_embeddings, m_merged_image_embeddings) = 
            run_video_image_embeddings_merger(images, images_sequence, videos, videos_sequence);
    }

    m_lm_extra_inputs["visual_pos_masks"] = qwen3_vl_utils::create_visual_pos_masks(input_ids, image_pad_token_id, video_pad_token_id);

    return qwen2_vl_utils::merge_text_and_video_image_embeddings(
        input_ids, text_embeds, m_merged_image_embeddings, m_merged_video_embeddings,
        image_pad_token_id, video_pad_token_id);
}

void InputsEmbedderQwen3VL::start_chat(const std::string& system_message) {
    InputsEmbedderQwen2VL::start_chat(system_message);
    m_lm_extra_inputs["deepstack_visual_embeds"] = ov::Tensor();
    m_lm_extra_inputs["visual_pos_masks"] = ov::Tensor();
}

void InputsEmbedderQwen3VL::finish_chat() {
    InputsEmbedderQwen2VL::finish_chat();
    m_lm_extra_inputs["deepstack_visual_embeds"] = ov::Tensor();
    m_lm_extra_inputs["visual_pos_masks"] = ov::Tensor();
}

const std::unordered_map<std::string, ov::Tensor>& InputsEmbedderQwen3VL::get_lm_extra_inputs() const {
    return m_lm_extra_inputs;
}

} // namespace ov::genai
