// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/qwen3_vl/classes.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {

/**
 * @brief Calculates timestamps for video frames based on encoded video metadata.
 * @return Vector of float timestamps corresponding to each video frame.
 */
std::vector<float> calculate_timestamps(const VideoMetadata& video_metadata, size_t merge_size) {
    OPENVINO_ASSERT(video_metadata.fps > 0.0f, "Video metadata fps must be positive for timestamp calculation.");

    // Copy frame_indices since padding may be needed
    std::vector<size_t> frame_indices = video_metadata.frames_indices;
    if (frame_indices.size() % merge_size != 0) {
        frame_indices.resize(
            frame_indices.size() + (merge_size - frame_indices.size() % merge_size),
            frame_indices.back()
        );
    }

    std::vector<float> timestamps;
    timestamps.reserve(frame_indices.size() / merge_size);
    for (size_t i = 0; i < frame_indices.size(); i += merge_size) {
        const float timestamp = (static_cast<float>(frame_indices[i] + frame_indices[i + merge_size - 1]))
            / 2.0f / video_metadata.fps;
        timestamps.push_back(timestamp);
    }
    return timestamps;
}

/**
 * @brief Populates video metadata in encoded_video struct.
 * Computes frame sampling indices for encoded video based on video processor config.
 */
void fill_video_metadata(EncodedVideo& encoded_video, size_t total_num_frames, const VideoProcessorConfig& video_config) {
    OPENVINO_ASSERT(!(video_config.fps != 0.0f && video_config.num_frames != 0),
        "num_frames and fps are mutually exclusive video config arguments.");
    
    if (!video_config.do_sample_frames) {
        encoded_video.metadata.frames_indices.resize(total_num_frames);
        std::iota(encoded_video.metadata.frames_indices.begin(), encoded_video.metadata.frames_indices.end(), 0);
        return;
    }
    // Sample frame indices if needed
    size_t num_frames = video_config.num_frames;
    
    if (num_frames == 0 && video_config.fps != 0.0f) {
        OPENVINO_ASSERT(encoded_video.metadata.fps != 0.0f,
            "Requested to sample frames by fps but video metadata fps is not set. "
            "Provide VideoMetadata with fps or use a fixed num_frames.");

        num_frames = static_cast<size_t>(
            total_num_frames / static_cast<double>(encoded_video.metadata.fps) * static_cast<double>(video_config.fps)
        );
        num_frames = std::clamp(num_frames, video_config.min_frames, std::min(video_config.max_frames, total_num_frames));
    } else if (num_frames == 0) {
        num_frames = std::clamp(total_num_frames, video_config.min_frames, video_config.max_frames);
    }

    OPENVINO_ASSERT(num_frames > 1 && num_frames <= total_num_frames,
        "Invalid number of frames (" + std::to_string(num_frames) +") for video sampling.");

    encoded_video.metadata.frames_indices.reserve(num_frames);
    for (size_t i = 0; i < num_frames; ++i) {
        size_t frame_idx = static_cast<size_t>(std::round(
            static_cast<double>(i) * static_cast<double>(total_num_frames - 1) / static_cast<double>(num_frames - 1)
        ));
        encoded_video.metadata.frames_indices.push_back(frame_idx);
    }
}

/**
 * @brief Computes indices and weights for bilinear position embedding interpolation.
 * @return Pair of:
 *   - indices tensor [4, num_positions] - input for vision_embeddings_pos model
 *   - weights tensor [4, num_positions] - bilinear interpolation weights
 */
std::pair<ov::Tensor, ov::Tensor> get_position_interpolation_indices_and_weights(
    const std::vector<std::array<size_t, 3>>& grids_thw,
    size_t num_grid_per_side
) {
    std::vector<std::vector<int64_t>> indices_list(4);
    std::vector<std::vector<float>> weights_list(4);
    
    for (const auto& grid_thw : grids_thw) {
        const auto [t, h, w] = grid_thw;
        
        // Create linearly spaced indices for h and w
        std::vector<float> h_idxs(h), w_idxs(w);
        const float h_scale = h > 1 ? static_cast<float>(num_grid_per_side - 1) / (h - 1) : 0.0f;
        const float w_scale = w > 1 ? static_cast<float>(num_grid_per_side - 1) / (w - 1) : 0.0f;
        
        for (size_t i = 0; i < h; ++i) {
            h_idxs[i] = static_cast<float>(i) * h_scale;
        }
        for (size_t i = 0; i < w; ++i) {
            w_idxs[i] = static_cast<float>(i) * w_scale;
        }
        
        // Compute floor/ceil indices and interpolation weights
        for (size_t ti = 0; ti < t; ++ti) {
            for (size_t hi = 0; hi < h; ++hi) {
                const int64_t h_floor = static_cast<int64_t>(h_idxs[hi]);
                const int64_t h_ceil = std::min(h_floor + 1, static_cast<int64_t>(num_grid_per_side - 1));
                const float dh = h_idxs[hi] - static_cast<float>(h_floor);
                
                for (size_t wi = 0; wi < w; ++wi) {
                    const int64_t w_floor = static_cast<int64_t>(w_idxs[wi]);
                    const int64_t w_ceil = std::min(w_floor + 1, static_cast<int64_t>(num_grid_per_side - 1));
                    const float dw = w_idxs[wi] - static_cast<float>(w_floor);
                    
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
    
    const size_t total_positions = indices_list[0].size();
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

/**
 * @brief Reorders position embeddings according to spatial merge pattern in vision encoder.
 * 
 * @param pos_embeds Interpolated position embeddings [num_positions, embed_dim]
 * @param grids_thw Grid dimensions for permutation
 * @param spatial_merge_size Spatial merge size from processor config
 * @return Permuted position embeddings [num_merged_positions, embed_dim]
 */
ov::Tensor permute_with_spatial_merge(
    const ov::Tensor& pos_embeds,
    const std::vector<std::array<size_t, 3>>& grids_thw,
    size_t spatial_merge_size
) {
    const size_t num_positions = pos_embeds.get_shape()[0];
    const size_t embed_dim = pos_embeds.get_shape()[1];
    const float* pos_embeds_data = pos_embeds.data<const float>();
    
    std::vector<float> permuted_data;
    permuted_data.reserve(num_positions * embed_dim);
    
    size_t offset = 0;
    for (const auto& grid_thw : grids_thw) {
        const auto [t, h, w] = grid_thw;
        const size_t hw = h * w;
        
        const size_t merge_h = h / spatial_merge_size;
        const size_t merge_w = w / spatial_merge_size;
        
        for (size_t ti = 0; ti < t; ++ti) {
            for (size_t mhi = 0; mhi < merge_h; ++mhi) {
                for (size_t mwi = 0; mwi < merge_w; ++mwi) {
                    for (size_t shi = 0; shi < spatial_merge_size; ++shi) {
                        for (size_t swi = 0; swi < spatial_merge_size; ++swi) {
                            const size_t src_h = mhi * spatial_merge_size + shi;
                            const size_t src_w = mwi * spatial_merge_size + swi;
                            const size_t src_idx = offset + ti * hw + src_h * w + src_w;
                            
                            const float* src = pos_embeds_data + src_idx * embed_dim;
                            permuted_data.insert(permuted_data.end(), src, src + embed_dim);
                        }
                    }
                }
            }
        }
        offset += t * hw;
    }
    
    const size_t permuted_len = permuted_data.size() / embed_dim;
    ov::Tensor result{ov::element::f32, {permuted_len, embed_dim}};
    std::memcpy(result.data<float>(), permuted_data.data(), permuted_data.size() * sizeof(float));
    
    return result;
}

/**
 * @brief Create visual position mask from input_ids by finding vision pad tokens.
 * @return Boolean tensor [batch, seq_len] with true at vision token positions
 */
ov::Tensor create_visual_pos_masks(
    const ov::Tensor& input_ids,
    int64_t image_pad_token_id,
    int64_t video_pad_token_id
) {
    const auto input_ids_shape = input_ids.get_shape();
    ov::Tensor result{ov::element::boolean, input_ids_shape};
    bool* result_data = result.data<bool>();
    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    for (size_t i = 0; i < ov::shape_size(input_ids_shape); ++i) {
        result_data[i] = (input_ids_data[i] == image_pad_token_id || input_ids_data[i] == video_pad_token_id);
    }
    return result;
}

} // namespace

EncodedVideo VisionEncoderQwen3VL::encode_frames(const std::vector<ov::Tensor>& frames, const ov::AnyMap& config_map) {
    EncodedVideo encoded_video;
    
    fill_video_metadata(encoded_video, frames.size(), m_video_processor_config);

    std::vector<ov::Tensor> sampled_frames;
    if (!m_video_processor_config.do_sample_frames) {
        sampled_frames = frames;
    } else {
        sampled_frames.reserve(encoded_video.metadata.frames_indices.size());
        for (size_t idx : encoded_video.metadata.frames_indices) {
            OPENVINO_ASSERT(idx < frames.size(),
                            "Frame index ", idx, " out of range for ", frames.size(), " frames.");
            sampled_frames.push_back(frames.at(idx));
        }
    }

    VisionEncoderQwen2VL::encode_frames_with_config(encoded_video, sampled_frames, m_video_processor_config);

    return encoded_video;
}

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

void InputsEmbedderQwen3VL::expand_video_tags_in_prompt(
    std::string& unified_prompt,
    const std::vector<EncodedVideo>& encoded_videos,
    const std::vector<size_t>& videos_sequence,
    size_t video_base_id
) const {
    std::vector<std::array<size_t, 3>> video_grid_thw_list;
    video_grid_thw_list.reserve(encoded_videos.size());

    for (const auto& encoded_video : encoded_videos) {
        size_t grid_t = encoded_video.frame_num;
        OPENVINO_ASSERT(grid_t > 0, "Video input must contain at least one frame.");
        size_t grid_h = encoded_video.resized_source_size.height;
        size_t grid_w = encoded_video.resized_source_size.width;
        video_grid_thw_list.push_back({grid_t, grid_h, grid_w});
    }

    for (size_t video_id : videos_sequence) {
        auto [grid_t, grid_h, grid_w] = video_grid_thw_list.at(video_id - video_base_id);
        // Calculate number of video pad tokens for each frame
        const size_t num_video_pad_tokens = calc_tokens_num(1, grid_h, grid_w);

        const auto& encoded_video = encoded_videos.at(video_id - video_base_id);
        const size_t spatial_merge_size = m_vision_encoder->get_processor_config().merge_size;
        auto timestamps = calculate_timestamps(encoded_video.metadata, spatial_merge_size);
        OPENVINO_ASSERT(timestamps.size() == grid_t, "Timestamps size does not match the number of frames");

        std::string expanded_tag;
        for (size_t grid_t_idx = 0; grid_t_idx < grid_t; ++grid_t_idx) {
            std::stringstream timestamp_ss;
            timestamp_ss << std::fixed << std::setprecision(1) << timestamps[grid_t_idx];
            const std::string timestamp_str = "<" + timestamp_ss.str() + " seconds>";
            expanded_tag.append(timestamp_str);
            expanded_tag.append(m_vlm_config.vision_start_token);
            for (size_t i = 0; i < num_video_pad_tokens; ++i) {
                expanded_tag.append(m_vlm_config.video_pad_token);
            }
            expanded_tag.append(m_vlm_config.vision_end_token);
        }

        unified_prompt.replace(unified_prompt.find(NATIVE_VIDEO_TAG), NATIVE_VIDEO_TAG.length(), expanded_tag);
    }
}

ov::Tensor InputsEmbedderQwen3VL::get_interpolated_pos_embeds(
    const std::vector<std::array<size_t, 3>>& grids_thw
) {
    const size_t num_grid_per_side = static_cast<size_t>(
        std::sqrt(static_cast<double>(m_vlm_config.vision_config_num_position_embeddings)));
    
    auto [indices, weights] = get_position_interpolation_indices_and_weights(grids_thw, num_grid_per_side);
    
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
    return permute_with_spatial_merge(weighted_sum, grids_thw, spatial_merge_size);
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

std::vector<std::array<size_t, 3>> InputsEmbedderQwen3VL::get_vision_grid_thw_for_position_ids(
    const std::vector<std::array<size_t, 3>>& images_grid_thw,
    const std::vector<size_t>& images_sequence,
    const size_t image_id,
    const std::vector<std::array<size_t, 3>>& videos_grid_thw,
    const std::vector<size_t>& videos_sequence,
    const size_t video_id,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count
) const {
    auto reordered_vision_grid_thw = InputsEmbedderQwen2VL::get_vision_grid_thw_for_position_ids(
        images_grid_thw,
        images_sequence,
        image_id,
        videos_grid_thw,
        videos_sequence,
        video_id,
        history_vision_count
    );

    // Split video grids per each frame for position_ids calculation as Qwen3-VL uses timestamp tokens between frames.
    std::vector<std::array<size_t, 3>> flattened_vision_grid_thw;

    for (const auto& vision_grid_thw : reordered_vision_grid_thw) {
        auto [grid_t, grid_h, grid_w] = vision_grid_thw;

        if (grid_t > 1) {
            for (size_t frame_idx = 0; frame_idx < grid_t; ++frame_idx) {
                flattened_vision_grid_thw.push_back({1, grid_h, grid_w});
            }
        } else {
            flattened_vision_grid_thw.push_back(vision_grid_thw);
        }
    }
    return flattened_vision_grid_thw;
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
        const size_t hidden_size = text_embeds.get_shape()[2];
        ov::Tensor deepstack_visual_embeds(ov::element::f32, {num_layers, 1, hidden_size});
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

    // Apply CDPruner if active
    ov::Tensor merged_video_embeddings_tensor = m_merged_video_embeddings;
    ov::Tensor merged_image_embeddings_tensor = m_merged_image_embeddings;
    
    auto apply_pruning = [&](size_t vision_count,
                             const std::vector<std::array<size_t, 3>>& grid_thw,
                             const std::vector<size_t>& sequence,
                             ov::Tensor& merged_embeddings,
                             int64_t vision_pad_token_id) {
        // Calculate tokens per vision input
        std::vector<size_t> tokens_per_vision;
        tokens_per_vision.reserve(grid_thw.size());
        for (const auto& [grid_t, grid_h, grid_w] : grid_thw) {
            tokens_per_vision.push_back(calc_tokens_num(grid_t, grid_h, grid_w));
        }

        const size_t spatial_merge_size = std::max<size_t>(1, m_vision_encoder->get_processor_config().merge_size);

        PruningContext pruning_context{input_ids,
                                       text_embeds,
                                       merged_embeddings,
                                       vision_count,
                                       grid_thw,
                                       sequence,
                                       tokens_per_vision,
                                       vision_pad_token_id,
                                       vision_start_token_id,
                                       vision_end_token_id,
                                       spatial_merge_size};

        if (auto pruning_result = execute_pruning_pipeline(pruning_context)) {
            merged_embeddings = pruning_result->pruned_embeddings;
            input_ids = pruning_result->pruned_input_ids;
            text_embeds = pruning_result->pruned_text_embeds;

            if (pruning_result->updated_rope_delta.has_value()) {
                m_rope_delta = pruning_result->updated_rope_delta.value();
            }
            
            // Log pruning results
            GENAI_INFO("[Qwen3-VL] [CDPruner] Visual tokens: %zu -> %zu (pruned %zu, %.1f%%)",
                       pruning_result->original_visual_tokens,
                       pruning_result->pruned_visual_tokens,
                       pruning_result->original_visual_tokens - pruning_result->pruned_visual_tokens,
                       100.0 * (pruning_result->original_visual_tokens - pruning_result->pruned_visual_tokens) / 
                       pruning_result->original_visual_tokens);
            
            // Diagnostic: check grid_thw vs actual vision tokens in pruned input_ids
            {
                const size_t pruned_seq_len = input_ids.get_shape().at(1);
                const int64_t* pruned_ids = input_ids.data<const int64_t>();
                size_t actual_vision_tokens = 0;
                for (size_t i = 0; i < pruned_seq_len; ++i) {
                    if (pruned_ids[i] == vision_pad_token_id) actual_vision_tokens++;
                }
                size_t grid_expected_tokens = 0;
                for (const auto& [gt, gh, gw] : grid_thw) {
                    grid_expected_tokens += calc_tokens_num(gt, gh, gw);
                }
                GENAI_INFO("[Qwen3-VL] [CDPruner] [DIAG] Before create_position_ids: "
                           "pruned input_ids seq_len=%zu, actual_vision_pad_tokens=%zu, "
                           "grid_thw expected tokens=%zu",
                           pruned_seq_len, actual_vision_tokens, grid_expected_tokens);
                if (grid_expected_tokens != actual_vision_tokens) {
                    GENAI_INFO("[Qwen3-VL] [CDPruner] [DIAG] WARNING: grid_thw expects %zu vision tokens "
                               "but pruned input_ids only has %zu! create_position_ids will OVERFLOW!",
                               grid_expected_tokens, actual_vision_tokens);
                }
            }
            
            // TODO(BUG): grid_thw still describes original unpruned grid dimensions,
            // but input_ids has been pruned. create_position_ids will write
            // grid_expected_tokens worth of position IDs but position_ids tensor
            // is only allocated for pruned_seq_len. This causes heap buffer overflow!
            m_position_ids = create_position_ids(input_ids, grid_thw, sequence, 0, 
                                                 std::vector<std::array<size_t, 3>>{}, 
                                                 std::vector<size_t>{}, 0, 
                                                 vision_start_token_id, 
                                                 std::vector<std::pair<size_t, size_t>>{});
            GENAI_INFO("[Qwen3-VL] [CDPruner] [DIAG] create_position_ids returned (may have overflowed)");
            
            // Log pruned position_ids
            auto pruned_pos_shape = m_position_ids.get_shape();
            const int64_t* pruned_pos_data = m_position_ids.data<const int64_t>();
            size_t pruned_total_size = m_position_ids.get_size();
            
            int64_t pruned_pos_max = *std::max_element(pruned_pos_data, pruned_pos_data + pruned_total_size);
            GENAI_INFO("[Qwen3-VL] [CDPruner] Pruned Position IDs Max: %ld, Updated RoPE Delta: %ld", 
                       pruned_pos_max, m_rope_delta);
            
            // Prune deepstack_visual_embeds
            // deepstack_visual_embeds shape: [num_layers, total_vision_tokens, hidden_size]
            auto& deepstack_embeds = m_lm_extra_inputs["deepstack_visual_embeds"];
            GENAI_INFO("[Qwen3-VL] [CDPruner] [DEBUG] Checking deepstack_embeds existence...");
            
            if (deepstack_embeds && deepstack_embeds.get_size() > 0) {
                auto deepstack_shape = deepstack_embeds.get_shape();
                GENAI_INFO("[Qwen3-VL] [CDPruner] [DEBUG] deepstack_shape.size() = %zu", deepstack_shape.size());
                
                if (deepstack_shape.size() == 3) {
                    size_t num_layers = deepstack_shape[0];
                    size_t original_tokens = deepstack_shape[1];
                    size_t hidden_size = deepstack_shape[2];
                    
                    GENAI_INFO("[Qwen3-VL] [CDPruner] [DEBUG] Original deepstack shape: [%zu, %zu, %zu]",
                               num_layers, original_tokens, hidden_size);
                    GENAI_INFO("[Qwen3-VL] [CDPruner] [DEBUG] pruning_result->pruned_visual_tokens = %zu",
                               pruning_result->pruned_visual_tokens);
                    GENAI_INFO("[Qwen3-VL] [CDPruner] [DEBUG] Number of regions: %zu",
                               pruning_result->keep_flags_per_region.size());
                    
                    // Build keep flags for all vision tokens across all regions
                    std::vector<bool> global_keep_flags;
                    for (size_t region_idx = 0; region_idx < pruning_result->keep_flags_per_region.size(); ++region_idx) {
                        const auto& region_flags = pruning_result->keep_flags_per_region[region_idx];
                        GENAI_INFO("[Qwen3-VL] [CDPruner] [DEBUG] Region %zu: %zu flags",
                                   region_idx, region_flags.size());
                        global_keep_flags.insert(global_keep_flags.end(), region_flags.begin(), region_flags.end());
                    }
                    
                    GENAI_INFO("[Qwen3-VL] [CDPruner] [DEBUG] Total global_keep_flags.size() = %zu",
                               global_keep_flags.size());
                    
                    // Count actual kept tokens
                    size_t actual_kept = std::count(global_keep_flags.begin(), global_keep_flags.end(), true);
                    GENAI_INFO("[Qwen3-VL] [CDPruner] [DEBUG] Actual kept tokens = %zu", actual_kept);
                    
                    // Validate that keep flags match original tokens
                    if (global_keep_flags.size() != original_tokens) {
                        GENAI_INFO("[Qwen3-VL] [CDPruner] [ERROR] Keep flags size (%zu) != original tokens (%zu)",
                                   global_keep_flags.size(), original_tokens);
                        OPENVINO_ASSERT(false, "Keep flags size mismatch");
                    }
                    
                    // Count actual kept tokens to validate against pruned_visual_tokens
                    if (actual_kept != pruning_result->pruned_visual_tokens) {
                        GENAI_INFO("[Qwen3-VL] [CDPruner] [ERROR] Actual kept (%zu) != pruned_visual_tokens (%zu)",
                                   actual_kept, pruning_result->pruned_visual_tokens);
                        OPENVINO_ASSERT(false, "Actual kept tokens mismatch");
                    }
                    
                    if (pruning_result->pruned_visual_tokens < original_tokens) {
                        GENAI_INFO("[Qwen3-VL] [CDPruner] [DEBUG] Creating pruned tensor [%zu, %zu, %zu]",
                                   num_layers, pruning_result->pruned_visual_tokens, hidden_size);
                        
                        // Create pruned deepstack tensor
                        ov::Tensor pruned_deepstack(deepstack_embeds.get_element_type(), 
                                                    {num_layers, pruning_result->pruned_visual_tokens, hidden_size});
                        
                        const float* src_data = deepstack_embeds.data<const float>();
                        float* dst_data = pruned_deepstack.data<float>();
                        
                        GENAI_INFO("[Qwen3-VL] [CDPruner] [DEBUG] Starting per-layer pruning...");
                        
                        // Prune each layer
                        for (size_t layer = 0; layer < num_layers; ++layer) {
                            size_t dst_idx = 0;
                            for (size_t token = 0; token < original_tokens; ++token) {
                                if (global_keep_flags[token]) {
                                    if (dst_idx >= pruning_result->pruned_visual_tokens) {
                                        GENAI_INFO("[Qwen3-VL] [CDPruner] [ERROR] Layer %zu: dst_idx (%zu) >= pruned_visual_tokens (%zu)",
                                                   layer, dst_idx, pruning_result->pruned_visual_tokens);
                                        OPENVINO_ASSERT(false, "dst_idx overflow");
                                    }
                                    
                                    size_t src_offset = (layer * original_tokens + token) * hidden_size;
                                    size_t dst_offset = (layer * pruning_result->pruned_visual_tokens + dst_idx) * hidden_size;
                                    
                                    // Validate offsets
                                    size_t src_size = deepstack_embeds.get_size();
                                    size_t dst_size = pruned_deepstack.get_size();
                                    if (src_offset + hidden_size > src_size) {
                                        GENAI_INFO("[Qwen3-VL] [CDPruner] [ERROR] src_offset overflow: %zu + %zu > %zu",
                                                   src_offset, hidden_size, src_size);
                                        OPENVINO_ASSERT(false, "src_offset overflow");
                                    }
                                    if (dst_offset + hidden_size > dst_size) {
                                        GENAI_INFO("[Qwen3-VL] [CDPruner] [ERROR] dst_offset overflow: %zu + %zu > %zu",
                                                   dst_offset, hidden_size, dst_size);
                                        OPENVINO_ASSERT(false, "dst_offset overflow");
                                    }
                                    
                                    std::memcpy(dst_data + dst_offset, src_data + src_offset, hidden_size * sizeof(float));
                                    dst_idx++;
                                }
                            }
                            
                            GENAI_INFO("[Qwen3-VL] [CDPruner] [DEBUG] Layer %zu: copied %zu tokens", layer, dst_idx);
                            
                            if (dst_idx != pruning_result->pruned_visual_tokens) {
                                GENAI_INFO("[Qwen3-VL] [CDPruner] [ERROR] Layer %zu: dst_idx (%zu) != pruned_visual_tokens (%zu)",
                                           layer, dst_idx, pruning_result->pruned_visual_tokens);
                                OPENVINO_ASSERT(false, "dst_idx final mismatch");
                            }
                        }
                        
                        GENAI_INFO("[Qwen3-VL] [CDPruner] [DEBUG] All layers pruned successfully, moving tensor...");
                        
                        m_lm_extra_inputs["deepstack_visual_embeds"] = std::move(pruned_deepstack);
                        
                        GENAI_INFO("[Qwen3-VL] [CDPruner] Pruned deepstack_visual_embeds: [%zu, %zu, %zu] -> [%zu, %zu, %zu]",
                                   num_layers, original_tokens, hidden_size,
                                   num_layers, pruning_result->pruned_visual_tokens, hidden_size);
                    } else {
                        GENAI_INFO("[Qwen3-VL] [CDPruner] [DEBUG] No pruning needed (pruned >= original)");
                    }
                } else {
                    GENAI_INFO("[Qwen3-VL] [CDPruner] [DEBUG] deepstack shape is not 3D, skipping");
                }
            } else {
                GENAI_INFO("[Qwen3-VL] [CDPruner] [DEBUG] deepstack_embeds is empty or null");
            }
        }
    };

    // Apply pruning to images
    if (!images.empty() && is_cdpruner_active()) {
        apply_pruning(images.size(),
                      images_grid_thw,
                      images_sequence,
                      merged_image_embeddings_tensor,
                      image_pad_token_id);
        m_merged_image_embeddings = merged_image_embeddings_tensor;
        GENAI_INFO("[Qwen3-VL] [DIAG] After apply_pruning: input_ids=[%zu,%zu], text_embeds=[%zu,%zu,%zu], "
                   "merged_image_embeds=[%zu,%zu]",
                   input_ids.get_shape()[0], input_ids.get_shape()[1],
                   text_embeds.get_shape()[0], text_embeds.get_shape()[1], text_embeds.get_shape()[2],
                   m_merged_image_embeddings.get_shape()[0], m_merged_image_embeddings.get_shape()[1]);
    }

    // TODO: Apply pruning to videos when video pruning is supported

    GENAI_INFO("[Qwen3-VL] [DIAG] Before create_visual_pos_masks");
    // Recalculate visual_pos_masks after potential pruning
    m_lm_extra_inputs["visual_pos_masks"] = create_visual_pos_masks(input_ids, image_pad_token_id, video_pad_token_id);
    GENAI_INFO("[Qwen3-VL] [DIAG] visual_pos_masks created, shape=[%zu,%zu]",
               m_lm_extra_inputs["visual_pos_masks"].get_shape()[0],
               m_lm_extra_inputs["visual_pos_masks"].get_shape()[1]);

    // Count vision tokens for merge validation
    {
        const int64_t* ids = input_ids.data<const int64_t>();
        size_t seq = input_ids.get_shape()[1];
        size_t img_count = 0, vid_count = 0;
        for (size_t i = 0; i < seq; ++i) {
            if (ids[i] == image_pad_token_id) img_count++;
            if (ids[i] == video_pad_token_id) vid_count++;
        }
        GENAI_INFO("[Qwen3-VL] [DIAG] Before merge: input_ids has %zu image_pad, %zu video_pad tokens; "
                   "image_embeds=%zu tokens, video_embeds=%zu tokens",
                   img_count, vid_count,
                   m_merged_image_embeddings.get_shape()[0],
                   m_merged_video_embeddings.get_size() > 0 ? m_merged_video_embeddings.get_shape()[0] : 0);
    }

    GENAI_INFO("[Qwen3-VL] [DIAG] Calling merge_text_and_video_image_embeddings...");
    auto result = qwen2_vl_utils::merge_text_and_video_image_embeddings(
        input_ids, text_embeds, m_merged_image_embeddings, m_merged_video_embeddings,
        image_pad_token_id, video_pad_token_id);
    GENAI_INFO("[Qwen3-VL] [DIAG] merge completed, result shape=[%zu,%zu,%zu]",
               result.get_shape()[0], result.get_shape()[1], result.get_shape()[2]);
    return result;
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
