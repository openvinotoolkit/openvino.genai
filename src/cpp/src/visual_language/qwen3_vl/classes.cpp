// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/qwen3_vl/classes.hpp"
#include "utils.hpp"
#include "logger.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace ov::genai {

namespace {

constexpr float DEFAULT_METADATA_FPS = 24.0f;

/**
 * @brief Calculates timestamps for video frames based on encoded video metadata.
 * @return Vector of float timestamps corresponding to each video frame.
 */
std::vector<float> calculate_timestamps(const VideoMetadata& video_metadata, size_t merge_size) {
    OPENVINO_ASSERT(video_metadata.fps >= 0.0f, "Video metadata fps must be non-negative for timestamp calculation.");

    float metadata_fps = video_metadata.fps;
    if (metadata_fps == 0.0f) {
        GENAI_WARN("Qwen3-VL requires frame timestamps to construct prompts, but VideoMetadata is missing or fps is not set. "
            "Defaulting to 24 fps. Please provide VideoMetadata with fps for more accurate results.");
        metadata_fps = DEFAULT_METADATA_FPS;
    }

    // Copy frames_indices since padding may be needed
    std::vector<size_t> frames_indices = video_metadata.frames_indices;
    if (frames_indices.size() % merge_size != 0) {
        frames_indices.resize(
            frames_indices.size() + (merge_size - frames_indices.size() % merge_size),
            frames_indices.back()
        );
    }

    std::vector<float> timestamps;
    timestamps.reserve(frames_indices.size() / merge_size);
    for (size_t i = 0; i < frames_indices.size(); i += merge_size) {
        const float timestamp = (static_cast<float>(frames_indices[i] + frames_indices[i + merge_size - 1]))
            / 2.0f / metadata_fps;
        timestamps.push_back(timestamp);
    }
    return timestamps;
}

/**
 * @brief Populates video metadata and computes frame sampling indices based on video processor config.
 */
void fill_video_metadata(VideoMetadata& video_metadata, size_t total_num_frames, const VideoProcessorConfig& video_config) {
    if (!video_metadata.frames_indices.empty()) {
        GENAI_WARN("Frames indices already provided in video metadata, skipping Qwen3-VL model-specific sampling.");
        return;
    }

    OPENVINO_ASSERT(!(video_config.fps != 0.0f && video_config.num_frames != 0),
        "num_frames and fps are mutually exclusive video config arguments.");
    
    if (!video_config.do_sample_frames) {
        // frames_indices is still needed for timestamp calculation
        video_metadata.frames_indices.resize(total_num_frames);
        std::iota(video_metadata.frames_indices.begin(), video_metadata.frames_indices.end(), 0);
        return;
    }

    // Sample frame indices if needed
    size_t num_frames = video_config.num_frames;
    
    if (num_frames == 0 && video_config.fps != 0.0f) {
        if (video_metadata.fps == 0.0f) {
            GENAI_WARN("Requested to sample frames by fps, but video metadata fps is not set. "
                "Defaulting to 24 fps for frame sampling. "
                "Please provide VideoMetadata with fps for more accurate results.");
            video_metadata.fps = DEFAULT_METADATA_FPS;
        }

        num_frames = static_cast<size_t>(
            total_num_frames / static_cast<double>(video_metadata.fps) * static_cast<double>(video_config.fps)
        );
        num_frames = std::clamp(num_frames, video_config.min_frames, std::min(video_config.max_frames, total_num_frames));
    } else if (num_frames == 0) {
        num_frames = std::clamp(total_num_frames, video_config.min_frames, video_config.max_frames);
    }

    OPENVINO_ASSERT(num_frames > 1 && num_frames <= total_num_frames,
        "Invalid number of frames (" + std::to_string(num_frames) +") for video sampling.");

    video_metadata.frames_indices.reserve(num_frames);
    for (size_t i = 0; i < num_frames; ++i) {
        size_t frame_idx = static_cast<size_t>(std::round(
            static_cast<double>(i) * static_cast<double>(total_num_frames - 1) / static_cast<double>(num_frames - 1)
        ));
        video_metadata.frames_indices.push_back(frame_idx);
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
    // Pre-compute total positions to allocate tensors directly
    size_t total_positions = 0;
    for (const auto& grid_thw : grids_thw) {
        total_positions += grid_thw[0] * grid_thw[1] * grid_thw[2];
    }

    ov::Tensor indices{ov::element::i64, {4, total_positions}};
    ov::Tensor weights{ov::element::f32, {4, total_positions}};
    int64_t* indices_data = indices.data<int64_t>();
    float* weights_data = weights.data<float>();
    size_t pos_offset = 0;

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
        const int64_t grid_max = static_cast<int64_t>(num_grid_per_side - 1);
        for (size_t ti = 0; ti < t; ++ti) {
            for (size_t hi = 0; hi < h; ++hi) {
                const int64_t h_floor = static_cast<int64_t>(h_idxs[hi]);
                const int64_t h_ceil = std::min(h_floor + 1, grid_max);
                const float dh = h_idxs[hi] - static_cast<float>(h_floor);
                const int64_t h_floor_row = h_floor * num_grid_per_side;
                const int64_t h_ceil_row = h_ceil * num_grid_per_side;

                for (size_t wi = 0; wi < w; ++wi) {
                    const int64_t w_floor = static_cast<int64_t>(w_idxs[wi]);
                    const int64_t w_ceil = std::min(w_floor + 1, grid_max);
                    const float dw = w_idxs[wi] - static_cast<float>(w_floor);

                    // 4 corners: (floor,floor), (floor,ceil), (ceil,floor), (ceil,ceil)
                    indices_data[0 * total_positions + pos_offset] = h_floor_row + w_floor;
                    indices_data[1 * total_positions + pos_offset] = h_floor_row + w_ceil;
                    indices_data[2 * total_positions + pos_offset] = h_ceil_row + w_floor;
                    indices_data[3 * total_positions + pos_offset] = h_ceil_row + w_ceil;

                    // Bilinear weights
                    weights_data[0 * total_positions + pos_offset] = (1.0f - dh) * (1.0f - dw);
                    weights_data[1 * total_positions + pos_offset] = (1.0f - dh) * dw;
                    weights_data[2 * total_positions + pos_offset] = dh * (1.0f - dw);
                    weights_data[3 * total_positions + pos_offset] = dh * dw;
                    ++pos_offset;
                }
            }
        }
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
    const size_t embed_bytes = embed_dim * sizeof(float);

    ov::Tensor result{ov::element::f32, {num_positions, embed_dim}};
    float* dst_data = result.data<float>();
    size_t dst_offset = 0;
    size_t src_offset = 0;

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
                            const size_t src_idx = src_offset + ti * hw + src_h * w + src_w;

                            std::memcpy(dst_data + dst_offset * embed_dim,
                                        pos_embeds_data + src_idx * embed_dim,
                                        embed_bytes);
                            ++dst_offset;
                        }
                    }
                }
            }
        }
        src_offset += t * hw;
    }
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

bool check_vision_pos_embeds_env() {
    const char* env = std::getenv("VISION_POS_EMBEDS");
    return !(env && std::string(env) == "CPP");
}

/**
 * @brief Patches vision_embeddings_pos model to perform weighted sum on device.
 *
 * Original: input [4, N] → model → [4, N, D] (then CPU weighted sum)
 * Patched:  input [4, N] + weights [4, N] → model → [4, N, D]
 *           → per-corner Multiply [N,D]×[N,1] → sequential Add → [N, D]
 */
std::shared_ptr<ov::Model> patch_weighted_sum_into_pos_model(
    const std::shared_ptr<ov::Model>& model_org
) {
    auto results = model_org->get_results();
    OPENVINO_ASSERT(results.size() == 1u, "Expected single output from vision_embeddings_pos model");
    auto orig_output = results[0]->input_value(0); // [4, N, D]

    auto weights_param = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::PartialShape{4, -1});
    weights_param->set_friendly_name("weights");
    weights_param->output(0).get_tensor().set_names({"weights"});

    constexpr size_t num_corners = 4;
    auto axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto sq_axis0 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto unsq_axis1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});

    // Split pos_embeds [4,N,D] → 4×[N,D] and weights [4,N] → 4×[N,1]
    auto split_e = std::make_shared<ov::op::v1::Split>(orig_output, axis0, num_corners);
    auto split_w = std::make_shared<ov::op::v1::Split>(weights_param, axis0, num_corners);

    // Per-corner: Squeeze → Multiply → sequential Add
    ov::Output<ov::Node> summed;
    for (size_t i = 0; i < num_corners; ++i) {
        auto embed = std::make_shared<ov::op::v0::Squeeze>(split_e->output(i), sq_axis0);     // [N, D]
        auto weight = std::make_shared<ov::op::v0::Unsqueeze>(
            std::make_shared<ov::op::v0::Squeeze>(split_w->output(i), sq_axis0), unsq_axis1); // [N, 1]
        auto weighted = std::make_shared<ov::op::v1::Multiply>(embed, weight);                // [N, D]

        summed = (i == 0) ? weighted->output(0)
                          : std::make_shared<ov::op::v1::Add>(summed, weighted)->output(0);
    }

    auto params = model_org->get_parameters();
    params.push_back(weights_param);

    return std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(summed)}, params);
}

} // namespace

EncodedVideo VisionEncoderQwen3VL::encode_frames(const std::vector<ov::Tensor>& frames) {
    EncodedVideo encoded_video;
    VisionEncoderQwen2VL::encode_frames_with_config(encoded_video, frames, m_video_processor_config);
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
    m_use_patched_pos_model = check_vision_pos_embeds_env();
    if (m_use_patched_pos_model) {
        pos_model = patch_weighted_sum_into_pos_model(pos_model);
    }
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
    m_use_patched_pos_model = check_vision_pos_embeds_env();
    if (m_use_patched_pos_model) {
        pos_model = patch_weighted_sum_into_pos_model(pos_model);
    }
    auto pos_compiled = utils::singleton_core().compile_model(pos_model, device, device_config);
    
    m_ireq_queue_vision_embeddings_pos = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        pos_compiled.get_property(ov::optimal_number_of_infer_requests),
        [&pos_compiled]() -> ov::InferRequest {
            return pos_compiled.create_infer_request();
        }
    );
}

std::vector<ov::genai::EncodedVideo> InputsEmbedderQwen3VL::encode_videos(
    const std::vector<ov::Tensor>& videos,
    const std::vector<VideoMetadata>& videos_metadata
) {
    OPENVINO_ASSERT(videos.size() == videos_metadata.size() || videos_metadata.empty(),
        "Number of videos and videos metadata must match if metadata provided.");

    std::vector<EncodedVideo> encoded_videos;
    for (size_t i = 0; i < videos.size(); ++i) {
        const ov::Tensor& video = videos[i];
        const size_t video_num_frames = video.get_shape()[0];
        VideoMetadata video_metadata = i < videos_metadata.size() ? videos_metadata[i] : VideoMetadata{};
        fill_video_metadata(video_metadata, video_num_frames, m_vision_encoder->get_video_processor_config());
        const auto sampled_video = sample_video_if_needed(video, video_metadata);
        std::vector<ov::Tensor> frames = to_single_image_tensors({sampled_video});
        auto encoded_video = m_vision_encoder->encode_frames(frames);
        encoded_video.metadata = video_metadata;
        encoded_videos.emplace_back(encoded_video);
    }
    return encoded_videos;
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

    ov::Tensor weighted_sum;
    if (m_use_patched_pos_model) {
        // Patched model: pass weights, get [N, D] directly from GPU
        vision_embeddings_pos.set_tensor("weights", weights);
        vision_embeddings_pos.infer();
        weighted_sum = vision_embeddings_pos.get_output_tensor();
    } else {
        // Original model: get [4, N, D], do CPU weighted sum
        vision_embeddings_pos.infer();
        ov::Tensor pos_embeds = vision_embeddings_pos.get_output_tensor();

        size_t num_positions = pos_embeds.get_shape()[1];
        size_t embed_dim = pos_embeds.get_shape()[2];

        // Weighted sum over 4 corners
        weighted_sum = ov::Tensor{ov::element::f32, {num_positions, embed_dim}};
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
    
    if (has_lm_extra_input("deepstack_visual_embeds")) {
        m_lm_extra_inputs["deepstack_visual_embeds"] = vision_embeddings_merger.get_tensor("deepstack_feature_lists");
    }
    
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

    std::tie(m_position_ids, m_rope_delta) = create_position_ids(
        input_ids,
        images_grid_thw,
        images_sequence,
        0,
        videos_grid_thw,
        videos_sequence,
        0,
        vision_start_token_id,
        history_vision_count
    );

    if (images.empty() && videos.empty()) {
        if (has_lm_extra_input("visual_pos_masks")) {
            const size_t batch_size = input_ids.get_shape()[0];
            ov::Tensor visual_pos_masks(ov::element::boolean, {batch_size, 1});
            std::fill_n(visual_pos_masks.data<bool>(), visual_pos_masks.get_size(), false);
            m_lm_extra_inputs["visual_pos_masks"] = std::move(visual_pos_masks);
        }

        if (has_lm_extra_input("deepstack_visual_embeds")) {
            const size_t num_layers = m_vlm_config.vision_config_deepstack_visual_indexes.size();
            const size_t hidden_size = text_embeds.get_shape()[2];
            ov::Tensor deepstack_visual_embeds(ov::element::f32, {num_layers, 1, hidden_size});
            std::fill_n(deepstack_visual_embeds.data<float>(), deepstack_visual_embeds.get_size(), 0.0f);
            m_lm_extra_inputs["deepstack_visual_embeds"] = std::move(deepstack_visual_embeds);
        }

        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    if (recalculate_merged_embeddings) {
        std::tie(m_merged_video_embeddings, m_merged_image_embeddings) = 
            run_video_image_embeddings_merger(images, images_sequence, videos, videos_sequence);
    }

    if (has_lm_extra_input("visual_pos_masks")) {
        m_lm_extra_inputs["visual_pos_masks"] = create_visual_pos_masks(input_ids, image_pad_token_id, video_pad_token_id);
    }

    return qwen2_vl_utils::merge_text_and_video_image_embeddings(
        input_ids, text_embeds, m_merged_image_embeddings, m_merged_video_embeddings,
        image_pad_token_id, video_pad_token_id);
}

void InputsEmbedderQwen3VL::start_chat(const std::string& system_message) {
    InputsEmbedderQwen2VL::start_chat(system_message);
    if (has_lm_extra_input("deepstack_visual_embeds")) {
        m_lm_extra_inputs["deepstack_visual_embeds"] = ov::Tensor();
    }
    if (has_lm_extra_input("visual_pos_masks")) {
        m_lm_extra_inputs["visual_pos_masks"] = ov::Tensor();
    }
}

void InputsEmbedderQwen3VL::finish_chat() {
    InputsEmbedderQwen2VL::finish_chat();
    if (has_lm_extra_input("deepstack_visual_embeds")) {
        m_lm_extra_inputs["deepstack_visual_embeds"] = ov::Tensor();
    }
    if (has_lm_extra_input("visual_pos_masks")) {
        m_lm_extra_inputs["visual_pos_masks"] = ov::Tensor();
    }
}

const std::unordered_map<std::string, ov::Tensor>& InputsEmbedderQwen3VL::get_lm_extra_inputs() const {
    return m_lm_extra_inputs;
}

bool InputsEmbedderQwen3VL::has_lm_extra_input(const std::string& input_name) const {
    const auto& lm_extra_inputs = get_lm_extra_inputs();
    return lm_extra_inputs.find(input_name) != lm_extra_inputs.end();
}

} // namespace ov::genai
