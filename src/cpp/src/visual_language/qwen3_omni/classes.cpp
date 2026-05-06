// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/qwen3_omni/classes.hpp"

#include <cmath>
#include <cstring>

#include "utils.hpp"
#include "visual_language/clip.hpp"

namespace ov::genai {

// --- VisionEncoderQwen3Omni ---

VisionEncoderQwen3Omni::VisionEncoderQwen3Omni(const std::filesystem::path& model_dir,
                                               const std::string& device,
                                               const ov::AnyMap properties)
    : VisionEncoderQwen3VL(model_dir, ConfigOnlyTag{}) {}

VisionEncoderQwen3Omni::VisionEncoderQwen3Omni(const ModelsMap& models_map,
                                               const std::filesystem::path& config_dir_path,
                                               const std::string& device,
                                               const ov::AnyMap properties)
    : VisionEncoderQwen3VL(models_map, config_dir_path, ConfigOnlyTag{}) {}

void VisionEncoderQwen3Omni::preprocess_to_patches(const std::vector<ov::Tensor>& images,
                                                   const ProcessorConfig& config,
                                                   ov::Tensor& out_tensor,
                                                   ImageSize& out_rsz_size,
                                                   size_t frame_num,
                                                   size_t frame_id) {
    OPENVINO_ASSERT(config.temporal_patch_size == 2u, "temporal_patch_size != 2.");
    if (images.size() > 1) {
        OPENVINO_ASSERT(config.temporal_patch_size == images.size(), "temporal_patch_size != images.size()");
    }

    const auto& orig_shape = images[0].get_shape();
    const auto target_image_size = qwen2_vl_utils::smart_resize(orig_shape.at(1),
                                                                orig_shape.at(2),
                                                                config.patch_size * config.merge_size,
                                                                config.min_pixels,
                                                                config.max_pixels);

    ov::Tensor tiled_patches(ov::element::f32,
                             {config.temporal_patch_size, 3, target_image_size.height, target_image_size.width});

    for (size_t i = 0; i < config.temporal_patch_size; i++) {
        const auto& image = images.size() > i ? images[i] : images[0];
        auto input_image = tensor_to_clip_image_u8(image);
        clip_image_u8 resized_image;
        bicubic_resize(input_image, resized_image, target_image_size.width, target_image_size.height);
        clip_ctx ctx;
        std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
        std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);
        auto normalized_image = clip_image_preprocess(ctx, resized_image);
        auto patch = clip_image_f32_to_tensor(normalized_image);
        std::memcpy(tiled_patches.data<float>() + i * patch.get_byte_size() / sizeof(float),
                    patch.data<float>(),
                    patch.get_byte_size());
    }

    const auto channel = tiled_patches.get_shape().at(1);
    const auto grid_t = tiled_patches.get_shape().at(0) / config.temporal_patch_size;
    const auto grid_h = target_image_size.height / config.patch_size;
    const auto grid_w = target_image_size.width / config.patch_size;

    auto reshaped = qwen2_vl_utils::reshape_image_patches(tiled_patches,
                                                          grid_t,
                                                          grid_h,
                                                          grid_w,
                                                          channel,
                                                          config.temporal_patch_size,
                                                          config.patch_size,
                                                          config.merge_size);
    auto transposed = qwen2_vl_utils::transpose_image_patches(reshaped);

    ov::Shape flat_shape{grid_t * grid_h * grid_w,
                         channel * config.temporal_patch_size * config.patch_size * config.patch_size};
    ov::Tensor flattened(transposed.get_element_type(), flat_shape);
    std::memcpy(flattened.data(), transposed.data(), transposed.get_byte_size());

    // Accumulate into output tensor (same pattern as VisionEncoderQwen2VL)
    if (frame_id == 0u) {
        auto out_shape = flat_shape;
        out_shape[0] = flat_shape[0] * frame_num;
        out_tensor = ov::Tensor(flattened.get_element_type(), out_shape);
    }
    std::memcpy(reinterpret_cast<uint8_t*>(out_tensor.data()) + frame_id * flattened.get_byte_size(),
                flattened.data(),
                flattened.get_byte_size());
    out_rsz_size = ImageSize{grid_h, grid_w};
}

EncodedImage VisionEncoderQwen3Omni::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    EncodedImage encoded_img;
    preprocess_to_patches({image},
                          m_processor_config,
                          encoded_img.resized_source,
                          encoded_img.resized_source_size,
                          1,
                          0);
    return encoded_img;
}

EncodedVideo VisionEncoderQwen3Omni::encode_frames(const std::vector<ov::Tensor>& frames,
                                                   const ov::AnyMap& config_map) {
    EncodedVideo encoded_video;
    const auto& config = m_video_processor_config;

    fill_video_metadata(encoded_video, frames.size(), config);

    std::vector<ov::Tensor> sampled_frames;
    if (!config.do_sample_frames) {
        sampled_frames = frames;
    } else {
        sampled_frames.reserve(encoded_video.metadata.frames_indices.size());
        for (size_t idx : encoded_video.metadata.frames_indices) {
            OPENVINO_ASSERT(idx < frames.size(), "Frame index ", idx, " out of range for ", frames.size(), " frames.");
            sampled_frames.push_back(frames.at(idx));
        }
    }

    const auto frames_size = sampled_frames.size();
    encoded_video.frame_num = (frames_size + config.temporal_patch_size - 1) / config.temporal_patch_size;

    size_t frame_id = 0;
    size_t i = 0;
    for (; i + config.temporal_patch_size <= frames_size; i += config.temporal_patch_size) {
        preprocess_to_patches(std::vector<ov::Tensor>(sampled_frames.begin() + i,
                                                      sampled_frames.begin() + i + config.temporal_patch_size),
                              config,
                              encoded_video.video_features,
                              encoded_video.resized_source_size,
                              encoded_video.frame_num,
                              frame_id);
        frame_id++;
    }
    for (; i < frames_size; i++) {
        preprocess_to_patches({sampled_frames[i]},
                              config,
                              encoded_video.video_features,
                              encoded_video.resized_source_size,
                              encoded_video.frame_num,
                              frame_id);
        frame_id++;
    }
    return encoded_video;
}

// --- InputsEmbedderQwen3Omni ---

InputsEmbedderQwen3Omni::InputsEmbedderQwen3Omni(const VLMConfig& vlm_config,
                                                 const std::filesystem::path& model_dir,
                                                 const std::string& device,
                                                 const ov::AnyMap device_config)
    : InputsEmbedderQwen3VL(vlm_config, model_dir, device, device_config),
      m_audio_token_id(vlm_config.audio_token_id) {
    // Audio encoder is optional — check is_available() / has_audio_encoder() before encoding
    m_audio_encoder = std::make_unique<AudioEncoderQwen3Omni>(model_dir, vlm_config, device, device_config);

    // Merged vision model is optional — check has_merged_vision_model() before calling
    // run_video_image_embeddings_merger(). Without it the model works as text+audio only.
    auto vision_model_path = model_dir / "openvino_vision_embeddings_model.xml";
    if (std::filesystem::exists(vision_model_path)) {
        auto model = utils::singleton_core().read_model(vision_model_path);
        auto compiled = utils::singleton_core().compile_model(model, device, device_config);
        m_ireq_queue_merged_vision = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
            compiled.get_property(ov::optimal_number_of_infer_requests),
            [&compiled]() -> ov::InferRequest {
                return compiled.create_infer_request();
            });
        // Cache rotary dim to avoid queue lock in get_rotary_pos_emb
        auto rotary_pshape = compiled.input("rotary_pos_emb").get_partial_shape();
        m_rotary_dim = static_cast<size_t>(rotary_pshape[rotary_pshape.rank().get_length() - 1].get_length());
    }
}

InputsEmbedderQwen3Omni::InputsEmbedderQwen3Omni(const VLMConfig& vlm_config,
                                                 const ModelsMap& models_map,
                                                 const Tokenizer& tokenizer,
                                                 const std::filesystem::path& config_dir_path,
                                                 const std::string& device,
                                                 const ov::AnyMap device_config)
    : InputsEmbedderQwen3VL(vlm_config, models_map, tokenizer, config_dir_path, device, device_config),
      m_audio_token_id(vlm_config.audio_token_id) {
    // Audio encoder is optional — check is_available() / has_audio_encoder() before encoding
    m_audio_encoder = std::make_unique<AudioEncoderQwen3Omni>(config_dir_path, vlm_config, device, device_config);

    // Merged vision model is optional — check has_merged_vision_model() before calling
    // run_video_image_embeddings_merger(). Without it the model works as text+audio only.
    if (models_map.count("vision_embeddings")) {
        const auto& [model_str, weights] = utils::get_model_weights_pair(models_map, "vision_embeddings");
        auto model = utils::singleton_core().read_model(model_str, weights);
        auto compiled = utils::singleton_core().compile_model(model, device, device_config);
        m_ireq_queue_merged_vision = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
            compiled.get_property(ov::optimal_number_of_infer_requests),
            [&compiled]() -> ov::InferRequest {
                return compiled.create_infer_request();
            });
        // Cache rotary dim to avoid queue lock in get_rotary_pos_emb
        auto rotary_pshape = compiled.input("rotary_pos_emb").get_partial_shape();
        m_rotary_dim = static_cast<size_t>(rotary_pshape[rotary_pshape.rank().get_length() - 1].get_length());
    }
}

void InputsEmbedderQwen3Omni::encode_audios(const std::vector<ov::Tensor>& audios) {
    if (audios.empty() || !has_audio_encoder()) {
        m_audio_embeddings = ov::Tensor();
        return;
    }

    std::vector<ov::Tensor> all_features;
    size_t total_tokens = 0;
    size_t hidden_size = 0;

    for (const auto& audio : audios) {
        auto features = m_audio_encoder->encode(audio);
        total_tokens += features.get_shape()[0];
        hidden_size = features.get_shape()[1];
        all_features.push_back(features);
    }

    m_audio_embeddings = ov::Tensor(ov::element::f32, {total_tokens, hidden_size});
    auto* dst = m_audio_embeddings.data<float>();
    for (const auto& feat : all_features) {
        auto byte_size = feat.get_byte_size();
        std::memcpy(dst, feat.data<float>(), byte_size);
        dst += feat.get_size();
    }
}

ov::Tensor InputsEmbedderQwen3Omni::get_inputs_embeds(
    const std::string& prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    const std::vector<ov::genai::EncodedVideo>& videos,
    ov::genai::VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& image_sequence,
    const std::vector<size_t>& videos_sequence,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    auto input_embeds = InputsEmbedderQwen3VL::get_inputs_embeds(prompt,
                                                                 images,
                                                                 videos,
                                                                 metrics,
                                                                 recalculate_merged_embeddings,
                                                                 image_sequence,
                                                                 videos_sequence,
                                                                 history_vision_count);

    // Capture input_ids set by parent's get_inputs_embeds() into a local variable,
    // making the cross-class data dependency explicit rather than relying on
    // implicit ordering of m_last_input_ids population.
    std::vector<int64_t> input_ids_vec(m_last_input_ids.data<int64_t>(),
                                       m_last_input_ids.data<int64_t>() + m_last_input_ids.get_size());

    // If we have audio embeddings, replace audio token positions
    if (m_audio_embeddings && m_audio_embeddings.get_size() > 0 && m_audio_token_id >= 0) {
        merge_audio_embeddings(input_embeds, input_ids_vec);
    }

    return input_embeds;
}

void InputsEmbedderQwen3Omni::merge_audio_embeddings(ov::Tensor& input_embeds, const std::vector<int64_t>& input_ids) {
    if (!m_audio_embeddings || m_audio_embeddings.get_size() == 0) {
        return;
    }

    const auto& shape = input_embeds.get_shape();
    const auto seq_len = shape[1];
    const auto hidden_size = shape[2];

    const auto audio_hidden_size = m_audio_embeddings.get_shape()[1];
    OPENVINO_ASSERT(audio_hidden_size == hidden_size,
                    "Audio embedding hidden_size (",
                    audio_hidden_size,
                    ") must match input embedding hidden_size (",
                    hidden_size,
                    "). Check that audio encoder output dimension matches the language model.");

    OPENVINO_ASSERT(input_ids.size() >= seq_len,
                    "input_ids size (",
                    input_ids.size(),
                    ") must be >= embedding seq_len (",
                    seq_len,
                    "). Ensure input_ids are not from a stale or re-tokenized source.");

    auto* embed_data = input_embeds.data<float>();
    const auto* audio_data = m_audio_embeddings.data<const float>();
    const auto audio_total_tokens = m_audio_embeddings.get_shape()[0];
    const size_t bytes_per_token = hidden_size * sizeof(float);
    size_t audio_idx = 0;

    for (size_t i = 0; i < seq_len && audio_idx < audio_total_tokens; i++) {
        if (input_ids[i] == m_audio_token_id) {
            std::memcpy(embed_data + i * hidden_size, audio_data + audio_idx * hidden_size, bytes_per_token);
            audio_idx++;
        }
    }
}

NormalizedPrompt InputsEmbedderQwen3Omni::normalize_prompt(const std::string& prompt,
                                                           size_t base_id,
                                                           const std::vector<EncodedImage>& images) const {
    auto result = normalize_prompt(prompt, base_id, 0, images, {});
    return {result.unified_prompt, result.images_sequence};
}

NormalizedPrompt InputsEmbedderQwen3Omni::normalize_prompt(const std::string& prompt,
                                                           size_t image_base_id,
                                                           size_t video_base_id,
                                                           const std::vector<EncodedImage>& images,
                                                           const std::vector<EncodedVideo>& videos) const {
    auto result = InputsEmbedderQwen3VL::normalize_prompt(prompt, image_base_id, video_base_id, images, videos);

    if (m_audio_embeddings && m_audio_embeddings.get_size() > 0) {
        const auto num_audio_tokens = m_audio_embeddings.get_shape()[0];

        const std::string audio_start = "<|audio_start|>";
        const std::string audio_pad = "<|audio_pad|>";
        const std::string audio_end = "<|audio_end|>";

        if (result.unified_prompt.find(audio_start) == std::string::npos) {
            std::string audio_tag;
            audio_tag.reserve(audio_start.size() + audio_pad.size() * num_audio_tokens + audio_end.size());
            audio_tag.append(audio_start);
            for (size_t i = 0; i < num_audio_tokens; ++i) {
                audio_tag.append(audio_pad);
            }
            audio_tag.append(audio_end);
            result.unified_prompt = audio_tag + result.unified_prompt;
        }
    }

    return result;
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderQwen3Omni::run_video_image_embeddings_merger(
    const std::vector<EncodedImage>& images,
    const std::vector<size_t>& images_sequence,
    const std::vector<EncodedVideo>& videos,
    const std::vector<size_t>& videos_sequence) {
    OPENVINO_ASSERT(has_merged_vision_model(),
                    "Merged vision model not loaded but images/videos were provided. "
                    "Ensure openvino_vision_embeddings_model.xml is present in the model directory.");

    auto [reordered_image_embeds, reordered_images_grid_thw] =
        qwen2_vl_utils::reorder_image_embeds_and_grid_thw(images, images_sequence);
    auto [reordered_video_embeds, reordered_videos_grid_thw] =
        qwen2_vl_utils::reorder_video_embeds_and_grid_thw(videos, videos_sequence);

    // These are raw patches now (not features) — shape [total_patches, patch_dim]
    auto concatenated_patches =
        qwen2_vl_utils::concatenate_video_image_embeds(reordered_video_embeds, reordered_image_embeds);

    std::vector<std::array<size_t, 3>> combined_grid_thw;
    combined_grid_thw.insert(combined_grid_thw.end(),
                             reordered_videos_grid_thw.begin(),
                             reordered_videos_grid_thw.end());
    combined_grid_thw.insert(combined_grid_thw.end(),
                             reordered_images_grid_thw.begin(),
                             reordered_images_grid_thw.end());

    // Compute pos_embeds as separate input (model adds them internally after Conv3d)
    ov::Tensor pos_embeds;
    if (!combined_grid_thw.empty()) {
        pos_embeds = get_interpolated_pos_embeds(combined_grid_thw);
    }

    auto attention_mask =
        qwen2_vl_utils::get_attention_mask(reordered_images_grid_thw, reordered_videos_grid_thw);
    auto rotary_pos_emb = get_rotary_pos_emb(combined_grid_thw);

    CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue_merged_vision.get());
    auto& ireq = guard.get();

    ireq.set_tensor("hidden_states", concatenated_patches);
    if (pos_embeds) {
        ireq.set_tensor("pos_embeds", pos_embeds);
    }
    ireq.set_tensor("attention_mask", attention_mask);
    ireq.set_tensor("rotary_pos_emb", rotary_pos_emb);
    ireq.infer();

    auto vision_embeds = ireq.get_tensor("last_hidden_state");
    m_lm_extra_inputs["deepstack_visual_embeds"] = ireq.get_tensor("deepstack_feature_lists");

    const auto& vision_embeds_shape = vision_embeds.get_shape();

    const auto video_tokens = calc_vec_tokens_num(reordered_videos_grid_thw);
    const auto image_tokens = calc_vec_tokens_num(reordered_images_grid_thw);
    const auto total_tokens = video_tokens + image_tokens;

    size_t video_count = 0;
    if (total_tokens > 0) {
        video_count = vision_embeds_shape[0] * video_tokens / total_tokens;
    }
    const auto image_count = vision_embeds_shape[0] - video_count;

    ov::Tensor video_embeds{vision_embeds.get_element_type(), {video_count, vision_embeds_shape[1]}};
    ov::Tensor image_embeds{vision_embeds.get_element_type(), {image_count, vision_embeds_shape[1]}};

    std::memcpy(video_embeds.data(), vision_embeds.data(), video_embeds.get_byte_size());
    std::memcpy(image_embeds.data(),
                static_cast<uint8_t*>(vision_embeds.data()) + video_embeds.get_byte_size(),
                image_embeds.get_byte_size());

    return {video_embeds, image_embeds};
}

ov::Tensor InputsEmbedderQwen3Omni::get_rotary_pos_emb(const std::vector<std::array<size_t, 3>>& grids_thw) const {
    const auto spatial_merge_size = m_vision_encoder->get_processor_config().merge_size;

    std::vector<std::vector<size_t>> all_pos_ids;
    size_t max_grid_size = 0;

    for (const auto& grid_thw : grids_thw) {
        const auto t = grid_thw.at(0);
        const auto h = grid_thw.at(1);
        const auto w = grid_thw.at(2);

        max_grid_size = std::max({max_grid_size, h, w});

        std::vector<size_t> hpos_ids;
        std::vector<size_t> wpos_ids;
        const auto h_blocks = h / spatial_merge_size;
        const auto w_blocks = w / spatial_merge_size;
        hpos_ids.reserve(h * w);
        wpos_ids.reserve(h * w);

        for (size_t hb = 0; hb < h_blocks; ++hb) {
            for (size_t wb = 0; wb < w_blocks; ++wb) {
                for (size_t hs = 0; hs < spatial_merge_size; ++hs) {
                    for (size_t ws = 0; ws < spatial_merge_size; ++ws) {
                        hpos_ids.push_back(hb * spatial_merge_size + hs);
                        wpos_ids.push_back(wb * spatial_merge_size + ws);
                    }
                }
            }
        }

        for (size_t i = 0; i < t; ++i) {
            for (size_t j = 0; j < hpos_ids.size(); ++j) {
                all_pos_ids.push_back({hpos_ids[j], wpos_ids[j]});
            }
        }
    }

    // Use cached rotary dim — Qwen3Omni uses a merged vision model instead of the separate
    // merger model that Qwen2VL/Qwen3VL use, so we cannot reuse the parent get_rotary_pos_emb()
    // which obtains dim via queue lock on m_ireq_queue_vision_embeddings_merger.
    OPENVINO_ASSERT(m_rotary_dim > 0, "Rotary dim not initialized — merged vision model not loaded");
    const auto dim = m_rotary_dim;
    const auto half_dim = dim / 2;
    constexpr float theta = 10000.0f;

    std::vector<float> inv_freq(half_dim);
    for (size_t i = 0; i < half_dim; ++i) {
        inv_freq[i] = 1.0f / std::pow(theta, static_cast<float>(i) / static_cast<float>(half_dim));
    }

    std::vector<std::vector<float>> freqs(max_grid_size);
    for (size_t i = 0; i < max_grid_size; ++i) {
        freqs[i].resize(half_dim);
        for (size_t j = 0; j < half_dim; ++j) {
            freqs[i][j] = static_cast<float>(i) * inv_freq[j];
        }
    }

    const size_t half_dim_bytes = half_dim * sizeof(float);
    ov::Tensor rotary_pos_emb(ov::element::f32, {all_pos_ids.size(), dim});
    auto* output_data = rotary_pos_emb.data<float>();

    for (size_t i = 0; i < all_pos_ids.size(); ++i) {
        const auto& pos = all_pos_ids[i];
        const auto h_idx = pos[0];
        const auto w_idx = pos[1];
        std::memcpy(output_data + i * dim, freqs[h_idx].data(), half_dim_bytes);
        std::memcpy(output_data + i * dim + half_dim, freqs[w_idx].data(), half_dim_bytes);
    }

    return rotary_pos_emb;
}

void InputsEmbedderQwen3Omni::start_chat(const std::string& system_message) {
    InputsEmbedderQwen3VL::start_chat(system_message);
    m_audio_embeddings = ov::Tensor();
}

void InputsEmbedderQwen3Omni::finish_chat() {
    InputsEmbedderQwen3VL::finish_chat();
    m_audio_embeddings = ov::Tensor();
}

}  // namespace ov::genai
