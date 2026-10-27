// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/lfm3_vl/classes.hpp"

#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>

#include "openvino/genai/visual_language/perf_metrics.hpp"
#include "visual_language/clip.hpp"
#include "visual_language/qwen2vl/classes.hpp"
#include "visual_language/qwen3_vl/classes.hpp"

namespace ov::genai {

EncodedImage VisionEncoderLFM3VL::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    const ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);
    OPENVINO_ASSERT(config.patch_size > 0 && config.merge_size > 0 && config.temporal_patch_size > 0,
                    "LFM3-VL patch sizes and spatial merge size must be positive.");
    const ov::Shape& shape = image.get_shape();
    OPENVINO_ASSERT(shape.size() == 4 && shape[0] == 1 && shape[1] > 0 && shape[2] > 0 && shape[3] == 3,
                    "LFM3-VL expects a non-empty RGB image with shape [1, height, width, 3].");
    const ImageSize target = qwen2_vl_utils::smart_resize(shape[1], shape[2],
                                                        config.patch_size * config.merge_size,
                                                        config.min_pixels, config.max_pixels);
    auto encoded = encode_frames_with_config({image}, config, target);
    EncodedImage result{std::move(encoded.video_features), encoded.resized_source_size};
    result.num_image_tokens = encoded.num_video_tokens;
    return result;
}

EncodedVideo VisionEncoderLFM3VL::encode_frames(const std::vector<ov::Tensor>& frames) {
    OPENVINO_ASSERT(!frames.empty(), "LFM3-VL video must contain at least one frame.");
    const auto& config = m_video_processor_config;
    OPENVINO_ASSERT(config.patch_size > 0 && config.merge_size > 0 && config.temporal_patch_size > 0 &&
                        config.min_pixels > 0 && config.max_pixels >= config.min_pixels,
                    "LFM3-VL video patch sizes and pixel limits must be positive and ordered.");
    const auto& shape = frames.front().get_shape();
    OPENVINO_ASSERT(shape.size() == 4 && shape[0] == 1 && shape[1] > 0 && shape[2] > 0 && shape[3] == 3,
                    "LFM3-VL video frames must have shape [1, height, width, 3].");
    const size_t factor = config.patch_size * config.merge_size;
    size_t height = shape[1];
    size_t width = shape[2];
    if (height < factor || width < factor) {
        const double scale = std::max(static_cast<double>(factor) / height, static_cast<double>(factor) / width);
        height = static_cast<size_t>(height * scale);
        width = static_cast<size_t>(width * scale);
    }
    OPENVINO_ASSERT(static_cast<double>(std::max(height, width)) / std::min(height, width) <= 200.0,
                    "LFM3-VL video aspect ratio must not exceed 200.");
    const size_t num_frames = std::max(frames.size(), config.temporal_patch_size);
    const size_t rounded_frames =
        static_cast<size_t>(std::nearbyint(static_cast<double>(num_frames) / config.temporal_patch_size)) *
        config.temporal_patch_size;
    ImageSize target{
        static_cast<size_t>(std::nearbyint(static_cast<double>(height) / factor)) * factor,
        static_cast<size_t>(std::nearbyint(static_cast<double>(width) / factor)) * factor,
    };
    if (rounded_frames * target.height * target.width > config.max_pixels) {
        const double beta = std::sqrt(static_cast<double>(num_frames) * height * width / config.max_pixels);
        target.height = std::max(factor, static_cast<size_t>(std::floor(height / beta / factor)) * factor);
        target.width = std::max(factor, static_cast<size_t>(std::floor(width / beta / factor)) * factor);
    } else if (rounded_frames * target.height * target.width < config.min_pixels) {
        const double beta = std::sqrt(config.min_pixels / (static_cast<double>(num_frames) * height * width));
        target.height = static_cast<size_t>(std::ceil(height * beta / factor)) * factor;
        target.width = static_cast<size_t>(std::ceil(width * beta / factor)) * factor;
    }
    return encode_frames_with_config(frames, config, target);
}

EncodedVideo VisionEncoderLFM3VL::encode_frames_with_config(const std::vector<ov::Tensor>& frames,
                                                          const ProcessorConfig& config,
                                                          const ImageSize& target) {
    OPENVINO_ASSERT(!frames.empty(), "LFM3-VL vision encoding requires at least one frame.");
    std::vector<clip_image_u8> resized_frames;
    resized_frames.reserve(frames.size());
    const auto& shape = frames.front().get_shape();
    for (const auto& frame : frames) {
        OPENVINO_ASSERT(frame.get_shape() == shape && frame.get_element_type() == ov::element::u8,
                        "LFM3-VL frames must have the same shape and uint8 element type.");
        clip_image_u8 resized;
        bicubic_resize(tensor_to_clip_image_u8(frame), resized, target.width, target.height);
        resized_frames.emplace_back(std::move(resized));
    }

    const size_t grid_h = target.height / config.patch_size;
    const size_t grid_w = target.width / config.patch_size;
    const size_t patch_dim = 3 * config.temporal_patch_size * config.patch_size * config.patch_size;
    const size_t num_groups = (frames.size() + config.temporal_patch_size - 1) / config.temporal_patch_size;
    ov::Tensor pixel_values(ov::element::f32, {num_groups, grid_h, grid_w, patch_dim});
    float* dst = pixel_values.data<float>();
    // Optimum undoes Qwen's merge-block ordering before invoking the exported vision model.
    for (size_t group = 0; group < num_groups; ++group) {
        for (size_t row = 0; row < grid_h; ++row) {
            for (size_t col = 0; col < grid_w; ++col) {
                for (size_t channel = 0; channel < 3; ++channel) {
                    // Match Torchvision's fused rescaling and normalization.
                    const float mean = config.image_mean[channel] * 255.0f;
                    const float stddev = config.image_std[channel] * 255.0f;
                    for (size_t temporal = 0; temporal < config.temporal_patch_size; ++temporal) {
                        const auto& resized =
                            resized_frames[std::min(group * config.temporal_patch_size + temporal, frames.size() - 1)];
                        for (size_t y = 0; y < config.patch_size; ++y) {
                            const size_t src_offset = ((row * config.patch_size + y) * target.width +
                                                       col * config.patch_size) * 3 + channel;
                            for (size_t x = 0; x < config.patch_size; ++x) {
                                *dst++ = (static_cast<float>(resized.buf[src_offset + x * 3]) - mean) / stddev;
                            }
                        }
                    }
                }
            }
        }
    }
    CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = guard.get();
    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();
    const ov::Tensor output = encoder.get_output_tensor();
    const size_t num_tokens = grid_h * grid_w / (config.merge_size * config.merge_size);
    OPENVINO_ASSERT(output.get_shape().size() == 3 && output.get_shape()[0] == num_groups &&
                        output.get_shape()[1] == num_tokens,
                    "LFM3-VL vision output shape does not match the temporal and spatial patch grid.");
    EncodedVideo result;
    result.frame_num = num_groups;
    result.num_video_tokens = num_groups * num_tokens;
    result.resized_source_size = {grid_h, grid_w};
    result.video_features = ov::Tensor(output.get_element_type(), output.get_shape());
    output.copy_to(result.video_features);
    result.video_features.set_shape({1, result.num_video_tokens, output.get_shape()[2]});
    return result;
}

InputsEmbedderLFM3VL::InputsEmbedderLFM3VL(const VLMConfig& vlm_config,
                                         const std::filesystem::path& model_dir,
                                         const Tokenizer& tokenizer,
                                         const std::string& device,
                                         const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) {}

InputsEmbedderLFM3VL::InputsEmbedderLFM3VL(const VLMConfig& vlm_config,
                                         const ModelsMap& models_map,
                                         const Tokenizer& tokenizer,
                                         const std::filesystem::path& config_dir_path,
                                         const std::string& device,
                                         const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

std::vector<EncodedImage> InputsEmbedderLFM3VL::encode_images(const std::vector<ov::Tensor>& images) {
    const auto single_images = to_single_image_tensors(images);
    std::vector<EncodedImage> encoded_images;
    encoded_images.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        encoded_images.emplace_back(m_vision_encoder->encode(image));
    }
    return encoded_images;
}

std::vector<EncodedVideo> InputsEmbedderLFM3VL::encode_videos(const std::vector<ov::Tensor>& videos,
                                                           const std::vector<VideoMetadata>& videos_metadata) {
    OPENVINO_ASSERT(videos_metadata.empty() || videos_metadata.size() == videos.size(),
                    "Number of videos and video metadata entries must match.");
    std::vector<EncodedVideo> encoded_videos;
    encoded_videos.reserve(videos.size());
    for (size_t i = 0; i < videos.size(); ++i) {
        const auto& video = videos[i];
        const auto& shape = video.get_shape();
        OPENVINO_ASSERT(shape.size() == 4 && shape[0] > 0 && shape[1] > 0 && shape[2] > 0 && shape[3] == 3 &&
                            video.get_element_type() == ov::element::u8,
                        "LFM3-VL videos must be non-empty uint8 RGB tensors with shape [frames, height, width, 3].");
        EncodedVideo sampled;
        sampled.metadata = videos_metadata.empty() ? VideoMetadata{} : videos_metadata[i];
        OPENVINO_ASSERT(std::isfinite(sampled.metadata.fps) && sampled.metadata.fps >= 0.0f,
                        "LFM3-VL video FPS must be finite and non-negative.");
        fill_video_metadata(sampled, shape[0], m_vision_encoder->get_video_processor_config());
        const auto sampled_video = sample_video_if_needed(video, sampled.metadata);
        const auto frames = to_single_image_tensors({sampled_video});
        auto encoded = m_vision_encoder->encode_frames(frames);
        encoded.metadata = std::move(sampled.metadata);
        encoded_videos.emplace_back(std::move(encoded));
    }
    return encoded_videos;
}

NormalizedPrompt InputsEmbedderLFM3VL::normalize_prompt(const std::string& prompt,
                                                      size_t base_id,
                                                      const std::vector<EncodedImage>& images) const {
    const std::string native_tag =
        m_vlm_config.vision_start_token + m_vlm_config.image_token + m_vlm_config.vision_end_token;
    auto [unified_prompt, images_sequence] = normalize(prompt, native_tag, native_tag, base_id, images.size());
    size_t offset = 0;
    for (size_t image_id : images_sequence) {
        const auto& image = images.at(image_id - base_id);
        std::string expanded_tag = m_vlm_config.vision_start_token;
        for (size_t idx = 0; idx < image.num_image_tokens; ++idx) {
            expanded_tag += m_vlm_config.image_token;
        }
        expanded_tag += m_vlm_config.vision_end_token;
        const size_t pos = unified_prompt.find(native_tag, offset);
        OPENVINO_ASSERT(pos != std::string::npos, "LFM3-VL image tag is missing during prompt normalization.");
        unified_prompt.replace(pos, native_tag.size(), expanded_tag);
        offset = pos + expanded_tag.size();
    }
    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

NormalizedPrompt InputsEmbedderLFM3VL::normalize_prompt(const std::string& prompt,
                                                      size_t base_image_id,
                                                      size_t base_video_id,
                                                      const std::vector<EncodedImage>& images,
                                                      const std::vector<EncodedVideo>& videos) const {
    auto normalized = normalize_prompt(prompt, base_image_id, images);
    const std::string native_tag =
        m_vlm_config.vision_start_token + m_vlm_config.video_token + m_vlm_config.vision_end_token;
    auto [unified_prompt, video_sequence] =
        normalize(normalized.unified_prompt, native_tag, native_tag, base_video_id, videos.size(), ModalityType::VIDEO);
    size_t offset = 0;
    for (size_t video_id : video_sequence) {
        const auto& video = videos.at(video_id - base_video_id);
        const auto timestamps =
            calculate_video_timestamps(video.metadata,
                                       m_vision_encoder->get_video_processor_config().temporal_patch_size);
        OPENVINO_ASSERT(video.frame_num > 0 && timestamps.size() == video.frame_num &&
                            video.num_video_tokens % video.frame_num == 0,
                        "LFM3-VL video timestamps and token counts must match the encoded temporal groups.");
        std::ostringstream expanded;
        expanded.imbue(std::locale::classic());
        expanded << std::fixed << std::setprecision(1);
        expanded << m_vlm_config.vision_start_token;
        for (double timestamp : timestamps) {
            expanded << '<' << timestamp << " seconds>" << m_vlm_config.vision_start_token;
            for (size_t token = 0; token < video.num_video_tokens / video.frame_num; ++token) {
                expanded << m_vlm_config.video_token;
            }
            expanded << m_vlm_config.vision_end_token;
        }
        expanded << m_vlm_config.vision_end_token;
        const std::string expanded_tag = expanded.str();
        const size_t pos = unified_prompt.find(native_tag, offset);
        OPENVINO_ASSERT(pos != std::string::npos, "LFM3-VL video tag is missing during prompt normalization.");
        unified_prompt.replace(pos, native_tag.size(), expanded_tag);
        offset = pos + expanded_tag.size();
    }
    return {std::move(unified_prompt), std::move(normalized.images_sequence), std::move(video_sequence)};
}

ov::Tensor InputsEmbedderLFM3VL::get_inputs_embeds(const std::string& prompt,
                                                 const std::vector<EncodedImage>& images,
                                                 VLMPerfMetrics& metrics,
                                                 bool recalculate_merged_embeddings,
                                                 const std::vector<size_t>& image_sequence) {
    return get_inputs_embeds(prompt, images, {}, metrics, recalculate_merged_embeddings, image_sequence, {});
}

ov::Tensor InputsEmbedderLFM3VL::get_inputs_embeds(const std::string& prompt,
                                                 const std::vector<EncodedImage>& images,
                                                 const std::vector<EncodedVideo>& videos,
                                                 VLMPerfMetrics& metrics,
                                                 bool recalculate_merged_embeddings,
                                                 const std::vector<size_t>& image_sequence,
                                                 const std::vector<size_t>& videos_sequence,
                                                 const std::vector<std::pair<size_t, size_t>>& history_vision_count) {
    const ov::Tensor input_ids = get_encoded_input_ids(prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> guard(m_embedding->get_request_queue().get());
    ov::Tensor text_embeds = get_text_embedding(guard.get(), input_ids, metrics);

    std::vector<ov::Tensor> image_embeds;
    for (size_t image_id : image_sequence) {
        image_embeds.push_back(images.at(image_id).resized_source);
    }
    std::vector<ov::Tensor> video_embeds;
    for (size_t video_id : videos_sequence) {
        video_embeds.push_back(videos.at(video_id).video_features);
    }
    const size_t hidden_size = text_embeds.get_shape()[2];
    const int64_t* ids = input_ids.data<const int64_t>();
    auto prepare_features = [&](std::vector<ov::Tensor>& features, int64_t token_id, const char* modality) {
        size_t num_tokens = 0;
        for (auto& feature : features) {
            const auto& shape = feature.get_shape();
            OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1 && shape[2] == hidden_size,
                            "LFM3-VL vision embeddings must have shape [1, tokens, text hidden size].");
            num_tokens += shape[1];
            // Encoded images and videos own the buffers until embedding insertion completes.
            feature = ov::Tensor(feature.get_element_type(), {shape[1], hidden_size}, feature.data());
        }
        OPENVINO_ASSERT(static_cast<size_t>(std::count(ids, ids + input_ids.get_size(), token_id)) == num_tokens,
                        "LFM3-VL ", modality, " token count does not match the number of vision embeddings.");
        return features.empty() ? ov::Tensor(text_embeds.get_element_type(), {0, hidden_size})
                                : qwen2_vl_utils::concatenate_video_image_embeds({}, features);
    };
    const auto flat_images = prepare_features(image_embeds, m_vlm_config.image_token_id, "image");
    const auto flat_videos = prepare_features(video_embeds, m_vlm_config.video_token_id, "video");
    return qwen2_vl_utils::merge_text_and_video_image_embeddings(input_ids, text_embeds, flat_images, flat_videos,
                                                               m_vlm_config.image_token_id, m_vlm_config.video_token_id);
}

}  // namespace ov::genai
