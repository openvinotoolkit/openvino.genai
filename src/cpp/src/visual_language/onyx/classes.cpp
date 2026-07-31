// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/onyx/classes.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <tuple>
#include <vector>

#include "openvino/genai/visual_language/perf_metrics.hpp"
#include "utils.hpp"
#include "visual_language/clip.hpp"

namespace {

constexpr const char* IMAGE_SENTINEL = "<|image|>";
constexpr const char* IMAGE_START = "<|image_start|>";
constexpr const char* PATCH_TOKEN = "<|patch|>";
constexpr const char* IMAGE_END = "<|image_end|>";
constexpr const char* VIDEO_SENTINEL = "<|video|>";
constexpr const char* VIDEO_START = "<|vid_start|>";
constexpr const char* VIDEO_SEPARATOR = "<|vid_frame_separator|>";
constexpr const char* VIDEO_END = "<|vid_end|>";

std::pair<int, int> compute_image_size(const int image_width,
                                       const int image_height,
                                       const size_t patch_hw,
                                       const size_t max_tokens) {
    float image_grid_height = static_cast<float>(image_height) / static_cast<float>(patch_hw);
    float image_grid_width = static_cast<float>(image_width) / static_cast<float>(patch_hw);
    const float ratio = image_grid_height > 0.0f ? image_grid_width / image_grid_height : 1.0f;
    if (image_grid_height * image_grid_width > static_cast<float>(max_tokens)) {
        image_grid_height = std::sqrt(static_cast<float>(max_tokens) / ratio);
        image_grid_width = image_grid_height * ratio;
    }

    const std::array<int, 2> height_candidates{static_cast<int>(std::floor(image_grid_height)),
                                               static_cast<int>(std::ceil(image_grid_height))};
    const std::array<int, 2> width_candidates{static_cast<int>(std::floor(image_grid_width)),
                                              static_cast<int>(std::ceil(image_grid_width))};

    std::vector<std::pair<int, int>> candidates;
    for (const int grid_height : height_candidates) {
        for (const int grid_width : width_candidates) {
            if (grid_height >= 1 && grid_width >= 1 && static_cast<size_t>(grid_height * grid_width) <= max_tokens) {
                candidates.emplace_back(grid_height, grid_width);
            }
        }
    }
    if (candidates.empty()) {
        candidates.emplace_back(std::max(1, static_cast<int>(std::round(image_grid_height))),
                                std::max(1, static_cast<int>(std::round(image_grid_width))));
    }

    const float source_ratio = static_cast<float>(image_height) / static_cast<float>(image_width);
    const auto best = std::min_element(
        candidates.cbegin(),
        candidates.cend(),
        [source_ratio](const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) {
            const float lhs_delta =
                std::abs(static_cast<float>(lhs.first) / static_cast<float>(lhs.second) - source_ratio);
            const float rhs_delta =
                std::abs(static_cast<float>(rhs.first) / static_cast<float>(rhs.second) - source_ratio);
            return lhs_delta < rhs_delta;
        });

    return {best->first * static_cast<int>(patch_hw), best->second * static_cast<int>(patch_hw)};
}

ov::Tensor get_pixel_values(const std::vector<ov::Tensor>& frames,
                            const ov::genai::ProcessorConfig& config,
                            const size_t max_tokens) {
    OPENVINO_ASSERT(!frames.empty(), "Onyx vision input must contain at least one frame");
    OPENVINO_ASSERT(frames.size() == 1 || frames.size() == config.patch_temporal,
                    "Onyx vision input must contain one image or exactly ",
                    config.patch_temporal,
                    " video frames, got ",
                    frames.size());

    clip_image_u8 input_image = tensor_to_clip_image_u8(frames.front());
    const size_t patch_hw = config.patch_size * config.downsample_factor;
    const auto [target_height, target_width] = compute_image_size(input_image.nx, input_image.ny, patch_hw, max_tokens);

    ov::Tensor pixel_values(
        ov::element::f32,
        {config.patch_temporal * 3, static_cast<size_t>(target_height), static_cast<size_t>(target_width)});
    float* output_data = pixel_values.data<float>();
    const size_t plane_size = static_cast<size_t>(target_height) * static_cast<size_t>(target_width);
    const size_t image_size = 3 * plane_size;

    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);
    for (size_t frame_idx = 0; frame_idx < config.patch_temporal; ++frame_idx) {
        // onyx encoder expects [patch_temporal * 3, H, W] pixel_values as input,
        // so we need to duplicate in case of single frame
        const ov::Tensor& frame = frames.size() == 1 ? frames.front() : frames.at(frame_idx);
        clip_image_u8 resized_image;
        lanczos_resize(tensor_to_clip_image_u8(frame), resized_image, target_width, target_height);
        const clip_image_f32 normalized_image = clip_image_preprocess(ctx, resized_image);
        std::copy_n(normalized_image.buf.data(), image_size, output_data + frame_idx * image_size);
    }
    return pixel_values;
}

void fill_video_metadata(ov::genai::VideoMetadata& metadata,
                         const size_t total_num_frames,
                         const ov::genai::VideoProcessorConfig& config) {
    OPENVINO_ASSERT(total_num_frames >= config.patch_temporal,
                    "Onyx video must contain at least ",
                    config.patch_temporal,
                    " frames, got ",
                    total_num_frames);
    OPENVINO_ASSERT(metadata.fps > 0.0f, "Onyx video metadata fps must be positive");

    if (!metadata.frames_indices.empty()) {
        OPENVINO_ASSERT(metadata.frames_indices.size() % config.patch_temporal == 0,
                        "Onyx sampled frame count must be a multiple of patch_temporal=",
                        config.patch_temporal,
                        ", got ",
                        metadata.frames_indices.size());
        return;
    }

    OPENVINO_ASSERT(config.fps > 0.0f, "Onyx video_sampling_fps must be positive");
    OPENVINO_ASSERT(config.max_frames > 0, "Onyx video_num_frames must be positive");
    // number of frames to sample, based on the original video fps and the target sampling fps
    size_t num_frames =
        static_cast<size_t>(static_cast<double>(total_num_frames) * static_cast<double>(config.fps) / metadata.fps);
    // should be no more than model config video_num_frames, and no more than the original video frame count
    num_frames = std::min({num_frames, config.max_frames, total_num_frames});
    // round it down to a multiple of config.patch_temporal
    num_frames -= num_frames % config.patch_temporal;
    // ensure that the number of frames is at least config.patch_temporal
    num_frames = std::max(num_frames, config.patch_temporal);
    OPENVINO_ASSERT(num_frames % config.patch_temporal == 0,
                    "Onyx sampled frame count must be a multiple of patch_temporal=",
                    config.patch_temporal,
                    ", got ",
                    num_frames);

    metadata.frames_indices.reserve(num_frames);
    for (size_t idx = 0; idx < num_frames; ++idx) {
        const double position =
            static_cast<double>(idx) * static_cast<double>(total_num_frames - 1) / static_cast<double>(num_frames - 1);
        metadata.frames_indices.push_back(static_cast<size_t>(position));
    }
}

}  // namespace

namespace ov::genai {

EncodedImage VisionEncoderOnyx::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    const ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);
    return encode_with_config({image}, config, config.max_image_tokens);
}

EncodedImage VisionEncoderOnyx::encode_with_config(const std::vector<ov::Tensor>& frames,
                                                   const ProcessorConfig& config,
                                                   const size_t max_tokens) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();

    ov::Tensor pixel_values = get_pixel_values(frames, config, max_tokens);

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    const ov::Shape& infer_output_shape = infer_output.get_shape();
    OPENVINO_ASSERT(infer_output_shape.size() == 2,
                    "Onyx vision embeddings output must have rank 2 [num_patches, hidden_size], got ",
                    infer_output_shape);
    const size_t num_image_tokens = infer_output_shape.at(0);
    const size_t hidden_size = infer_output_shape.at(1);

    ov::Tensor image_features(infer_output.get_element_type(), {1, num_image_tokens, hidden_size});
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    EncodedImage encoded_image{std::move(image_features)};
    encoded_image.num_image_tokens = num_image_tokens;
    return encoded_image;
}

EncodedVideo VisionEncoderOnyx::encode_frames(const std::vector<ov::Tensor>& frames) {
    const size_t patch_temporal = m_video_processor_config.patch_temporal;
    OPENVINO_ASSERT(!frames.empty() && frames.size() % patch_temporal == 0,
                    "Onyx video frame count must be a positive multiple of patch_temporal=",
                    patch_temporal,
                    ", got ",
                    frames.size());

    std::vector<EncodedImage> encoded_groups;
    encoded_groups.reserve(frames.size() / patch_temporal);
    for (size_t begin = 0; begin < frames.size(); begin += patch_temporal) {
        const std::vector<ov::Tensor> group(frames.cbegin() + begin, frames.cbegin() + begin + patch_temporal);
        encoded_groups.emplace_back(
            encode_with_config(group, m_video_processor_config, m_video_processor_config.max_video_frame_tokens));
    }

    const ov::Shape& group_shape = encoded_groups.front().resized_source.get_shape();
    const size_t tokens_per_group = group_shape.at(1);
    const size_t hidden_size = group_shape.at(2);
    const size_t total_tokens = encoded_groups.size() * tokens_per_group;
    ov::Tensor video_features(encoded_groups.front().resized_source.get_element_type(), {1, total_tokens, hidden_size});
    uint8_t* destination = static_cast<uint8_t*>(video_features.data());
    for (const EncodedImage& group : encoded_groups) {
        OPENVINO_ASSERT(group.resized_source.get_shape() == group_shape,
                        "Onyx video temporal groups must produce equal embedding shapes");
        std::memcpy(destination, group.resized_source.data(), group.resized_source.get_byte_size());
        destination += group.resized_source.get_byte_size();
    }

    EncodedVideo encoded_video;
    encoded_video.video_features = std::move(video_features);
    encoded_video.num_video_tokens = total_tokens;
    encoded_video.frame_num = encoded_groups.size();
    return encoded_video;
}

InputsEmbedderOnyx::InputsEmbedderOnyx(const VLMConfig& vlm_config,
                                       const std::filesystem::path& model_dir,
                                       const Tokenizer& tokenizer,
                                       const std::string& device,
                                       const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) {}

InputsEmbedderOnyx::InputsEmbedderOnyx(const VLMConfig& vlm_config,
                                       const ModelsMap& models_map,
                                       const Tokenizer& tokenizer,
                                       const std::filesystem::path& config_dir_path,
                                       const std::string& device,
                                       const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

std::vector<EncodedImage> InputsEmbedderOnyx::encode_images(const std::vector<ov::Tensor>& images) {
    const ov::AnyMap patch_temporal_config = {{"patch_temporal", m_vlm_config.vision_patch_temporal}};

    std::vector<EncodedImage> embeds;
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image, patch_temporal_config));
    }
    return embeds;
}

std::vector<EncodedVideo> InputsEmbedderOnyx::encode_videos(const std::vector<ov::Tensor>& videos,
                                                            const std::vector<VideoMetadata>& videos_metadata) {
    OPENVINO_ASSERT(videos.size() == videos_metadata.size() || videos_metadata.empty(),
                    "Number of videos and video metadata entries must match if metadata is provided");

    std::vector<EncodedVideo> encoded_videos;
    encoded_videos.reserve(videos.size());
    for (size_t video_idx = 0; video_idx < videos.size(); ++video_idx) {
        const ov::Tensor& video = videos.at(video_idx);
        OPENVINO_ASSERT(video.get_shape().size() == 4, "Onyx video tensor must have rank 4 [N, H, W, C]");
        VideoMetadata metadata = video_idx < videos_metadata.size() ? videos_metadata.at(video_idx) : VideoMetadata{};
        fill_video_metadata(metadata, video.get_shape().at(0), m_vision_encoder->get_video_processor_config());
        const ov::Tensor sampled_video = sample_video_if_needed(video, metadata);
        EncodedVideo encoded_video = m_vision_encoder->encode_frames(to_single_image_tensors({sampled_video}));
        encoded_video.metadata = std::move(metadata);
        encoded_videos.emplace_back(std::move(encoded_video));
    }
    return encoded_videos;
}

NormalizedPrompt InputsEmbedderOnyx::normalize_prompt(const std::string& prompt,
                                                      size_t base_id,
                                                      const std::vector<EncodedImage>& images) const {
    return normalize_prompt(prompt, base_id, 0, images, {});
}

NormalizedPrompt InputsEmbedderOnyx::normalize_prompt(const std::string& prompt,
                                                      const size_t base_image_id,
                                                      const size_t base_video_id,
                                                      const std::vector<EncodedImage>& images,
                                                      const std::vector<EncodedVideo>& videos) const {
    auto [unified_prompt, images_sequence] =
        normalize(prompt, IMAGE_SENTINEL, IMAGE_SENTINEL, base_image_id, images.size());

    size_t searched_pos = 0;
    for (const size_t new_image_id : images_sequence) {
        const EncodedImage& image = images.at(new_image_id - base_image_id);
        const ov::Shape& image_features_shape = image.resized_source.get_shape();
        OPENVINO_ASSERT(image_features_shape.size() == 3,
                        "Onyx image features must have rank 3 [1, num_patches, hidden_size], got ",
                        image_features_shape);
        const size_t num_image_tokens = image_features_shape.at(1);

        std::string expanded_tag = IMAGE_START;
        for (size_t idx = 0; idx < num_image_tokens; ++idx) {
            expanded_tag += PATCH_TOKEN;
        }
        expanded_tag += IMAGE_END;

        searched_pos = unified_prompt.find(IMAGE_SENTINEL, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos, "Onyx image sentinel is missing from normalized prompt");
        unified_prompt.replace(searched_pos, std::char_traits<char>::length(IMAGE_SENTINEL), expanded_tag);
        searched_pos += expanded_tag.length();
    }

    std::vector<size_t> videos_sequence;
    std::tie(unified_prompt, videos_sequence) =
        normalize(unified_prompt, VIDEO_SENTINEL, VIDEO_SENTINEL, base_video_id, videos.size(), VisionType::VIDEO);
    searched_pos = 0;
    const size_t patch_temporal = m_vision_encoder->get_video_processor_config().patch_temporal;
    for (const size_t new_video_id : videos_sequence) {
        const EncodedVideo& video = videos.at(new_video_id - base_video_id);
        OPENVINO_ASSERT(video.frame_num > 0 && video.num_video_tokens % video.frame_num == 0,
                        "Onyx video embeddings must contain an equal positive token count per temporal group");
        OPENVINO_ASSERT(video.metadata.frames_indices.size() == video.frame_num * patch_temporal,
                        "Onyx video metadata frame indices do not match the encoded temporal groups");
        const size_t tokens_per_group = video.num_video_tokens / video.frame_num;

        std::string expanded_tag = VIDEO_START;
        for (size_t group_idx = 0; group_idx < video.frame_num; ++group_idx) {
            const float timestamp =
                static_cast<float>(video.metadata.frames_indices.at(group_idx * patch_temporal)) / video.metadata.fps;
            std::ostringstream timestamp_stream;
            timestamp_stream << std::fixed << std::setprecision(1) << timestamp;
            expanded_tag += "Time: " + timestamp_stream.str() + "s";
            for (size_t token_idx = 0; token_idx < tokens_per_group; ++token_idx) {
                expanded_tag += VIDEO_SENTINEL;
            }
            expanded_tag += group_idx + 1 < video.frame_num ? VIDEO_SEPARATOR : VIDEO_END;
        }

        searched_pos = unified_prompt.find(VIDEO_SENTINEL, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos, "Onyx video sentinel is missing from normalized prompt");
        unified_prompt.replace(searched_pos, std::char_traits<char>::length(VIDEO_SENTINEL), expanded_tag);
        searched_pos += expanded_tag.length();
    }
    return {std::move(unified_prompt), std::move(images_sequence), std::move(videos_sequence)};
}

ov::Tensor InputsEmbedderOnyx::get_inputs_embeds(const std::string& unified_prompt,
                                                 const std::vector<EncodedImage>& images,
                                                 VLMPerfMetrics& metrics,
                                                 bool recalculate_merged_embeddings,
                                                 const std::vector<size_t>& images_sequence) {
    return get_inputs_embeds(unified_prompt, images, {}, metrics, recalculate_merged_embeddings, images_sequence, {});
}

ov::Tensor InputsEmbedderOnyx::get_inputs_embeds(
    const std::string& unified_prompt,
    const std::vector<EncodedImage>& images,
    const std::vector<EncodedVideo>& videos,
    VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& images_sequence,
    const std::vector<size_t>& videos_sequence,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (const size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id).resized_source);
    }

    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    if (image_embeds.empty() && videos_sequence.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    // Keep initialization lazy so pipeline construction can overlap with the tokenizer's asynchronous warmup.
    encode_vision_token_ids();

    ov::Tensor inputs_embeds =
        image_embeds.empty()
            ? std::move(text_embeds)
            : utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, m_image_token_id);

    std::vector<ov::Tensor> video_group_embeds;
    for (const size_t video_id : videos_sequence) {
        const EncodedVideo& video = videos.at(video_id);
        const ov::Shape& shape = video.video_features.get_shape();
        const size_t tokens_per_group = video.num_video_tokens / video.frame_num;
        const size_t bytes_per_group = tokens_per_group * shape.at(2) * video.video_features.get_element_type().size();
        const uint8_t* source = static_cast<const uint8_t*>(video.video_features.data());
        for (size_t group_idx = 0; group_idx < video.frame_num; ++group_idx) {
            video_group_embeds.emplace_back(video.video_features.get_element_type(),
                                            ov::Shape{1, tokens_per_group, shape.at(2)},
                                            source + group_idx * bytes_per_group);
        }
    }
    if (!video_group_embeds.empty()) {
        inputs_embeds = utils::merge_text_and_image_embeddings_llava(input_ids,
                                                                     inputs_embeds,
                                                                     video_group_embeds,
                                                                     m_video_token_id);
    }

    return inputs_embeds;
}

void InputsEmbedderOnyx::encode_vision_token_ids() {
    std::call_once(m_vision_token_ids_once_flag, [this]() {
        const ov::Tensor encoded_vision_tokens =
            m_tokenizer.encode(std::string(PATCH_TOKEN) + VIDEO_SENTINEL, ov::genai::add_special_tokens(false))
                .input_ids;
        OPENVINO_ASSERT(encoded_vision_tokens.get_size() == 2,
                        "Onyx patch and video markers must encode to two tokens");
        m_image_token_id = encoded_vision_tokens.data<int64_t>()[0];
        m_video_token_id = encoded_vision_tokens.data<int64_t>()[1];
    });
}

}  // namespace ov::genai
