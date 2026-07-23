// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/gemma4/classes.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <numeric>
#include <regex>
#include <sstream>

#include "logger.hpp"
#include "utils.hpp"
#include "visual_language/clip.hpp"

namespace {

constexpr float DEFAULT_METADATA_FPS = 24.0f;

/// @brief Compute target dimensions for aspect-ratio-preserving resize.
/// Total pixel count should match max_patches * patch_size^2
/// Dimensions are divisible by pooling_kernel_size * patch_size.
/// Matches HF Gemma4ImageProcessor.get_aspect_ratio_preserving_size().
std::pair<size_t, size_t> get_aspect_ratio_preserving_size(size_t height,
                                                           size_t width,
                                                           size_t patch_size,
                                                           size_t max_patches,
                                                           size_t pooling_kernel_size) {
    const double total_px = static_cast<double>(height * width);
    const double target_px = static_cast<double>(max_patches) * static_cast<double>(patch_size * patch_size);
    const double factor = std::sqrt(target_px / total_px);
    const double ideal_height = factor * static_cast<double>(height);
    const double ideal_width = factor * static_cast<double>(width);
    const size_t side_mult = pooling_kernel_size * patch_size;

    size_t target_height = static_cast<size_t>(std::floor(ideal_height / static_cast<double>(side_mult))) * side_mult;
    size_t target_width = static_cast<size_t>(std::floor(ideal_width / static_cast<double>(side_mult))) * side_mult;

    OPENVINO_ASSERT(target_height != 0 || target_width != 0,
                    "Cannot resize image to 0x0. Dimensions must be divisible by ",
                    side_mult);

    const size_t max_side_length = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;
    if (target_height == 0) {
        target_height = side_mult;
        target_width = std::min(
            static_cast<size_t>(std::floor(static_cast<double>(width) / static_cast<double>(height))) * side_mult,
            max_side_length);
    } else if (target_width == 0) {
        target_width = side_mult;
        target_height = std::min(
            static_cast<size_t>(std::floor(static_cast<double>(height) / static_cast<double>(width))) * side_mult,
            max_side_length);
    }

    return {target_height, target_width};
}

/// @brief Extract patches from a CHW float image into a flat [num_patches, patch_dim] tensor.
/// Each patch stores pixels in HWC order within the patch, matching HF convert_image_to_patches().
/// @param float_image CHW float image (clip_image_f32)
/// @param patch_size Size of each square patch
/// @param output Pointer to output buffer, must have space for num_patches * patch_dim floats
/// @param num_patches_h Number of patches along height
/// @param num_patches_w Number of patches along width
void extract_patches(const clip_image_f32& float_image,
                     size_t patch_size,
                     float* output,
                     size_t num_patches_h,
                     size_t num_patches_w) {
    const size_t patch_dim = patch_size * patch_size * 3;
    const size_t nx = static_cast<size_t>(float_image.nx);
    const size_t ny = static_cast<size_t>(float_image.ny);

    for (size_t py = 0; py < num_patches_h; py++) {
        for (size_t px = 0; px < num_patches_w; px++) {
            const size_t patch_idx = py * num_patches_w + px;
            float* dst = output + patch_idx * patch_dim;
            for (size_t y = 0; y < patch_size; y++) {
                for (size_t x = 0; x < patch_size; x++) {
                    const size_t src_y = py * patch_size + y;
                    const size_t src_x = px * patch_size + x;
                    for (size_t c = 0; c < 3; c++) {
                        // CHW source: buf[c * ny * nx + src_y * nx + src_x]
                        // HWC within patch: dst[y * patch_size * 3 + x * 3 + c]
                        dst[y * patch_size * 3 + x * 3 + c] = float_image.buf[c * ny * nx + src_y * nx + src_x];
                    }
                }
            }
        }
    }
}

size_t get_static_pixel_values_patch_dim(ov::InferRequest& encoder) {
    ov::PartialShape pixel_values_shape = encoder.get_compiled_model().input("pixel_values").get_partial_shape();
    OPENVINO_ASSERT(pixel_values_shape.rank().is_static() && pixel_values_shape.rank().get_length() == 3,
                    "Gemma4 vision embeddings pixel_values input must be rank 3, got ",
                    pixel_values_shape);
    OPENVINO_ASSERT(pixel_values_shape[2].is_static(),
                    "Gemma4 vision embeddings pixel_values patch dimension must be static, got ",
                    pixel_values_shape);
    return static_cast<size_t>(pixel_values_shape[2].get_length());
}

enum class PatchExtractionMode {
    UnmergedPatches,
    MergedPatches,
};

struct PatchExtractionConfig {
    PatchExtractionMode mode;
    size_t patch_size;
    size_t max_patches;
    size_t patch_dim;
};

PatchExtractionConfig get_patch_extraction_config(const ov::genai::ProcessorConfig& config, size_t patch_dim) {
    const size_t model_patch_size = config.patch_size * config.pooling_kernel_size;
    const size_t patch_dim_for_patch_size = config.patch_size * config.patch_size * 3;
    const size_t patch_dim_for_model_patch_size = model_patch_size * model_patch_size * 3;
    OPENVINO_ASSERT(patch_dim == patch_dim_for_patch_size || patch_dim == patch_dim_for_model_patch_size,
                    "Gemma4 vision embeddings pixel_values patch dimension ",
                    patch_dim,
                    " does not match patch_size dimension ",
                    patch_dim_for_patch_size,
                    " or merged patch dimension ",
                    patch_dim_for_model_patch_size);

    if (patch_dim == patch_dim_for_patch_size) {
        const size_t max_unmerged_patches =
            config.max_soft_tokens * config.pooling_kernel_size * config.pooling_kernel_size;
        return {PatchExtractionMode::UnmergedPatches, config.patch_size, max_unmerged_patches, patch_dim};
    }
    return {PatchExtractionMode::MergedPatches, model_patch_size, config.max_soft_tokens, patch_dim};
}

size_t get_num_valid_soft_tokens(const PatchExtractionConfig& patch_config,
                                 size_t output_tokens,
                                 size_t num_patches_h,
                                 size_t num_patches_w) {
    if (patch_config.mode == PatchExtractionMode::UnmergedPatches) {
        return output_tokens;
    }
    return num_patches_h * num_patches_w;
}

/**
 * @brief Populates video metadata and computes frame sampling indices.
 */
void fill_video_metadata(ov::genai::VideoMetadata& video_metadata,
                         size_t total_num_frames,
                         const ov::genai::VideoProcessorConfig& video_config) {
    if (video_metadata.fps == 0.0f) {
        GENAI_WARN("Gemma4 requires frame timestamps to construct prompts, but fps is not set. "
                   "Defaulting to 24 fps. Please provide VideoMetadata with fps for more accurate results.");
        video_metadata.fps = DEFAULT_METADATA_FPS;
    }

    if (!video_metadata.frames_indices.empty()) {
        GENAI_WARN("Frames indices already provided in video metadata, skipping Gemma4 model-specific sampling.");
        return;
    }

    if (!video_config.do_sample_frames) {
        video_metadata.frames_indices.resize(total_num_frames);
        std::iota(video_metadata.frames_indices.begin(), video_metadata.frames_indices.end(), 0);
        return;
    }

    size_t num_frames = video_config.num_frames;

    if (num_frames == 0) {
        num_frames =
            std::min(total_num_frames, video_config.max_frames > 0 ? video_config.max_frames : total_num_frames);
    }

    num_frames = std::min(num_frames, total_num_frames);
    OPENVINO_ASSERT(num_frames > 0, "Number of frames to sample must be positive.");

    if (num_frames == total_num_frames) {
        video_metadata.frames_indices.resize(total_num_frames);
        std::iota(video_metadata.frames_indices.begin(), video_metadata.frames_indices.end(), 0);
        return;
    }

    video_metadata.frames_indices.reserve(num_frames);
    const double step = static_cast<double>(total_num_frames) / static_cast<double>(num_frames);
    for (size_t i = 0; i < num_frames; ++i) {
        video_metadata.frames_indices.push_back(static_cast<size_t>(static_cast<double>(i) * step));
    }
}

}  // namespace

namespace ov::genai {

EncodedImage VisionEncoderGemma4::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);
    return encode_with_config(image, config);
}

EncodedImage VisionEncoderGemma4::encode_with_config(const ov::Tensor& image, const ProcessorConfig& config) {
    // 1. Convert input tensor (NHWC uint8) to clip_image_u8 (HWC uint8)
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);

    // 2. Compute aspect-ratio-preserving target size
    const size_t max_unmerged_patches =
        config.max_soft_tokens * config.pooling_kernel_size * config.pooling_kernel_size;
    const auto [target_height, target_width] = get_aspect_ratio_preserving_size(static_cast<size_t>(input_image.ny),
                                                                                static_cast<size_t>(input_image.nx),
                                                                                config.patch_size,
                                                                                max_unmerged_patches,
                                                                                config.pooling_kernel_size);

    // 3. Bicubic resize
    clip_image_u8 resized_image;
    bicubic_resize(input_image, resized_image, static_cast<int>(target_width), static_cast<int>(target_height));

    // 4. Rescale to [0,1] and convert to CHW float
    // With mean=[0,0,0] and std=[1,1,1], clip_image_preprocess produces pixel/255.0
    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);
    clip_image_f32 float_image = clip_image_preprocess(ctx, resized_image);

    // 5. Extract patches in the format expected by the exported vision embeddings model.
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    const size_t patch_dim = get_static_pixel_values_patch_dim(encoder);
    const PatchExtractionConfig patch_config = get_patch_extraction_config(config, patch_dim);
    const size_t num_patches_h = target_height / patch_config.patch_size;
    const size_t num_patches_w = target_width / patch_config.patch_size;

    ov::Tensor pixel_values(ov::element::f32, {1, patch_config.max_patches, patch_config.patch_dim});
    float* pv_data = pixel_values.data<float>();
    std::fill(pv_data, pv_data + patch_config.max_patches * patch_config.patch_dim, 0.0f);

    extract_patches(float_image, patch_config.patch_size, pv_data, num_patches_h, num_patches_w);

    // 6. Compute 2D patch position IDs and pad unused positions with -1
    ov::Tensor image_position_ids(ov::element::i64, {1, patch_config.max_patches, 2});
    int64_t* pos_data = image_position_ids.data<int64_t>();
    std::fill(pos_data, pos_data + patch_config.max_patches * 2, int64_t{-1});

    for (size_t py = 0; py < num_patches_h; py++) {
        for (size_t px = 0; px < num_patches_w; px++) {
            const size_t patch_idx = py * num_patches_w + px;
            pos_data[patch_idx * 2 + 0] = static_cast<int64_t>(px);  // x coordinate
            pos_data[patch_idx * 2 + 1] = static_cast<int64_t>(py);  // y coordinate
        }
    }

    // 7. Run vision encoder
    encoder.set_tensor("pixel_values", pixel_values);
    encoder.set_tensor("image_position_ids", image_position_ids);
    encoder.infer();

    // 8. Output shape is [num_soft_tokens, hidden_size] or [1, max_soft_tokens, hidden_size].
    const ov::Tensor& infer_output = encoder.get_output_tensor();
    const ov::Shape& output_shape = infer_output.get_shape();
    OPENVINO_ASSERT(output_shape.size() == 3 || output_shape.size() == 2,
                    "Gemma4 vision embeddings output rank must be 2 or 3, got ",
                    output_shape.size());

    const size_t output_tokens = output_shape.size() == 3 ? output_shape[1] : output_shape[0];
    const size_t hidden_size = output_shape.size() == 3 ? output_shape[2] : output_shape[1];
    const size_t num_valid_soft_tokens =
        get_num_valid_soft_tokens(patch_config, output_tokens, num_patches_h, num_patches_w);
    OPENVINO_ASSERT(num_valid_soft_tokens <= output_tokens,
                    "Gemma4 valid soft token count ",
                    num_valid_soft_tokens,
                    " exceeds vision output token count ",
                    output_tokens);

    ov::Tensor image_features(infer_output.get_element_type(), {1, num_valid_soft_tokens, hidden_size});
    std::memcpy(image_features.data(), infer_output.data(), image_features.get_byte_size());

    return {std::move(image_features)};
}

EncodedVideo VisionEncoderGemma4::encode_frames(const std::vector<ov::Tensor>& frames) {
    OPENVINO_ASSERT(!frames.empty(), "Cannot encode an empty list of video frames.");

    std::vector<ov::Tensor> frame_features;
    frame_features.reserve(frames.size());

    for (const auto& frame : frames) {
        EncodedImage encoded = encode_with_config(frame, m_video_processor_config);
        frame_features.push_back(std::move(encoded.resized_source));
    }

    // Concatenate all frame features: [1, total_tokens, hidden_size]
    size_t num_soft_tokens_per_frame = frame_features[0].get_shape()[1];
    const size_t hidden_size = frame_features[0].get_shape()[2];
    const size_t total_tokens = frames.size() * num_soft_tokens_per_frame;

    ov::Tensor video_features(frame_features[0].get_element_type(), {1, total_tokens, hidden_size});
    uint8_t* dst = static_cast<uint8_t*>(video_features.data());
    for (const auto& frame_feature : frame_features) {
        std::memcpy(dst, frame_feature.data(), frame_feature.get_byte_size());
        dst += frame_feature.get_byte_size();
    }

    EncodedVideo result;
    result.video_features = std::move(video_features);
    result.num_video_tokens = total_tokens;
    result.frame_num = frames.size();
    return result;
}

InputsEmbedderGemma4::InputsEmbedderGemma4(const VLMConfig& vlm_config,
                                           const std::filesystem::path& model_dir,
                                           const Tokenizer& tokenizer,
                                           const std::string& device,
                                           const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) {
    patch_chat_template();

    // per-layer embeddings model is optional, large MOE models don't have it
    if (!has_per_layer_embeddings()) {
        return;
    }

    auto per_layer_model_path = model_dir / "openvino_text_embeddings_per_layer_model.xml";
    auto compiled = utils::singleton_core().compile_model(
        per_layer_model_path,
        device,
        utils::get_model_properties(device_config, "text_embeddings_per_layer", device));
    ov::genai::utils::print_compiled_model_properties(compiled, "VLM per-layer text embeddings model");
    m_per_layer_embeddings_requests = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled.get_property(ov::optimal_number_of_infer_requests),
        [&compiled]() -> ov::InferRequest {
            return compiled.create_infer_request();
        });
}

InputsEmbedderGemma4::InputsEmbedderGemma4(const VLMConfig& vlm_config,
                                           const ModelsMap& models_map,
                                           const Tokenizer& tokenizer,
                                           const std::filesystem::path& config_dir_path,
                                           const std::string& device,
                                           const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    patch_chat_template();

    // per-layer embeddings model is optional, large MOE models don't have it
    if (!has_per_layer_embeddings()) {
        return;
    }

    auto it = models_map.find("text_embeddings_per_layer");

    OPENVINO_ASSERT(it != models_map.end(), "Per-layer text embeddings model not found in models map");
    const auto& [model_str, weights] = it->second;
    auto compiled = utils::singleton_core().compile_model(
        model_str,
        weights,
        device,
        utils::get_model_properties(device_config, "text_embeddings_per_layer", device));
    ov::genai::utils::print_compiled_model_properties(compiled, "VLM per-layer text embeddings model");
    m_per_layer_embeddings_requests = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled.get_property(ov::optimal_number_of_infer_requests),
        [&compiled]() -> ov::InferRequest {
            return compiled.create_infer_request();
        });
}

std::vector<ov::genai::EncodedImage> InputsEmbedderGemma4::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;

    ov::AnyMap vision_config = {{"patch_size", m_vlm_config.vision_config_patch_size}};

    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image, vision_config));
    }

    return embeds;
}

std::vector<ov::genai::EncodedVideo> InputsEmbedderGemma4::encode_videos(
    const std::vector<ov::Tensor>& videos,
    const std::vector<VideoMetadata>& videos_metadata) {
    OPENVINO_ASSERT(videos.size() == videos_metadata.size() || videos_metadata.empty(),
                    "Number of videos and videos metadata must match if metadata provided.");

    std::vector<EncodedVideo> encoded_videos;
    encoded_videos.reserve(videos.size());

    for (size_t i = 0; i < videos.size(); ++i) {
        const ov::Tensor& video = videos[i];
        const size_t video_num_frames = video.get_shape()[0];
        VideoMetadata video_metadata = i < videos_metadata.size() ? videos_metadata[i] : VideoMetadata{};
        fill_video_metadata(video_metadata, video_num_frames, m_vision_encoder->get_video_processor_config());
        const auto sampled_video = sample_video_if_needed(video, video_metadata);
        std::vector<ov::Tensor> frames = to_single_image_tensors({sampled_video});
        auto encoded_video = m_vision_encoder->encode_frames(frames);
        encoded_video.metadata = video_metadata;
        encoded_videos.emplace_back(std::move(encoded_video));
    }
    return encoded_videos;
}

NormalizedPrompt InputsEmbedderGemma4::normalize_prompt(const std::string& prompt,
                                                        size_t base_id,
                                                        const std::vector<EncodedImage>& images) const {
    return normalize_prompt(prompt, base_id, 0, images, {});
}

NormalizedPrompt InputsEmbedderGemma4::normalize_prompt(const std::string& prompt,
                                                        size_t base_image_id,
                                                        size_t base_video_id,
                                                        const std::vector<EncodedImage>& images,
                                                        const std::vector<EncodedVideo>& videos) const {
    const auto& boi = m_vlm_config.boi_token;
    const auto& eoi = m_vlm_config.eoi_token;
    const auto& image_token = m_vlm_config.image_token;
    const auto& video_token = m_vlm_config.video_token;

    // Images
    auto [unified_prompt, images_sequence] =
        normalize(prompt, image_token, image_token, base_image_id, images.size(), VisionType::IMAGE);

    size_t search_offset = 0;
    for (size_t new_image_id : images_sequence) {
        const size_t num_image_tokens = images.at(new_image_id - base_image_id).resized_source.get_shape().at(1);
        std::string expanded_tag;
        expanded_tag.reserve(boi.size() + num_image_tokens * image_token.size() + eoi.size());
        expanded_tag = boi;
        for (size_t i = 0; i < num_image_tokens; i++) {
            expanded_tag += image_token;
        }
        expanded_tag += eoi;

        size_t pos = unified_prompt.find(image_token, search_offset);
        OPENVINO_ASSERT(pos != std::string::npos, "Failed to find image token in prompt during normalization");
        unified_prompt.replace(pos, image_token.length(), expanded_tag);
        search_offset = pos + expanded_tag.size();
    }

    // Videos
    std::vector<size_t> videos_sequence;
    std::tie(unified_prompt, videos_sequence) =
        normalize(unified_prompt, video_token, video_token, base_video_id, videos.size(), VisionType::VIDEO);

    expand_video_tags_in_prompt(unified_prompt, videos, videos_sequence, base_video_id);

    return {std::move(unified_prompt), std::move(images_sequence), std::move(videos_sequence)};
}

void InputsEmbedderGemma4::expand_video_tags_in_prompt(std::string& unified_prompt,
                                                       const std::vector<EncodedVideo>& encoded_videos,
                                                       const std::vector<size_t>& videos_sequence,
                                                       size_t video_base_id) const {
    const auto& boi = m_vlm_config.boi_token;
    const auto& eoi = m_vlm_config.eoi_token;
    const auto& video_token = m_vlm_config.video_token;

    size_t search_offset = 0;
    for (size_t video_id : videos_sequence) {
        const auto& encoded_video = encoded_videos.at(video_id - video_base_id);
        OPENVINO_ASSERT(encoded_video.frame_num > 0, "Video must contain at least one frame.");
        OPENVINO_ASSERT(encoded_video.metadata.frames_indices.size() >= encoded_video.frame_num,
                        "Video metadata frames_indices size (",
                        encoded_video.metadata.frames_indices.size(),
                        ") must be >= frame_num (",
                        encoded_video.frame_num,
                        ")");
        OPENVINO_ASSERT(encoded_video.num_video_tokens % encoded_video.frame_num == 0,
                        "num_video_tokens (",
                        encoded_video.num_video_tokens,
                        ") must be divisible by frame_num (",
                        encoded_video.frame_num,
                        ")");
        OPENVINO_ASSERT(encoded_video.metadata.fps > 0.0f,
                        "Video metadata fps must be positive for timestamp calculation");

        const size_t tokens_per_frame = encoded_video.num_video_tokens / encoded_video.frame_num;

        // Build expanded tag: "MM:SS <boi><video_token>*N<eoi> MM:SS <boi><video_token>×N<eoi> ..."
        std::string expanded;
        // MM:SS (5) + whitespace (1) + <boi> + <video_token>*N + <eoi> + whitespace (1)
        const size_t per_frame_expanded_size =
            5 + 1 + boi.size() + video_token.size() * tokens_per_frame + eoi.size() + 1;
        expanded.reserve(encoded_video.frame_num * per_frame_expanded_size);
        for (size_t i = 0; i < encoded_video.frame_num; ++i) {
            const float seconds =
                static_cast<float>(encoded_video.metadata.frames_indices[i]) / encoded_video.metadata.fps;
            const int mins = static_cast<int>(seconds) / 60;
            const int secs = static_cast<int>(seconds) % 60;

            std::ostringstream timestamp_ss;
            timestamp_ss << std::setfill('0') << std::setw(2) << mins << ":" << std::setfill('0') << std::setw(2)
                         << secs;

            expanded += timestamp_ss.str();
            expanded += " ";
            expanded += boi;
            for (size_t t = 0; t < tokens_per_frame; ++t) {
                expanded += video_token;
            }
            expanded += eoi;
            if (i + 1 < encoded_video.frame_num) {
                expanded += " ";
            }
        }

        const size_t pos = unified_prompt.find(video_token, search_offset);
        OPENVINO_ASSERT(pos != std::string::npos, "Failed to find video token in prompt during expansion");
        unified_prompt.replace(pos, video_token.length(), expanded);
        search_offset = pos + expanded.size();
    }
}

ov::Tensor InputsEmbedderGemma4::get_per_layer_embeddings(const ov::Tensor& input_ids) {
    OPENVINO_ASSERT(m_per_layer_embeddings_requests, "Per-layer embeddings model is not loaded");

    CircularBufferQueueElementGuard<ov::InferRequest> guard(m_per_layer_embeddings_requests.get());
    ov::InferRequest& req = guard.get();
    req.set_tensor("input_ids", input_ids);
    req.infer();

    const ov::Tensor& output = req.get_output_tensor();
    ov::Tensor result(output.get_element_type(), output.get_shape());
    output.copy_to(result);
    return result;
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderGemma4::compute_inputs_embeds(
    const std::string& prompt,
    const std::vector<EncodedImage>& images,
    const std::vector<EncodedVideo>& videos,
    VLMPerfMetrics& metrics,
    const std::vector<size_t>& images_sequence,
    const std::vector<size_t>& videos_sequence) {
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id).resized_source);
    }

    // Collect per-frame video embeddings as individual "image" embeds for the video token scatter
    std::vector<ov::Tensor> video_embeds;
    for (size_t video_id : videos_sequence) {
        const auto& encoded_video = videos.at(video_id);
        // video_features is [1, total_tokens, hidden_size] — reshape to individual frame embeds
        const size_t tokens_per_frame = encoded_video.num_video_tokens / encoded_video.frame_num;
        const size_t hidden_size = encoded_video.video_features.get_shape()[2];
        const auto elem_type = encoded_video.video_features.get_element_type();
        const size_t bytes_per_frame = tokens_per_frame * hidden_size * elem_type.size();
        const uint8_t* src = static_cast<const uint8_t*>(encoded_video.video_features.data());

        for (size_t f = 0; f < encoded_video.frame_num; ++f) {
            auto* frame_ptr = const_cast<uint8_t*>(src + f * bytes_per_frame);
            ov::Tensor frame_embed(elem_type, {1, tokens_per_frame, hidden_size}, frame_ptr);
            video_embeds.push_back(std::move(frame_embed));
        }
    }

    ov::Tensor input_ids = get_encoded_input_ids(prompt, metrics);

    if (has_per_layer_embeddings()) {
        m_lm_extra_inputs["per_layer_inputs"] = get_per_layer_embeddings(input_ids);
    }

    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    encode_vision_token_ids();

    ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());

    if (image_embeds.empty() && video_embeds.empty()) {
        text_embeds.copy_to(inputs_embeds);
        return {std::move(inputs_embeds), std::move(input_ids)};
    }

    // Merge image embeddings at image_token_id positions
    if (!image_embeds.empty()) {
        inputs_embeds =
            utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, m_image_token_id);
    } else {
        inputs_embeds = std::move(text_embeds);
    }

    // Merge video embeddings at video_token_id positions
    if (!video_embeds.empty()) {
        inputs_embeds =
            utils::merge_text_and_image_embeddings_llava(input_ids, inputs_embeds, video_embeds, m_video_token_id);
    }

    return {std::move(inputs_embeds), std::move(input_ids)};
}

ov::Tensor InputsEmbedderGemma4::get_inputs_embeds(const std::string& prompt,
                                                   const std::vector<EncodedImage>& images,
                                                   VLMPerfMetrics& metrics,
                                                   bool recalculate_merged_embeddings,
                                                   const std::vector<size_t>& images_sequence) {
    return compute_inputs_embeds(prompt, images, {}, metrics, images_sequence, {}).first;
}

ov::Tensor InputsEmbedderGemma4::get_inputs_embeds(
    const std::string& prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    const std::vector<ov::genai::EncodedVideo>& videos,
    ov::genai::VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& images_sequence,
    const std::vector<size_t>& videos_sequence,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    return compute_inputs_embeds(prompt, images, videos, metrics, images_sequence, videos_sequence).first;
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderGemma4::get_inputs_embeds_with_token_type_ids(
    const std::string& prompt,
    const std::vector<EncodedImage>& images,
    VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& images_sequence) {
    auto [inputs_embeds, input_ids] = compute_inputs_embeds(prompt, images, {}, metrics, images_sequence, {});
    ov::Tensor token_type_ids = get_token_type_ids(input_ids);
    return {std::move(inputs_embeds), std::move(token_type_ids)};
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderGemma4::get_inputs_embeds_with_token_type_ids(
    const std::string& prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    const std::vector<ov::genai::EncodedVideo>& videos,
    ov::genai::VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& images_sequence,
    const std::vector<size_t>& videos_sequence,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    auto [inputs_embeds, input_ids] =
        compute_inputs_embeds(prompt, images, videos, metrics, images_sequence, videos_sequence);
    ov::Tensor token_type_ids = get_token_type_ids(input_ids);
    return {std::move(inputs_embeds), std::move(token_type_ids)};
}

bool InputsEmbedderGemma4::has_token_type_ids() const {
    // Optimum-intel exports `token_type_ids` as an LM input when
    // `text_config.use_bidirectional_attention == "vision"` (see Gemma4OpenVINOConfig.with_behavior).
    return m_vlm_config.use_bidirectional_attention == "vision";
}

ov::Tensor InputsEmbedderGemma4::get_token_type_ids(const ov::Tensor& input_ids) {
    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    const size_t num_elements = input_ids.get_size();
    ov::Tensor token_type_ids(ov::element::i64, input_ids.get_shape());
    int64_t* token_type_data = token_type_ids.data<int64_t>();
    for (size_t i = 0; i < num_elements; ++i) {
        token_type_data[i] = (input_ids_data[i] == m_image_token_id || input_ids_data[i] == m_video_token_id) ? 1 : 0;
    }
    return token_type_ids;
}

void InputsEmbedderGemma4::encode_vision_token_ids() {
    std::call_once(m_vision_token_ids_once_flag, [this]() {
        const auto encoded_vision_tokens =
            m_tokenizer
                .encode(m_vlm_config.image_token + m_vlm_config.video_token, ov::genai::add_special_tokens(false))
                .input_ids;
        OPENVINO_ASSERT(encoded_vision_tokens.get_size() == 2, "Encoded vision tokens must contain two tokens");
        m_image_token_id = encoded_vision_tokens.data<int64_t>()[0];
        m_video_token_id = encoded_vision_tokens.data<int64_t>()[1];
    });
}

const std::unordered_map<std::string, ov::Tensor>& InputsEmbedderGemma4::get_lm_extra_inputs() const {
    return m_lm_extra_inputs;
}

void InputsEmbedderGemma4::patch_chat_template() {
    if (m_vlm_config.model_type != VLMModelType::GEMMA4_UNIFIED) {
        return;
    }

    std::string patched_chat_template = m_tokenizer.get_chat_template();
    // minja does not support Python-style implicit concatenation of adjacent multiline string literals:
    //     "first "
    //     "second"
    // Normalize the pair to "first second" before parsing.
    const std::regex multiline_string_concatenation{R"("[ \t]*\r?\n[ \t]*")"};
    patched_chat_template = std::regex_replace(patched_chat_template, multiline_string_concatenation, "");

    m_tokenizer.set_chat_template(patched_chat_template);
}

}  // namespace ov::genai
