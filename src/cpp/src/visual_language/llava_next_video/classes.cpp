// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/llava_next_video/classes.hpp"
#include "visual_language/clip.hpp"
#include "visual_language/processor_config.hpp"


namespace ov::genai {

std::pair<size_t, size_t> get_unpadded_features(size_t height, size_t width, size_t patches_height, size_t patches_width, size_t scale_height, size_t scale_width) {
    size_t current_height = patches_height * scale_height;
    size_t current_width = patches_width * scale_width;
    
    float original_aspect_ratio = (float)width / height;
    float current_aspect_ratio = (float)current_width / current_height;
    if (original_aspect_ratio > current_aspect_ratio) {
        size_t new_height = std::floor(height * ((float)current_width / width));
        size_t padding = (current_height - new_height) / 2;
        current_height -= padding * 2;
    }
    else {
        size_t new_width = std::floor(width * ((float)current_height / height));
        size_t padding = (current_width - new_width) / 2;
        current_width -= padding * 2;
    }

    size_t unpadded_features = current_height * current_width;
    size_t newline_features = current_height;
    return {unpadded_features, newline_features};
}

clip_image_f32 preprocess_clip_image_llava_next_video(const clip_image_u8& image, ProcessorConfig& config) {
    // Resize
    clip_image_u8 resized_image;
    int target_size = config.size_shortest_edge;
    float scale = static_cast<float>(target_size) / std::min(image.nx, image.ny);
    int new_width = static_cast<int>(image.nx * scale);
    int new_height = static_cast<int>(image.ny * scale);
    bicubic_resize(image, resized_image, new_width, new_height);

    // Center crop
    clip_image_u8 cropped_image = center_crop(resized_image, config.crop_size_height, config.crop_size_width);

    // Normalize
    clip_ctx_double ctx;

    // apply fused normalize and rescale to 1.0/255, by the formula: 
    // new_mean = mean * (1.0 / scale), new_std = std * (1.0 / rescale_factor)
    for (size_t c = 0; c < 3; c++) {
        ctx.image_mean[c] = config.image_mean[c] * 255;
        ctx.image_std[c] = config.image_std[c] * 255;
    }

    return normalize_and_convert_to_chw(cropped_image, ctx);
}

VisionEncoderLLaVANextVideo::VisionEncoderLLaVANextVideo(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap properties) : VisionEncoderLLaVANext(model_dir, device, properties) {
    auto compiled_model = utils::singleton_core().compile_model(model_dir / "openvino_multi_modal_projector_model.xml", device, {});
    m_ireq_queue_multi_modal_projector = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    compiled_model = utils::singleton_core().compile_model(model_dir / "openvino_vision_resampler_model.xml", device, {});
    m_ireq_queue_vision_resampler = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    auto vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_dir, "config.json");
    m_patch_size = vlm_config.vision_config_patch_size;
}

VisionEncoderLLaVANextVideo::VisionEncoderLLaVANextVideo(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config) : VisionEncoderLLaVANext{models_map, config_dir_path, device, device_config} {
    const auto& resampler_model = utils::get_model_weights_pair(models_map, "resampler").first;
    const auto& resampler_weights = utils::get_model_weights_pair(models_map, "resampler").second;
    const auto& mm_projector_model = utils::get_model_weights_pair(models_map, "multi_modal_projector").first;
    const auto& mm_projector_weights = utils::get_model_weights_pair(models_map, "multi_modal_projector").second;

    auto compiled_model = utils::singleton_core().compile_model(resampler_model, resampler_weights, device, device_config);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM resampler model");
    m_ireq_queue_vision_resampler = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });

    compiled_model = utils::singleton_core().compile_model(mm_projector_model, mm_projector_weights, device, device_config);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM multi modal projector model");
    m_ireq_queue_multi_modal_projector = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    auto vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");
    m_patch_size = vlm_config.vision_config_patch_size;
}

EncodedImage VisionEncoderLLaVANextVideo::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard_mm_projector(this->m_ireq_queue_multi_modal_projector.get());
    ov::InferRequest& mm_projector = infer_request_guard_mm_projector.get();
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    // preprocess image
    ov::Tensor pixel_values = get_pixel_values_llava_next(image, config);
    auto pixel_values_shape = pixel_values.get_shape();

    // infer vision eztracting models
    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();
    mm_projector.set_tensor("image_features", encoder.get_tensor("last_hidden_state"));
    mm_projector.infer();
    const ov::Tensor& infer_output = mm_projector.get_output_tensor();

    ImageSize resized_source_size{config.crop_size_height / config.patch_size, config.crop_size_width / config.patch_size};

    // Gen number of patches
    ImageSize original_image_size{image.get_shape().at(1), image.get_shape().at(2)};
    auto best_resolution = select_best_resolution({original_image_size.width, original_image_size.height}, config.image_grid_pinpoints);
    int num_patches_w = best_resolution.first / config.size_shortest_edge;
    int num_patches_h = best_resolution.second / config.size_shortest_edge;

    // Get unpadded features
    size_t height = pixel_values_shape[2];
    size_t width = pixel_values_shape[3];
    size_t patches_height = height / config.patch_size;
    size_t patches_width = width / config.patch_size;
    size_t scale_height = best_resolution.second / height;
    size_t scale_width = best_resolution.first / width;
    size_t unpadded_features, newline_features;
    std::tie(unpadded_features, newline_features) = get_unpadded_features(original_image_size.height, original_image_size.width, patches_height, patches_width, scale_height, scale_width);

    // get number of image tokens
    size_t base_features = patches_height * patches_width;
    size_t num_image_tokens = unpadded_features + newline_features + base_features;

    EncodedImage encoded_image;
    // copy infer output to ensure it is not overwritten during next inference
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    encoded_image.resized_source = std::move(image_features);
    encoded_image.resized_source_size = resized_source_size;
    encoded_image.patches_grid = {num_patches_h, num_patches_w};
    encoded_image.original_image_size = original_image_size;
    encoded_image.num_image_tokens = num_image_tokens;
    return encoded_image;
}


NormlizedPrompt InputsEmbedderLLaVANextVideo::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    std::string image_token = m_vlm_config.im_start;
    auto [unified_prompt, images_sequence] = normalize(prompt, image_token, image_token, base_id, images.size());
    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        const EncodedImage& encoded_image = images.at(new_image_id - base_id);
        std::string expanded_tag;
        for (size_t idx = 0; idx < encoded_image.num_image_tokens; ++idx) {
            expanded_tag += image_token;
        }
        expanded_tag += '\n';
        OPENVINO_ASSERT(searched_pos < unified_prompt.length());
        searched_pos = unified_prompt.find(image_token, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos);
        unified_prompt.replace(searched_pos, image_token.length(), expanded_tag);
        searched_pos += expanded_tag.length();
    }
    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

InputsEmbedderLLaVANextVideo::InputsEmbedderLLaVANextVideo(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config) :
    InputsEmbedderLLaVANext(vlm_config, model_dir, device, device_config) { }

InputsEmbedderLLaVANextVideo::InputsEmbedderLLaVANextVideo(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    InputsEmbedderLLaVANext(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) { }


ov::Tensor InputsEmbedderLLaVANextVideo::get_inputs_embeds(
    const std::string& prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    const std::vector<ov::genai::EncodedVideo>& videos,
    ov::genai::VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& images_sequence,
    const std::vector<size_t>& videos_sequence) {
    
    ov::Tensor image_newline;
    size_t searched_pos = 0;
    std::vector<ov::Tensor> image_embeds;
    for (size_t new_image_id : images_sequence) {
        const EncodedImage& encoded_image = images.at(new_image_id);
        if (!image_newline) {
            size_t embed_dim = encoded_image.resized_source.get_shape().at(2);
            image_newline = ov::Tensor(encoded_image.resized_source.get_element_type(), {embed_dim});
            float* image_newline_data = image_newline.data<float>();
            std::copy(m_vlm_config.image_newline.begin(), m_vlm_config.image_newline.end(), image_newline_data);
        }
        image_embeds.push_back(pack_image_features_llava_next(encoded_image, image_newline));
    }

    std::vector<ov::Tensor> video_embeds;
    for (size_t video_id : videos_sequence) {
        const EncodedVideo& encoded_video = videos.at(video_id);
        video_embeds.push_back(encoded_video.video_feautures);
    }

    // llava-next-video tokenizer always adds special tokens in pytorch
    set_add_special_tokens(true);
    ov::Tensor input_ids = get_encoded_input_ids(prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    if (image_embeds.empty() && video_embeds.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    auto start_tokenizer_time = std::chrono::steady_clock::now();
    ov::Tensor encoded_image_token = m_tokenizer.encode(m_vlm_config.im_start, ov::genai::add_special_tokens(false)).input_ids;
    ov::Tensor encoded_video_token = m_tokenizer.encode(m_vlm_config.video_start, ov::genai::add_special_tokens(false)).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    if (!image_embeds.empty()) {
        int64_t image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];
        text_embeds = utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, image_token_id);
    }
    if (!video_embeds.empty()) {
        int64_t video_token_id = encoded_video_token.data<int64_t>()[encoded_video_token.get_size() - 1];
        text_embeds = utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, video_embeds, video_token_id);
    }
    return text_embeds;
}

std::pair<std::vector<ov::Tensor>, size_t> VisionEncoderLLaVANextVideo::preprocess_frames(const std::vector<ov::Tensor>& frames) {
    std::vector<ov::Tensor> res;

    // preprocess frames
    ProcessorConfig config = utils::from_any_map({}, m_processor_config);
    size_t num_frames = frames.size();
    for (size_t i=0; i < num_frames; i++) {
        clip_image_u8 clip_image = tensor_to_clip_image_u8(frames[i]);
        auto preprocessed = preprocess_clip_image_llava_next_video(clip_image, config);
        auto preprocessed_tensor = clip_image_f32_to_tensor(preprocessed);
        res.push_back(preprocessed_tensor);

    }

    ov::Shape resized_shape = res[0].get_shape();
    size_t height = resized_shape[2];
    size_t width = resized_shape[3];

    size_t num_video_tokens = ((float)height / m_patch_size) * ((float)width / m_patch_size);
    num_video_tokens = num_video_tokens / 4 * num_frames;

    return {res, num_video_tokens};
}

std::vector<ov::genai::EncodedVideo> InputsEmbedderLLaVANextVideo::encode_videos(const std::vector<ov::Tensor>& videos) {
    std::vector<ov::genai::EncodedVideo> encoded_videos;
    for (const auto video: videos) {
        std::vector<ov::Tensor> frames = to_single_image_tensors({video});
        auto vision_encoder = std::static_pointer_cast<VisionEncoderLLaVANextVideo>(m_vision_encoder);
        auto [prepprocessed_frames, num_video_tokens] = vision_encoder->preprocess_frames(frames);

        // concat preprocessed frames to single tensor
        ov::Shape concat_shape = prepprocessed_frames[0].get_shape();
        concat_shape[0] = prepprocessed_frames.size();
        ov::Tensor concatinated_frames = ov::Tensor(prepprocessed_frames[0].get_element_type(), concat_shape);
        float* frames_data = concatinated_frames.data<float>();
        for (size_t i = 0; i < prepprocessed_frames.size(); i++) {
            memcpy(frames_data, prepprocessed_frames[i].data(), prepprocessed_frames[i].get_byte_size());
            frames_data+=ov::shape_size(prepprocessed_frames[i].get_shape());
        }

        // infer video feature extration models
        CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(vision_encoder->get_vision_encoder());
        ov::InferRequest& encoder = infer_request_guard.get();
        CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard_mm_projector(vision_encoder->get_multi_modal_projector());
        ov::InferRequest& mm_projector = infer_request_guard_mm_projector.get();
        CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard_resampler(vision_encoder->get_vision_resampler());
        ov::InferRequest& resampler = infer_request_guard_resampler.get();
        encoder.set_tensor("pixel_values", concatinated_frames);
        encoder.infer();
        resampler.set_input_tensor(encoder.get_tensor("last_hidden_state"));
        resampler.infer();
        mm_projector.set_tensor("image_features", resampler.get_output_tensor());
        mm_projector.infer();

        // copy infer output to ensure it is not overwritten during next inference
        const ov::Tensor& infer_output = mm_projector.get_output_tensor();
        ov::Tensor video_features(infer_output.get_element_type(), infer_output.get_shape());
        std::memcpy(video_features.data(), infer_output.data(), infer_output.get_byte_size());

        EncodedVideo encoded_video;
        ov::Shape new_shape = {1, video_features.get_shape()[0] * video_features.get_shape()[1], video_features.get_shape()[2]};
        video_features.set_shape(new_shape);
        encoded_video.video_feautures = std::move(video_features);
        encoded_video.num_video_tokens = num_video_tokens;
        encoded_videos.push_back(encoded_video);
    }
    return encoded_videos;
}

NormlizedPrompt InputsEmbedderLLaVANextVideo::normalize_prompt(const std::string& prompt,
    size_t base_image_id,
    size_t base_video_id,
    const std::vector<EncodedImage>& images,
    const std::vector<EncodedVideo>& videos) const {
    if (!videos.size()) {
        return normalize_prompt(prompt, base_image_id, images);
    }
    std::string video_token = m_vlm_config.video_start;
    auto [unified_prompt, video_sequence] = normalize(prompt, video_token, video_token, base_video_id, videos.size());
    size_t searched_pos = 0;
    for (size_t new_image_id : video_sequence) {
        const EncodedVideo& encoded_video = videos.at(new_image_id - base_video_id);
        std::string expanded_tag;
        for (size_t idx = 0; idx < encoded_video.num_video_tokens; ++idx) {
            expanded_tag += video_token;
        }
        expanded_tag += '\n';
        OPENVINO_ASSERT(searched_pos < unified_prompt.length());
        searched_pos = unified_prompt.find(video_token, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos);
        unified_prompt.replace(searched_pos, video_token.length(), expanded_tag);
        searched_pos += expanded_tag.length();
    }
    std::vector<size_t> images_sequence;
    // normalize images after videos to make sure image tokens appended at the start of prompt before video tokens
    if (images.size()) {
        auto normalize_res = normalize_prompt(unified_prompt, base_image_id, images);
        unified_prompt = normalize_res.unified_prompt;
        images_sequence = normalize_res.images_sequence;
    }
    return {std::move(unified_prompt), std::move(images_sequence), std::move(video_sequence)};
}

} // namespace ov::genai