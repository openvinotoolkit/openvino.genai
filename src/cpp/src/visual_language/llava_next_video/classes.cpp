// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/llava_next_video/classes.hpp"
#include "visual_language/clip.hpp"
#include "visual_language/processor_config.hpp"
#include "openvino/opsets/opset13.hpp"


namespace ov::genai {

namespace {

std::shared_ptr<ov::Node> create_bicubic_resize(std::shared_ptr<ov::Node> input, std::shared_ptr<ov::Node> target_size) {
    using namespace ov::op;

    // Convert to float32 before interpolation (required for bicubic)
    auto input_f32 = std::make_shared<v0::Convert>(input, ov::element::f32);

    // For NHWC format, resize axes are [1, 2] (height, width dimensions)
    auto axes = v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 2});

    v11::Interpolate::InterpolateAttrs attrs;
    attrs.mode = v11::Interpolate::InterpolateMode::CUBIC;
    attrs.shape_calculation_mode = v11::Interpolate::ShapeCalcMode::SIZES;
    attrs.coordinate_transformation_mode = v11::Interpolate::CoordinateTransformMode::ASYMMETRIC;
    attrs.cube_coeff = -0.5f;  // Catmull-Rom bicubic coefficient (a = -0.5), chosen to match CPU preprocessing
    attrs.nearest_mode = v11::Interpolate::NearestMode::FLOOR;
    attrs.pads_begin = {0, 0};
    attrs.pads_end = {0, 0};
    attrs.antialias = false;

    return std::make_shared<v11::Interpolate>(input_f32, target_size, axes, attrs);
}

std::shared_ptr<ov::Node> create_mean_scale(std::shared_ptr<ov::Node> input_u8_or_f32, const ProcessorConfig& config) {
    using namespace ov::op;

    std::shared_ptr<ov::Node> input_f32;

    // Convert to float32 if input is uint8, otherwise use as-is
    if (input_u8_or_f32->get_element_type() == ov::element::u8) {
        input_f32 = std::make_shared<v0::Convert>(input_u8_or_f32, ov::element::f32);
    } else {
        input_f32 = std::move(input_u8_or_f32);
    }

    // Follow the original mean_scale() function logic exactly, in tensor form:
    // Per-element, per-channel normalization:
    // (float(x) / 255.0f - config.image_mean[c]) / config.image_std[c], implemented via OV ops with broadcasting.
    // Step 1: x / 255.0
    auto scale_255 = v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{255.0f});
    auto divided_by_255 = std::make_shared<v1::Divide>(input_f32, scale_255);

    // Step 2: Create mean and std constants [R, G, B] - broadcasted along channel dimension
    // For NHWC format, we need shape [1, 1, 1, 3] to broadcast correctly
    auto mean_const = v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 3},
        std::vector<float>{config.image_mean[0], config.image_mean[1], config.image_mean[2]});
    auto std_const = v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 3},
        std::vector<float>{config.image_std[0], config.image_std[1], config.image_std[2]});

    // Step 3: (x/255.0 - mean)
    auto mean_subtracted = std::make_shared<v1::Subtract>(divided_by_255, mean_const);

    // Step 4: (x/255.0 - mean) / std
    auto result = std::make_shared<v1::Divide>(mean_subtracted, std_const);

    return result;
}

std::shared_ptr<ov::Node> create_channels_first(std::shared_ptr<ov::Node> input_nhwc) {
    using namespace ov::op;

    // Transpose from NHWC (0,1,2,3) to NCHW (0,3,1,2)
    auto transpose_order = v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
    return std::make_shared<v1::Transpose>(input_nhwc, transpose_order);
}

std::shared_ptr<ov::Node> create_center_crop(std::shared_ptr<ov::Node> input, std::shared_ptr<ov::Node> crop_size) {
    using namespace ov::op;

    // Extract crop height and width from crop_size
    auto gather_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
    auto idx_0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto idx_1 = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto idx_2 = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2});
    auto crop_height = std::make_shared<v8::Gather>(crop_size, idx_0, gather_axis);
    auto crop_width = std::make_shared<v8::Gather>(crop_size, idx_1, gather_axis);

    // Get input shape
    auto shape_node = std::make_shared<v3::ShapeOf>(input);
    auto H = std::make_shared<v8::Gather>(shape_node, idx_1, gather_axis);
    auto W = std::make_shared<v8::Gather>(shape_node, idx_2, gather_axis);

    // Calculate start positions: start_y = (H - crop_height) / 2, start_x = (W - crop_width) / 2
    auto const_2 = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2});
    auto start_y = std::make_shared<v1::Divide>(std::make_shared<v1::Subtract>(H, crop_height), const_2);
    auto start_x = std::make_shared<v1::Divide>(std::make_shared<v1::Subtract>(W, crop_width), const_2);

    // Calculate end positions: end_y = start_y + crop_height, end_x = start_x + crop_width
    auto end_y = std::make_shared<v1::Add>(start_y, crop_height);
    auto end_x = std::make_shared<v1::Add>(start_x, crop_width);

    // Create slice start and stop vectors
    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto max_val = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{std::numeric_limits<int64_t>::max()});

    // start = [0, start_y, start_x, 0]
    auto start = std::make_shared<v0::Concat>(ov::NodeVector{zero, start_y, start_x, zero}, 0);
    // stop = [max, end_y, end_x, max]
    auto stop = std::make_shared<v0::Concat>(ov::NodeVector{max_val, end_y, end_x, max_val}, 0);
    // step = [1, 1, 1, 1]
    auto step = v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, 1, 1, 1});

    // Apply slice
    auto sliced = std::make_shared<v8::Slice>(input, start, stop, step);

    return sliced;
}

// Helper function to calculate resize dimensions based on shortest edge
ImageSize calculate_resize_dimensions(
    const ImageSize& original_size,
    size_t target_shortest_edge) {
    float scale = static_cast<float>(target_shortest_edge) / std::min(original_size.height, original_size.width);
    size_t new_height = static_cast<size_t>(original_size.height * scale);
    size_t new_width = static_cast<size_t>(original_size.width * scale);
    return {new_height, new_width};
}

// Helper function to set preprocessing parameters for integrated OV preprocessing model
void set_preprocess_parameters(
    ov::InferRequest& encoder,
    const ov::Tensor& input_frames,
    const ImageSize& original_size,
    const ProcessorConfig& config) {

    // Calculate resize target size
    auto resized_size = calculate_resize_dimensions(original_size, config.size_shortest_edge);

    // Set resize target size
    ov::Tensor target_size_tensor(ov::element::i64, {2});
    target_size_tensor.data<int64_t>()[0] = resized_size.height;
    target_size_tensor.data<int64_t>()[1] = resized_size.width;

    // Set crop size
    ov::Tensor crop_size_tensor(ov::element::i64, {2});
    crop_size_tensor.data<int64_t>()[0] = config.crop_size_height;
    crop_size_tensor.data<int64_t>()[1] = config.crop_size_width;

    encoder.set_input_tensor(0, input_frames);
    encoder.set_input_tensor(1, target_size_tensor);
    encoder.set_input_tensor(2, crop_size_tensor);
}

bool can_use_ov_vision_preprocess() {
    const char* env = std::getenv("VISION_PREPROCESS");
    return !(env && std::string(env) == "CPP");
}

std::shared_ptr<ov::Model> patch_preprocess_into_vision_encoder_model(
    const std::shared_ptr<ov::Model>& vision_encoder_model,
    const ProcessorConfig& config) {
    using namespace ov;
    using namespace ov::op;

    // Input: concatenated image/video frames in NHWC format (uint8)
    // Shape: {num_frames, -1, -1, 3} => {batch=num_frames, height=dynamic, width=dynamic, channels=3 (RGB)}
    auto input_frames = std::make_shared<v0::Parameter>(element::u8, PartialShape{-1, -1, -1, 3});
    input_frames->set_friendly_name("input_frames");

    // Target size for bicubic resize [height, width]
    auto resize_target_size = std::make_shared<v0::Parameter>(element::i64, PartialShape{2});
    resize_target_size->set_friendly_name("resize_target_size");

    // Crop size [height, width]
    auto crop_size = std::make_shared<v0::Parameter>(element::i64, PartialShape{2});
    crop_size->set_friendly_name("crop_size");

    // Apply preprocessing operations
    auto resized = create_bicubic_resize(input_frames, resize_target_size);
    auto cropped = create_center_crop(resized, crop_size);
    auto normalized = create_mean_scale(cropped, config);
    auto preprocessed = create_channels_first(normalized);

    // Connect preprocessing output to vision encoder input
    auto vision_params = vision_encoder_model->get_parameters();
    auto vision_results = vision_encoder_model->get_results();

    // Replace pixel_values parameter with preprocessing output
    vision_params[0]->output(0).replace(preprocessed);

    return std::make_shared<Model>(
        vision_results,
        ParameterVector{input_frames, resize_target_size, crop_size}
    );
}

} // namespace

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
    auto resized_size = calculate_resize_dimensions({static_cast<size_t>(image.ny), static_cast<size_t>(image.nx)}, config.size_shortest_edge);
    bicubic_resize(image, resized_image, static_cast<int>(resized_size.width), static_cast<int>(resized_size.height));

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
    const ov::AnyMap properties) : VisionEncoderLLaVANext(model_dir, device, properties),
        use_ov_vision_preprocess(can_use_ov_vision_preprocess()) {
    if (use_ov_vision_preprocess) {
        // Create integrated preprocessing + vision encoder model for image/video processing
        auto vision_encoder_model = utils::singleton_core().read_model(model_dir / "openvino_vision_embeddings_model.xml");
        auto model = patch_preprocess_into_vision_encoder_model(vision_encoder_model, m_processor_config);
        auto compiled_model = utils::singleton_core().compile_model(model, device, properties);
        // Overwrite vision encoder queue with integrated model
        m_ireq_queue_vision_encoder = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
            compiled_model.get_property(ov::optimal_number_of_infer_requests),
            [&compiled_model]() -> ov::InferRequest {
                return compiled_model.create_infer_request();
            });
    }

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
    const ov::AnyMap device_config) : VisionEncoderLLaVANext{models_map, config_dir_path, device, device_config},
        use_ov_vision_preprocess(can_use_ov_vision_preprocess()) {
    if (use_ov_vision_preprocess) {
        // Create integrated preprocessing + vision encoder model for image/video processing
        const auto& [vision_encoder_model, vision_encoder_weights] = utils::get_model_weights_pair(models_map, "vision_embeddings");
        auto vision_encoder_model_original = utils::singleton_core().read_model(vision_encoder_model, vision_encoder_weights);
        auto model = patch_preprocess_into_vision_encoder_model(vision_encoder_model_original, m_processor_config);
        auto compiled_model = utils::singleton_core().compile_model(model, device, device_config);
        // Overwrite vision encoder queue with integrated model
        m_ireq_queue_vision_encoder = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
            compiled_model.get_property(ov::optimal_number_of_infer_requests),
            [&compiled_model]() -> ov::InferRequest {
                return compiled_model.create_infer_request();
            });
    }

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

    ov::Shape pixel_values_shape;
    if (use_ov_vision_preprocess) {
        // Use integrated OV preprocessing model with batch processing similar to get_pixel_values_llava_next
        clip_image_u8 input_image = tensor_to_clip_image_u8(image);

        std::pair<int, int> size{config.size_shortest_edge, config.size_shortest_edge};
        auto patch_size = config.crop_size_height;
        auto image_patches = get_image_patches(input_image, config.image_grid_pinpoints, size, patch_size);

        size_t num_patches = image_patches.size();

        // Get dimensions from first patch
        size_t patch_height = image_patches[0].ny;
        size_t patch_width = image_patches[0].nx;

        // Concatenate all patches into a single batch tensor (similar to preprocess_frames_ov)
        ov::Shape concat_shape = {num_patches, patch_height, patch_width, 3};
        ov::Tensor concatenated_patches(ov::element::u8, concat_shape);

        uint8_t* concat_data = concatenated_patches.data<uint8_t>();
        for (size_t i = 0; i < num_patches; i++) {
            // clip_image_u8 has layout HWC, copy directly
            std::memcpy(concat_data, image_patches[i].buf.data(), image_patches[i].buf.size());
            concat_data += image_patches[i].buf.size();
        }

        // Set inputs for integrated preprocessing model
        set_preprocess_parameters(encoder, concatenated_patches, {patch_height, patch_width}, config);

        // Set pixel_values_shape for later use
        pixel_values_shape = {num_patches, 3, static_cast<size_t>(config.crop_size_height), static_cast<size_t>(config.crop_size_width)};
    } else {
        // Use CPU preprocessing
        ov::Tensor pixel_values = get_pixel_values_llava_next(image, config);
        pixel_values_shape = pixel_values.get_shape();
        encoder.set_tensor("pixel_values", pixel_values);
    }

    // infer vision extracting models
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


NormalizedPrompt InputsEmbedderLLaVANextVideo::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    std::string image_token = m_vlm_config.im_start;
    auto [unified_prompt, images_sequence] = normalize(prompt, image_token, image_token, base_id, images.size(), VisionType::IMAGE);
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
    const std::vector<size_t>& videos_sequence,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {

    ov::Tensor image_newline;
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
        video_embeds.push_back(encoded_video.video_features);
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

ov::Tensor VisionEncoderLLaVANextVideo::preprocess_frames_cpp(const std::vector<ov::Tensor>& frames) {
    ProcessorConfig config = get_processor_config();
    size_t num_frames = frames.size();
    std::vector<ov::Tensor> preprocessed_frames;
    preprocessed_frames.reserve(num_frames);

    // Preprocess frames using CPU
    for (size_t i = 0; i < num_frames; i++) {
        clip_image_u8 clip_image = tensor_to_clip_image_u8(frames[i]);
        auto preprocessed = preprocess_clip_image_llava_next_video(clip_image, config);
        preprocessed_frames.push_back(clip_image_f32_to_tensor(preprocessed));
    }

    // Concatenate preprocessed frames to single tensor
    const ov::Shape& first_shape = preprocessed_frames[0].get_shape();
    ov::Shape concat_shape = first_shape;
    concat_shape[0] = num_frames;
    ov::Tensor concatenated_frames(preprocessed_frames[0].get_element_type(), concat_shape);

    size_t frame_byte_size = preprocessed_frames[0].get_byte_size();
    float* frames_data = concatenated_frames.data<float>();
    for (size_t i = 0; i < num_frames; i++) {
        std::memcpy(frames_data, preprocessed_frames[i].data(), frame_byte_size);
        frames_data += ov::shape_size(first_shape);
    }

    return concatenated_frames;
}

std::vector<ov::genai::EncodedVideo> InputsEmbedderLLaVANextVideo::encode_videos(const std::vector<ov::Tensor>& videos) {
    auto vision_encoder = std::static_pointer_cast<VisionEncoderLLaVANextVideo>(m_vision_encoder);
    auto config = vision_encoder->get_processor_config();

    std::vector<ov::genai::EncodedVideo> encoded_videos;
    for (const auto video: videos) {
        std::vector<ov::Tensor> frames = to_single_image_tensors({video});
        size_t num_frames = frames.size();

        // Calculate num_video_tokens (same for both OV and CPU preprocessing)
        size_t num_video_tokens = ((config.crop_size_height / vision_encoder->get_patch_size()) * 
                                   (config.crop_size_width / vision_encoder->get_patch_size()) / 4) * num_frames;

        // infer video feature extraction models
        CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(vision_encoder->get_vision_encoder());
        ov::InferRequest& encoder = infer_request_guard.get();

        if (vision_encoder->get_use_ov_vision_preprocess()) {
            // Use integrated OV preprocessing model - pass video tensor directly
            auto frame_shape = frames[0].get_shape();
            size_t orig_height = frame_shape[1];
            size_t orig_width = frame_shape[2];

            // Set inputs for integrated model
            set_preprocess_parameters(encoder, video, {orig_height, orig_width}, config);
        } else {
            // Use CPU preprocessing - preprocess and concatenate frames
            ov::Tensor concatenated_frames = vision_encoder->preprocess_frames_cpp(frames);
            encoder.set_tensor("pixel_values", concatenated_frames);
        }

        encoder.infer();

        CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard_resampler(vision_encoder->get_vision_resampler());
        ov::InferRequest& resampler = infer_request_guard_resampler.get();
        CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard_mm_projector(vision_encoder->get_multi_modal_projector());
        ov::InferRequest& mm_projector = infer_request_guard_mm_projector.get();

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
        encoded_video.video_features = std::move(video_features);
        encoded_video.num_video_tokens = num_video_tokens;
        encoded_videos.push_back(encoded_video);
    }
    return encoded_videos;
}

NormalizedPrompt InputsEmbedderLLaVANextVideo::normalize_prompt(const std::string& prompt,
    size_t base_image_id,
    size_t base_video_id,
    const std::vector<EncodedImage>& images,
    const std::vector<EncodedVideo>& videos) const
{
    std::string video_token = m_vlm_config.video_start;
    auto [unified_prompt, video_sequence] = normalize(prompt, video_token, video_token, base_video_id, videos.size(), VisionType::VIDEO);
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
    auto normalize_res = normalize_prompt(unified_prompt, base_image_id, images);
    unified_prompt = normalize_res.unified_prompt;
    images_sequence = normalize_res.images_sequence;
    return {std::move(unified_prompt), std::move(images_sequence), std::move(video_sequence)};
}

} // namespace ov::genai
