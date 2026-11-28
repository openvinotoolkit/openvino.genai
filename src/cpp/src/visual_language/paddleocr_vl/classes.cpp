// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/paddleocr_vl/classes.hpp"

#include <cmath>

#include "openvino/op/clamp.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"
#include "visual_language/qwen3_vl/classes.hpp"
#include "visual_language/vlm_utils.hpp"

namespace ov::genai {

namespace {

// Same order as Python round(): halfway cases are rounded to the nearest even value.
size_t round_half_to_even(double value) {
    return static_cast<size_t>(std::nearbyint(value));
}

/**
 * @brief Merges image preprocessing into the patch embeddings model, replacing its `pixel_values` input:
 * u8 NHWC image -> bicubic resize -> normalization -> patches [N, C * patch_size * patch_size] in
 * spatial-merge-block order, which is the order expected by the vision embeddings merger.
 * The `bilinear_indices` and `bilinear_weights` inputs are kept.
 */
std::shared_ptr<ov::Model> patch_preprocess_into_model(const std::shared_ptr<ov::Model>& model_org,
                                                       const ProcessorConfig& config) {
    std::vector<float> image_mean(config.image_mean.begin(), config.image_mean.end());
    std::vector<float> image_scale(config.image_std.begin(), config.image_std.end());
    for (auto& v : image_mean) {
        v *= 255.0f;
    }
    for (auto& v : image_scale) {
        v = 1.0f / (v * 255.0f);
    }
    auto mean = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, image_mean.size(), 1, 1}, image_mean);
    auto scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, image_scale.size(), 1, 1}, image_scale);

    auto raw_images = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{1, -1, -1, -1});
    raw_images->set_friendly_name("raw_images");
    raw_images->output(0).get_tensor().set_names({"raw_images"});
    auto resize_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2});
    resize_shape->set_friendly_name("resize_shape");
    resize_shape->output(0).get_tensor().set_names({"resize_shape"});
    // [C, grid_h / merge_size, merge_size, patch_size, grid_w / merge_size, merge_size, patch_size]
    auto reshape_shape7d = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{7});
    reshape_shape7d->set_friendly_name("reshape_shape7d");
    reshape_shape7d->output(0).get_tensor().set_names({"reshape_shape7d"});
    // [grid_h * grid_w, C * patch_size * patch_size]
    auto reshape_shape2d = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2});
    reshape_shape2d->set_friendly_name("reshape_shape2d");
    reshape_shape2d->output(0).get_tensor().set_names({"reshape_shape2d"});

    auto image_f32_nchw = qwen2_vl_utils::create_f32_nchw_input(raw_images);
    // HF resizes u8 images with antialiased bicubic and keeps the u8 result, which matches Pillow's bicubic.
    auto image_resized = vlm_utils::create_bicubic_pillow_resize(image_f32_nchw, resize_shape);
    auto image_clamped = std::make_shared<ov::op::v0::Clamp>(image_resized, 0.0, 255.0);
    auto image_rounded = std::make_shared<ov::op::v5::Round>(image_clamped, ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    auto image_normalized = qwen2_vl_utils::create_normalization(image_rounded, mean, scale);
    auto patches_7d = std::make_shared<ov::op::v1::Reshape>(image_normalized, reshape_shape7d, false);
    // -> [grid_h / merge_size, grid_w / merge_size, merge_size, merge_size, C, patch_size, patch_size]
    auto order = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{7}, std::vector<int32_t>{1, 4, 2, 5, 0, 3, 6});
    auto patches_transposed = std::make_shared<ov::op::v1::Transpose>(patches_7d, order);
    auto patches_2d = std::make_shared<ov::op::v1::Reshape>(patches_transposed, reshape_shape2d, false);

    ov::ParameterVector params{raw_images, resize_shape, reshape_shape7d, reshape_shape2d};
    bool pixel_values_replaced = false;
    for (const auto& param : model_org->get_parameters()) {
        if (param->get_output_tensor(0).get_names().count("pixel_values")) {
            ov::replace_node(param, patches_2d);
            pixel_values_replaced = true;
        } else {
            params.push_back(param);
        }
    }
    OPENVINO_ASSERT(pixel_values_replaced, "PaddleOCR-VL vision embeddings model must have a 'pixel_values' input");
    return std::make_shared<ov::Model>(model_org->get_results(), params);
}

}  // namespace

namespace paddleocr_vl_utils {

ImageSize smart_resize(size_t height, size_t width, size_t factor, size_t min_pixels, size_t max_pixels) {
    OPENVINO_ASSERT(height > 0 && width > 0, "Image height and width must be positive");
    if (height < factor) {
        width = round_half_to_even(static_cast<double>(width * factor) / height);
        height = factor;
    }
    if (width < factor) {
        height = round_half_to_even(static_cast<double>(height * factor) / width);
        width = factor;
    }
    OPENVINO_ASSERT(static_cast<double>(std::max(height, width)) / std::min(height, width) <= 200.0,
                    "Absolute aspect ratio must be smaller than 200");

    size_t h_bar = round_half_to_even(static_cast<double>(height) / factor) * factor;
    size_t w_bar = round_half_to_even(static_cast<double>(width) / factor) * factor;
    if (h_bar * w_bar > max_pixels) {
        const double beta = std::sqrt(static_cast<double>(height * width) / max_pixels);
        h_bar = std::max(factor, static_cast<size_t>(std::floor(height / beta / factor)) * factor);
        w_bar = std::max(factor, static_cast<size_t>(std::floor(width / beta / factor)) * factor);
    } else if (h_bar * w_bar < min_pixels) {
        const double beta = std::sqrt(static_cast<double>(min_pixels) / (height * width));
        h_bar = static_cast<size_t>(std::ceil(height * beta / factor)) * factor;
        w_bar = static_cast<size_t>(std::ceil(width * beta / factor)) * factor;
    }
    return ImageSize{h_bar, w_bar};
}

std::pair<ov::Tensor, ov::Tensor> get_bilinear_indices_and_weights(size_t grid_h,
                                                                   size_t grid_w,
                                                                   size_t num_grid_per_side,
                                                                   size_t merge_size) {
    OPENVINO_ASSERT(grid_h % merge_size == 0 && grid_w % merge_size == 0,
                    "Patch grid ", grid_h, "x", grid_w, " must be divisible by merge_size ", merge_size);
    auto [raster_indices, raster_weights] =
        qwen3_vl_utils::get_position_interpolation_indices_and_weights({{1, grid_h, grid_w}}, num_grid_per_side);

    const size_t num_patches = grid_h * grid_w;
    ov::Tensor indices{ov::element::i64, {4, num_patches}};
    ov::Tensor weights{ov::element::f32, {4, num_patches}};
    const int64_t* src_indices = raster_indices.data<const int64_t>();
    const float* src_weights = raster_weights.data<const float>();
    int64_t* dst_indices = indices.data<int64_t>();
    float* dst_weights = weights.data<float>();

    size_t dst = 0;
    for (size_t hb = 0; hb < grid_h / merge_size; ++hb) {
        for (size_t wb = 0; wb < grid_w / merge_size; ++wb) {
            for (size_t hs = 0; hs < merge_size; ++hs) {
                for (size_t ws = 0; ws < merge_size; ++ws, ++dst) {
                    const size_t src = (hb * merge_size + hs) * grid_w + wb * merge_size + ws;
                    for (size_t corner = 0; corner < 4; ++corner) {
                        dst_indices[corner * num_patches + dst] = src_indices[corner * num_patches + src];
                        dst_weights[corner * num_patches + dst] = src_weights[corner * num_patches + src];
                    }
                }
            }
        }
    }
    return {indices, weights};
}

}  // namespace paddleocr_vl_utils

VisionEncoderPaddleOCRVL::VisionEncoderPaddleOCRVL(const std::filesystem::path& model_dir,
                                                   const std::string& device,
                                                   const ov::AnyMap properties)
    : VisionEncoderQwen2VL(model_dir, ConfigOnlyTag{}) {
    const auto vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_dir, "config.json");
    m_num_grid_per_side = vlm_config.vision_config_image_size / vlm_config.vision_config_patch_size;
    compile(utils::singleton_core().read_model(model_dir / "openvino_vision_embeddings_model.xml"), device, properties);
}

VisionEncoderPaddleOCRVL::VisionEncoderPaddleOCRVL(const ModelsMap& models_map,
                                                   const std::filesystem::path& config_dir_path,
                                                   const std::string& device,
                                                   const ov::AnyMap properties)
    : VisionEncoderQwen2VL(models_map, config_dir_path, ConfigOnlyTag{}) {
    const auto vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");
    m_num_grid_per_side = vlm_config.vision_config_image_size / vlm_config.vision_config_patch_size;
    const auto& [model, weights] = utils::get_model_weights_pair(models_map, "vision_embeddings");
    compile(utils::singleton_core().read_model(model, weights), device, properties);
}

void VisionEncoderPaddleOCRVL::compile(const std::shared_ptr<ov::Model>& model,
                                       const std::string& device,
                                       const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_processor_config.temporal_patch_size == 1,
                    "PaddleOCR-VL expects temporal_patch_size == 1, got ", m_processor_config.temporal_patch_size);
    OPENVINO_ASSERT(m_num_grid_per_side > 0, "PaddleOCR-VL vision_config.image_size and patch_size must be set");
    auto compiled_model = utils::singleton_core().compile_model(patch_preprocess_into_model(model, m_processor_config),
                                                                device,
                                                                utils::get_model_properties(properties, "vision_embeddings", device));
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM vision embeddings model");
    m_ireq_queue_vision_encoder = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
}

EncodedImage VisionEncoderPaddleOCRVL::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    const ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);
    const ov::Shape image_shape = image.get_shape();
    OPENVINO_ASSERT(image_shape.size() == 4 && image_shape.at(0) == 1 && image_shape.at(3) == 3,
                    "PaddleOCR-VL expects a single NHWC RGB image, got shape ", image_shape);

    const size_t factor = config.patch_size * config.merge_size;
    const ImageSize target_size = paddleocr_vl_utils::smart_resize(image_shape.at(1),
                                                                   image_shape.at(2),
                                                                   factor,
                                                                   config.min_pixels,
                                                                   config.max_pixels);
    const size_t grid_h = target_size.height / config.patch_size;
    const size_t grid_w = target_size.width / config.patch_size;
    const size_t channels = image_shape.at(3);

    int64_t resize_shape_data[2] = {static_cast<int64_t>(target_size.height), static_cast<int64_t>(target_size.width)};
    int64_t reshape_shape7d_data[7] = {static_cast<int64_t>(channels),
                                       static_cast<int64_t>(grid_h / config.merge_size),
                                       static_cast<int64_t>(config.merge_size),
                                       static_cast<int64_t>(config.patch_size),
                                       static_cast<int64_t>(grid_w / config.merge_size),
                                       static_cast<int64_t>(config.merge_size),
                                       static_cast<int64_t>(config.patch_size)};
    int64_t reshape_shape2d_data[2] = {static_cast<int64_t>(grid_h * grid_w),
                                       static_cast<int64_t>(channels * config.patch_size * config.patch_size)};
    auto [bilinear_indices, bilinear_weights] =
        paddleocr_vl_utils::get_bilinear_indices_and_weights(grid_h, grid_w, m_num_grid_per_side, config.merge_size);

    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    encoder.set_tensor("raw_images", image);
    encoder.set_tensor("resize_shape", ov::Tensor(ov::element::i64, {2}, resize_shape_data));
    encoder.set_tensor("reshape_shape7d", ov::Tensor(ov::element::i64, {7}, reshape_shape7d_data));
    encoder.set_tensor("reshape_shape2d", ov::Tensor(ov::element::i64, {2}, reshape_shape2d_data));
    encoder.set_tensor("bilinear_indices", bilinear_indices);
    encoder.set_tensor("bilinear_weights", bilinear_weights);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    EncodedImage encoded_image;
    encoded_image.resized_source = ov::Tensor(infer_output.get_element_type(), infer_output.get_shape());
    infer_output.copy_to(encoded_image.resized_source);
    encoded_image.resized_source_size = ImageSize{grid_h, grid_w};
    return encoded_image;
}

EncodedVideo VisionEncoderPaddleOCRVL::encode_frames(const std::vector<ov::Tensor>& frames) {
    OPENVINO_THROW("PaddleOCR-VL does not support 'video' input, please use 'images' instead.");
}

InputsEmbedderPaddleOCRVL::InputsEmbedderPaddleOCRVL(const VLMConfig& vlm_config,
                                                     const std::filesystem::path& model_dir,
                                                     const Tokenizer& tokenizer,
                                                     const std::string& device,
                                                     const ov::AnyMap device_config)
    : InputsEmbedderQwen2VL(vlm_config, model_dir, tokenizer, device, device_config) {
    validate_vision_token_ids();
}

InputsEmbedderPaddleOCRVL::InputsEmbedderPaddleOCRVL(const VLMConfig& vlm_config,
                                                     const ModelsMap& models_map,
                                                     const Tokenizer& tokenizer,
                                                     const std::filesystem::path& config_dir_path,
                                                     const std::string& device,
                                                     const ov::AnyMap device_config)
    : InputsEmbedderQwen2VL(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    validate_vision_token_ids();
}

void InputsEmbedderPaddleOCRVL::validate_vision_token_ids() const {
    // Vision embeddings are merged at the image placeholder token positions, so the tokenizer must keep the
    // placeholder as the single token that the model was trained with.
    if (m_vlm_config.image_token_id >= 0) {
        OPENVINO_ASSERT(m_vision_token_ids.at("image_pad") == m_vlm_config.image_token_id,
                        "Tokenizer maps '", m_vlm_config.image_pad_token, "' to ", m_vision_token_ids.at("image_pad"),
                        ", but config.json image_token_id is ", m_vlm_config.image_token_id);
    }
}

std::vector<ov::genai::EncodedVideo> InputsEmbedderPaddleOCRVL::encode_videos(
    const std::vector<ov::Tensor>& videos,
    const std::vector<VideoMetadata>& videos_metadata) {
    OPENVINO_ASSERT(videos.empty(), "PaddleOCR-VL does not support 'video' input, please use 'images' instead.");
    return {};
}

}  // namespace ov::genai
