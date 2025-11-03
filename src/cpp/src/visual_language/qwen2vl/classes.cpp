// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <iostream>
#include <iostream>
#include <numeric>

#include "visual_language/qwen2vl/classes.hpp"

#include "visual_language/clip.hpp"
#include "visual_language/embedding_model.hpp"

#include "utils.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/tile.hpp"

#include "visual_language/vl_sdpa_transformations.hpp"

namespace ov::genai {

namespace {

// Chat template hardcodes char sequence instead of referring to tag values, so NATIVE_TAG is hardcoded as well.
const std::string NATIVE_TAG = "<|vision_start|><|image_pad|><|vision_end|>";

std::shared_ptr<ov::Node> create_f32_nchw_input(std::shared_ptr<ov::Node> input) {
    auto raw_images_f32 = std::make_shared<ov::op::v0::Convert>(input, ov::element::f32);
    auto img_trans = std::make_shared<ov::op::v1::Transpose>(
        raw_images_f32,
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{4}, std::vector<int32_t>{0, 3, 1, 2}));
    return img_trans;
}

/**
 * Creates a bicubic resize operation using OpenVINO nodes
 * @param input The input tensor node to resize
 * @param target_size Node containing the target width and height [height, width]
 * @return Node representing the resized tensor
 */
std::shared_ptr<ov::Node> create_bicubic_resize(std::shared_ptr<ov::Node> input,
                                                const std::shared_ptr<ov::Node>& target_size) {
    // Create axes for height and width dimensions (assuming NCHW layout)
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});

    // Configure interpolation attributes for bicubic resize
    ov::op::v11::Interpolate::InterpolateAttrs attrs;
    attrs.mode = ov::op::v11::Interpolate::InterpolateMode::CUBIC;
    attrs.shape_calculation_mode = ov::op::v11::Interpolate::ShapeCalcMode::SIZES;
    attrs.coordinate_transformation_mode = ov::op::v11::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
    attrs.cube_coeff = -0.75f;  // Standard bicubic coefficient
    attrs.nearest_mode = ov::op::v11::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = {0, 0};
    attrs.pads_end = {0, 0};
    attrs.antialias = false;

    // Create interpolate operation
    auto interpolate = std::make_shared<ov::op::v11::Interpolate>(input, target_size, axes, attrs);

    return interpolate;
}

/**
 * Creates a normalization operation using OpenVINO nodes
 * @param input The input tensor node to normalize (uint8 format)
 * @param mean Node containing the mean values for each channel
 * @param std Node containing the standard deviation values for each channel
 * @return Node representing the normalized tensor
 */
std::shared_ptr<ov::Node> create_normalization(std::shared_ptr<ov::Node> input,
                                               const std::shared_ptr<ov::Node>& mean,
                                               const std::shared_ptr<ov::Node>& std) {
    // clamp to 0 ~ 255
    auto image_clamp = std::make_shared<ov::op::v0::Clamp>(input, 0, 255);
    // Subtract mean
    auto mean_subtracted = std::make_shared<ov::op::v1::Subtract>(image_clamp, mean);

    // Divide by std
    auto normalized = std::make_shared<ov::op::v1::Multiply>(mean_subtracted, std);

    return normalized;
}

/**
 * @brief Creates a node that reshapes and transposes the input tensor to match the
 *        functionality of reshape_image_patches.
 * @param input The input node to reshape and transpose.
 * @param reshape_shape A constant node containing the target shape dimensions.
 * @return A node representing the reshaped and transposed tensor.
 */
std::shared_ptr<ov::Node> create_transpose_patches(std::shared_ptr<ov::Node> input,
                                                   const std::shared_ptr<ov::Node>& reshape_dims,
                                                   const std::shared_ptr<ov::Node>& transpose_order) {
    // Reshape input to the required dimensions
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(input, reshape_dims, true);

    // Transpose the reshaped tensor
    auto transposed = std::make_shared<ov::op::v1::Transpose>(reshaped, transpose_order);

    return transposed;
}

std::shared_ptr<ov::Node> create_flatten_patches(std::shared_ptr<ov::Node> input,
                                                 const std::shared_ptr<ov::Node>& flatten_shape) {
    // Reshape (flatten) the input tensor
    auto flattened = std::make_shared<ov::op::v1::Reshape>(input, flatten_shape, true);

    return flattened;
}

std::shared_ptr<ov::Model> patch_preprocess_into_model(std::shared_ptr<ov::Model> model_org,
                                                       const ov::Tensor& image_mean_tensor,
                                                       const ov::Tensor& image_scale_tensor) {
    auto input_images = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{-1, -1, -1, -1});
    auto resize_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2});
    auto tile_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{4});
    auto reshape_shape8d = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{8});
    auto reshape_shape4d = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{4});
    auto reshape_shape2d = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2});

    input_images->set_friendly_name("input_images");
    input_images->output(0).get_tensor().set_names({"input_images"});

    resize_shape->set_friendly_name("resize_shape");
    resize_shape->output(0).get_tensor().set_names({"resize_shape"});

    tile_shape->set_friendly_name("tile_shape");
    tile_shape->output(0).get_tensor().set_names({"tile_shape"});

    reshape_shape8d->set_friendly_name("reshape_shape8d");
    reshape_shape8d->output(0).get_tensor().set_names({"reshape_shape8d"});
    reshape_shape4d->set_friendly_name("reshape_shape4d");
    reshape_shape4d->output(0).get_tensor().set_names({"reshape_shape4d"});
    reshape_shape2d->set_friendly_name("reshape_shape2d");
    reshape_shape2d->output(0).get_tensor().set_names({"reshape_shape2d"});
    auto image_mean = std::make_shared<ov::op::v0::Constant>(image_mean_tensor);
    auto image_scale = std::make_shared<ov::op::v0::Constant>(image_scale_tensor);
    auto img_f32_nchw = create_f32_nchw_input(input_images);

    auto img_resized = create_bicubic_resize(img_f32_nchw, resize_shape);

    auto img_normalized = create_normalization(img_resized, image_mean, image_scale);

    auto temporal_images = std::make_shared<ov::op::v0::Tile>(img_normalized, tile_shape);

    auto img_8d =
        create_transpose_patches(temporal_images,
                                 reshape_shape8d,
                                 std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                                        Shape{8},
                                                                        std::vector<int32_t>{0, 2, 5, 3, 6, 1, 4, 7}));

    auto img_4d = create_transpose_patches(
        img_8d,
        reshape_shape4d,
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{4}, std::vector<int32_t>{0, 2, 1, 3}));

    auto img_2d = create_flatten_patches(img_4d, reshape_shape2d);

    auto params_org = model_org->get_parameters();
    OPENVINO_ASSERT(params_org.size() == 1);

    ov::replace_node(params_org[0], img_2d);

    auto results = model_org->get_results();

    return std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{input_images, resize_shape, tile_shape, reshape_shape8d, reshape_shape4d, reshape_shape2d});
}
} // namespace

namespace qwen2_vl_utils {

ImageSize smart_resize(size_t height, size_t width, size_t factor, size_t min_pixels, size_t max_pixels) {
    if (height < factor || width < factor) {
        OPENVINO_THROW("Height (" + std::to_string(height) + ") and width (" + std::to_string(width) + ") must be greater than factor (" + std::to_string(factor) + ")");
    }
    if (std::max(height, width) / std::min(height, width) > 200) {
        OPENVINO_THROW("Absolute aspect ratio must be smaller than 200");
    }

    size_t h_bar = std::round(static_cast<float>(height) / factor) * factor;
    size_t w_bar = std::round(static_cast<float>(width) / factor) * factor; 

    if (h_bar * w_bar > max_pixels) {
        double beta = std::sqrt((height * width) / static_cast<double>(max_pixels));
        h_bar = std::floor(height / beta / factor) * factor;
        w_bar = std::floor(width / beta / factor) * factor;
    } else if (h_bar * w_bar < min_pixels) {
        double beta = std::sqrt(min_pixels / static_cast<double>(height * width));
        h_bar = std::ceil(height * beta / factor) * factor;
        w_bar = std::ceil(width * beta / factor) * factor;
    }
    
    return ImageSize{h_bar, w_bar};
}

ov::Tensor reshape_image_patches(
    const ov::Tensor& patches,
    const size_t grid_t,
    const size_t grid_h,
    const size_t grid_w,
    const size_t channel,
    const size_t temporal_patch_size,
    const size_t patch_size,
    const size_t spatial_merge_size
) {
    ov::Shape output_shape{
        grid_t,                      
        temporal_patch_size,         
        channel,                     
        grid_h / spatial_merge_size, 
        spatial_merge_size,          
        patch_size,                  
        grid_w / spatial_merge_size, 
        spatial_merge_size,          
        patch_size                   
    };
    
    ov::Tensor reshaped_patches(patches.get_element_type(), output_shape);

    const float* input_data = patches.data<float>();
    float* output_data = reshaped_patches.data<float>();

    size_t input_idx = 0;
    
    for (size_t gt = 0; gt < output_shape.at(0); ++gt) {
        for (size_t tp = 0; tp < output_shape.at(1); ++tp) {
            for (size_t c = 0; c < output_shape.at(2); ++c) {
                for (size_t gh = 0; gh < output_shape.at(3); ++gh) {
                    for (size_t ms1 = 0; ms1 < output_shape.at(4); ++ms1) {
                        for (size_t p1 = 0; p1 < output_shape.at(5); ++p1) {
                            for (size_t gw = 0; gw < output_shape.at(6); ++gw) {
                                for (size_t ms2 = 0; ms2 < output_shape.at(7); ++ms2) {
                                    for (size_t p2 = 0; p2 < output_shape.at(8); ++p2) {
                                        size_t output_idx = gt;
                                        output_idx = output_idx * output_shape.at(1) + tp;
                                        output_idx = output_idx * output_shape.at(2) + c;
                                        output_idx = output_idx * output_shape.at(3) + gh;
                                        output_idx = output_idx * output_shape.at(4) + ms1;
                                        output_idx = output_idx * output_shape.at(5) + p1;
                                        output_idx = output_idx * output_shape.at(6) + gw;
                                        output_idx = output_idx * output_shape.at(7) + ms2;
                                        output_idx = output_idx * output_shape.at(8) + p2;

                                        output_data[output_idx] = input_data[input_idx];
                                        input_idx++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return reshaped_patches;
}
    
ov::Tensor transpose_image_patches(const ov::Tensor& reshaped_patches) {
    // Input dimensions order:  [0,1,2,3,4,5,6,7,8]
    // Output dimensions order: [0,3,6,4,7,2,1,5,8]
    auto input_shape = reshaped_patches.get_shape();
    
    ov::Shape output_shape = {
        input_shape.at(0), // grid_t
        input_shape.at(3), // grid_h / spatial_merge_size
        input_shape.at(6), // grid_w / spatial_merge_size
        input_shape.at(4), // spatial_merge_size
        input_shape.at(7), // spatial_merge_size
        input_shape.at(2), // channel
        input_shape.at(1), // temporal_patch_size
        input_shape.at(5), // patch_size
        input_shape.at(8)  // patch_size
    };

    ov::Tensor transposed_patches(reshaped_patches.get_element_type(), output_shape);
    
    const float* src = reshaped_patches.data<float>();
    float* dst = transposed_patches.data<float>();
    
    size_t shape_size = input_shape.size();
    std::vector<size_t> input_strides(shape_size);
    std::vector<size_t> output_strides(shape_size);
    
    input_strides[shape_size - 1] = 1;
    output_strides[shape_size - 1] = 1;
    for(int i = 7; i >= 0; i--) {
        input_strides[i] = input_strides[i+1] * input_shape[i+1];
        output_strides[i] = output_strides[i+1] * output_shape[i+1];
    }

    size_t total_elements = reshaped_patches.get_size();
    for(size_t idx = 0; idx < total_elements; idx++) {
        size_t remaining = idx;
        std::vector<size_t> input_indices(shape_size);
        for(int i = 0; i < shape_size; i++) {
            input_indices[i] = remaining / input_strides[i];
            remaining %= input_strides[i];
        }
        
        std::vector<size_t> output_indices = {
            input_indices.at(0),
            input_indices.at(3),
            input_indices.at(6),
            input_indices.at(4),
            input_indices.at(7),
            input_indices.at(2),
            input_indices.at(1),
            input_indices.at(5),
            input_indices.at(8)
        };
        
        size_t dst_idx = 0;
        for(int i = 0; i < shape_size; i++) {
            dst_idx += output_indices[i] * output_strides[i];
        }
        
        dst[dst_idx] = src[idx];
    }
    
    return transposed_patches;
}

std::pair<std::vector<ov::Tensor>, std::vector<std::array<size_t, 3>>> reorder_image_embeds_and_grid_thw(
    const std::vector<EncodedImage>& encoded_images,
    const std::vector<size_t>& images_sequence
) {
    std::vector<ov::Tensor> image_embeds;
    std::vector<std::array<size_t, 3>> images_grid_thw;
    image_embeds.reserve(encoded_images.size());
    images_grid_thw.reserve(encoded_images.size());
    
    for (const auto& encoded_image : encoded_images) {
        ov::Tensor single_image_embeds = encoded_image.resized_source;
        image_embeds.push_back(std::move(single_image_embeds));

        size_t grid_t = 1;
        size_t grid_h = encoded_image.resized_source_size.height;
        size_t grid_w = encoded_image.resized_source_size.width;
        images_grid_thw.push_back({grid_t, grid_h, grid_w});
    }

    std::vector<ov::Tensor> reordered_image_embeds;
    std::vector<std::array<size_t, 3>> reordered_images_grid_thw;
    for (size_t new_image_id : images_sequence) {
        reordered_image_embeds.push_back(image_embeds.at(new_image_id));
        reordered_images_grid_thw.push_back(images_grid_thw.at(new_image_id));
    }

    return {reordered_image_embeds, reordered_images_grid_thw};
}
    
ov::Tensor get_attention_mask(const std::vector<std::array<size_t, 3>>& reordered_images_grid_thw) {
    // Calculate cumulative sequence lengths for attention mask
    std::vector<int32_t> cu_seqlens;
    cu_seqlens.push_back(0);
    int32_t cumsum = 0;
    for (const auto& grid_thw : reordered_images_grid_thw) {
        size_t slice_len = grid_thw.at(1) * grid_thw.at(2);
        for (size_t t = 0; t < grid_thw.at(0); ++t) {
            cumsum += slice_len;
            cu_seqlens.push_back(cumsum);
        }
    }

    // Create attention mask for vision embeddings merger model
    size_t hidden_states_size = cumsum;
    ov::Tensor attention_mask{ov::element::f32, {1, hidden_states_size, hidden_states_size}};
    float* attention_mask_data = attention_mask.data<float>();
    std::fill_n(attention_mask_data, attention_mask.get_size(), -std::numeric_limits<float>::infinity());

    for (size_t i = 1; i < cu_seqlens.size(); ++i) {
        size_t start = cu_seqlens[i-1];
        size_t end = cu_seqlens[i];
        for (size_t row = start; row < end; ++row) {
            for (size_t col = start; col < end; ++col) {
                attention_mask_data[row * hidden_states_size + col] = 0.0f;
            }
        }
    }
    return attention_mask;
}

ov::Tensor get_cu_seqlens(const std::vector<std::array<size_t, 3>>& reordered_images_grid_thw) {
    // Calculate cumulative sequence lengths for attention mask
    std::vector<int32_t> cu_seqlens;
    cu_seqlens.push_back(0);
    int32_t cumsum = 0;
    for (const auto& grid_thw : reordered_images_grid_thw) {
        size_t slice_len = grid_thw.at(1) * grid_thw.at(2);
        for (size_t t = 0; t < grid_thw.at(0); ++t) {
            cumsum += slice_len;
            cu_seqlens.push_back(cumsum);
        }
    }

    ov::Tensor t_cu_seqlens(ov::element::i32, {cu_seqlens.size()});
    std::memcpy(t_cu_seqlens.data<int32_t>(), cu_seqlens.data(), cu_seqlens.size() * sizeof(int32_t));
    return t_cu_seqlens;
}

ov::Tensor concatenate_image_embeds(const std::vector<ov::Tensor>& reordered_image_embeds) {
    ov::Tensor concatenated_embeds;
    if (reordered_image_embeds.size() == 1) {
        concatenated_embeds = reordered_image_embeds.at(0);
    } else {
        size_t total_length = 0;
        for (const auto& embed : reordered_image_embeds) {
            total_length += embed.get_shape().at(0);
        }
        size_t hidden_dim = reordered_image_embeds.at(0).get_shape().at(1);
        
        concatenated_embeds = ov::Tensor(reordered_image_embeds.at(0).get_element_type(), {total_length, hidden_dim});
        float* concat_data = concatenated_embeds.data<float>();
        
        size_t offset = 0;
        for (const auto& embed : reordered_image_embeds) {
            size_t embed_size = embed.get_shape().at(0) * embed.get_shape().at(1);
            std::memcpy(concat_data + offset, embed.data(), embed.get_byte_size());
            offset += embed_size;
        }
    }
    return concatenated_embeds;
}

ov::Tensor merge_text_and_image_embeddings(
    const ov::Tensor& input_ids,
    const ov::Tensor& text_embeds, 
    const ov::Tensor& processed_vision_embeds,
    const int64_t image_pad_token_id
) {
    ov::Tensor merged_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
    std::memcpy(merged_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());

    auto text_embeds_shape = text_embeds.get_shape();
    size_t batch_size = text_embeds_shape.at(0);
    size_t seq_length = text_embeds_shape.at(1);
    size_t hidden_size = text_embeds_shape.at(2);

    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    float* merged_embeds_data = merged_embeds.data<float>();
    const float* vision_embeds_data = processed_vision_embeds.data<const float>();

    size_t vision_embed_idx = 0;
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (size_t seq_idx = 0; seq_idx < seq_length; ++seq_idx) {
            size_t flat_idx = batch_idx * seq_length + seq_idx;
            if (input_ids_data[flat_idx] == image_pad_token_id) {
                std::copy_n(
                    vision_embeds_data + vision_embed_idx * hidden_size,
                    hidden_size,
                    merged_embeds_data + flat_idx * hidden_size
                );
                ++vision_embed_idx;
            }
        }
    }
    return merged_embeds;
}
    
} // namespace qwen2vl_utils

std::unique_ptr<CircularBufferQueue<ov::InferRequest>> create_vision_encoder_ireq(
    const std::shared_ptr<ov::Model>& model_org,
    const ProcessorConfig& processor_config,
    const std::string& device,
    const ov::AnyMap& config) {
    std::vector<float> a_image_mean(processor_config.image_mean.begin(), processor_config.image_mean.end());
    std::vector<float> a_image_scale(processor_config.image_std.begin(), processor_config.image_std.end());
    for (auto& v : a_image_mean)
        v *= 255.0f;
    for (auto& v : a_image_scale)
        v = 1.0f / (v * 255.0f);

    ov::Tensor image_mean(ov::element::f32, {1, a_image_mean.size(), 1, 1}, a_image_mean.data());
    ov::Tensor image_scale(ov::element::f32, {1, a_image_scale.size(), 1, 1}, a_image_scale.data());

    auto model = patch_preprocess_into_model(model_org, image_mean, image_scale);
    auto compiled_model = utils::singleton_core().compile_model(model, device, config);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM vision embeddings model");
    return std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
}

bool check_image_preprocess_env() {
    const char* env = std::getenv("IMAGE_PREPROCESS");
    return !(env && std::string(env) == "CPP");
}

VisionEncoderQwen2VL::VisionEncoderQwen2VL(const std::filesystem::path& model_dir,
                                           const std::string& device,
                                           const ov::AnyMap properties)
    : VisionEncoder(model_dir, device, properties),
      use_ov_image_preprocess(check_image_preprocess_env()) {
    if (use_ov_image_preprocess) {
        auto model_org = utils::singleton_core().read_model(model_dir / "openvino_vision_embeddings_model.xml");
        m_ireq_queue_vision_encoder = create_vision_encoder_ireq(model_org, m_processor_config, device, properties);
    }
}

VisionEncoderQwen2VL::VisionEncoderQwen2VL(const ModelsMap& models_map,
                                           const std::filesystem::path& config_dir_path,
                                           const std::string& device,
                                           const ov::AnyMap properties)
    : VisionEncoder(models_map, config_dir_path, device, properties),
      use_ov_image_preprocess(check_image_preprocess_env()) {
    if (use_ov_image_preprocess) {
        const auto& [vision_encoder_model, vision_encoder_weights] =
            utils::get_model_weights_pair(models_map, "vision_embeddings");
        auto model_org = utils::singleton_core().read_model(vision_encoder_model, vision_encoder_weights);
        m_ireq_queue_vision_encoder = create_vision_encoder_ireq(model_org, m_processor_config, device, properties);
    }
}

// keep both implementations for comparison and testing, here is the cpp version
EncodedImage VisionEncoderQwen2VL::encode_with_imagepreprocess_cpp(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);
    ov::Shape image_shape = image.get_shape();
    auto original_height = image_shape.at(1);
    auto original_width = image_shape.at(2);
    ImageSize target_image_size = qwen2_vl_utils::smart_resize(
        original_height, 
        original_width, 
        config.patch_size * config.merge_size,
        config.min_pixels,
        config.max_pixels
    );

    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    clip_image_u8 resized_image;
    bicubic_resize(input_image, resized_image, target_image_size.width, target_image_size.height);

    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);
    clip_image_f32 normalized_image = clip_image_preprocess(ctx, resized_image);

    ov::Tensor patches = clip_image_f32_to_tensor(normalized_image);

    // For single patch tile it to match temporal_patch_size
    if (patches.get_shape().at(0) == 1) {
        auto orig_shape = patches.get_shape();
        ov::Tensor tiled_patches(patches.get_element_type(),
                                    {config.temporal_patch_size, orig_shape.at(1), orig_shape.at(2), orig_shape.at(3)});

        for (size_t i = 0; i < config.temporal_patch_size; i++) {
            std::memcpy(
                tiled_patches.data<float>() + i * patches.get_byte_size() / sizeof(float),
                patches.data<float>(),
                patches.get_byte_size()
            );
        }
        patches = std::move(tiled_patches);
    }

    auto patches_shape = patches.get_shape();
    size_t channel = patches_shape.at(1);

    size_t grid_t = patches_shape.at(0) / config.temporal_patch_size;
    size_t grid_h = target_image_size.height / config.patch_size;
    size_t grid_w = target_image_size.width / config.patch_size;

    ov::Tensor reshaped_patches = qwen2_vl_utils::reshape_image_patches(
        patches, grid_t, grid_h, grid_w, channel, config.temporal_patch_size, config.patch_size, config.merge_size
    );
    ov::Tensor transposed_patches = qwen2_vl_utils::transpose_image_patches(reshaped_patches);

    ov::Shape flattened_patches_shape{
        grid_t * grid_h * grid_w,
        channel * config.temporal_patch_size * config.patch_size * config.patch_size
    };
    ov::Tensor flattened_patches(transposed_patches.get_element_type(), flattened_patches_shape);
    std::memcpy(flattened_patches.data(), transposed_patches.data(), transposed_patches.get_byte_size());

    encoder.set_tensor("hidden_states", flattened_patches);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());
    ImageSize resized_source_size{grid_h, grid_w};
    return {std::move(image_features), resized_source_size};
}

// keep both implementations for comparison and testing, here is the ov version
EncodedImage VisionEncoderQwen2VL::encode_with_imagepreprocess_ov(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    ov::Shape image_shape = image.get_shape();
    auto original_height = image_shape.at(1);
    auto original_width = image_shape.at(2);

    ImageSize target_image_size = qwen2_vl_utils::smart_resize(
        original_height, 
        original_width, 
        config.patch_size * config.merge_size,
        config.min_pixels,
        config.max_pixels
    );

    ov::Tensor input_images(ov::element::u8, image_shape, image.data<uint8_t>());

    uint64_t a_target_shape[2] = {target_image_size.height, target_image_size.width};
    ov::Tensor target_shape(ov::element::i64, ov::Shape{2}, a_target_shape);

    auto patches_shape = image.get_shape();
    size_t temporal_patch_size = std::max(static_cast<size_t>(patches_shape.at(0)), static_cast<size_t>(config.temporal_patch_size));
    size_t channel = image_shape.at(3);

    size_t grid_t = temporal_patch_size / config.temporal_patch_size;
    size_t grid_h = target_image_size.height / config.patch_size;
    size_t grid_w = target_image_size.width / config.patch_size;

    size_t repeats = 1;
    if (patches_shape.at(0) == 1) {
        repeats = config.temporal_patch_size;
    }
    uint64_t a_broadcast_shape[4] = {static_cast<size_t>(repeats), 1, 1, 1};

    uint64_t a_temp_shape8d[8] = {
        grid_t, temporal_patch_size * channel, grid_h / config.merge_size, config.merge_size, config.patch_size, grid_w / config.merge_size, config.merge_size, config.patch_size
    };
    uint64_t a_temp_shape4d[4] = {
        grid_t * (grid_h / config.merge_size) * (grid_w / config.merge_size) * (config.merge_size * config.merge_size),
        temporal_patch_size,
        channel,
        config.patch_size * config.patch_size
    };
    uint64_t last_output_shape[2] = {grid_t * grid_h * grid_w, channel * temporal_patch_size * config.patch_size * config.patch_size};
    ov::Tensor tile_shape(ov::element::i64, ov::Shape{4}, a_broadcast_shape);
    ov::Tensor reshape_shape8d(ov::element::i64, ov::Shape{8}, a_temp_shape8d);
    ov::Tensor reshape_shape4d(ov::element::i64, ov::Shape{4}, a_temp_shape4d);
    ov::Tensor reshape_shape2d(ov::element::i64, ov::Shape{2}, last_output_shape);

    encoder.set_tensor("input_images", input_images);
    encoder.set_tensor("resize_shape", target_shape);
    encoder.set_tensor("tile_shape", tile_shape);
    encoder.set_tensor("reshape_shape8d", reshape_shape8d);
    encoder.set_tensor("reshape_shape4d", reshape_shape4d);
    encoder.set_tensor("reshape_shape2d", reshape_shape2d);

    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    ImageSize resized_source_size{grid_h, grid_w};

    return {std::move(image_features), resized_source_size};
}

EncodedImage VisionEncoderQwen2VL::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    if (use_ov_image_preprocess == false) {
        return encode_with_imagepreprocess_cpp(image, config_map);
    }
    return encode_with_imagepreprocess_ov(image, config_map);
}

InputsEmbedderQwen2VL::InputsEmbedderQwen2VL(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, device, device_config) {
    auto model = utils::singleton_core().read_model(model_dir / "openvino_vision_embeddings_merger_model.xml");
    utils::request_vl_sdpa_transformations(model);

    auto compiled_model = utils::singleton_core().compile_model(model, device, device_config);

    m_with_cu_seqlens_input = utils::check_vl_sdpa_transformations(compiled_model);
    ov::genai::utils::print_compiled_model_properties(compiled_model,
        m_with_cu_seqlens_input ? "VLM vision embeddings merger model with VLSDPA optimization ENABLED" :
        "VLM vision embeddings merger model with VLSDPA optimization DISABLED");

    m_ireq_queue_vision_embeddings_merger = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
}

InputsEmbedderQwen2VL::InputsEmbedderQwen2VL(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer, 
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    auto model = utils::singleton_core().read_model(
        utils::get_model_weights_pair(models_map, "vision_embeddings_merger").first,
        utils::get_model_weights_pair(models_map, "vision_embeddings_merger").second);
    utils::request_vl_sdpa_transformations(model);

    auto compiled_model = utils::singleton_core().compile_model(model,
        device,
        device_config
    );

    m_with_cu_seqlens_input = utils::check_vl_sdpa_transformations(compiled_model);
    ov::genai::utils::print_compiled_model_properties(compiled_model,
        m_with_cu_seqlens_input ? "VLM vision embeddings merger model with VLSDPA optimization ENABLED" :
        "VLM vision embeddings merger model with VLSDPA optimization DISABLED");

    m_ireq_queue_vision_embeddings_merger = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
}

NormlizedPrompt InputsEmbedderQwen2VL::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    auto [unified_prompt, images_sequence] = normalize(prompt, NATIVE_TAG, NATIVE_TAG, base_id, images.size());
        std::vector<std::array<size_t, 3>> images_grid_thw;
    images_grid_thw.reserve(images.size());
    
    for (const auto& encoded_image : images) {
        size_t grid_t = 1;
        size_t grid_h = encoded_image.resized_source_size.height;
        size_t grid_w = encoded_image.resized_source_size.width;
        images_grid_thw.push_back({grid_t, grid_h, grid_w});
    }

    for (size_t new_image_id : images_sequence) {
        auto [grid_t, grid_h, grid_w] = images_grid_thw.at(new_image_id - base_id);
        size_t merge_length = std::pow(m_vision_encoder->get_processor_config().merge_size, 2);
        size_t num_image_pad_tokens = grid_t * grid_h * grid_w / merge_length;

        std::string expanded_tag = m_vlm_config.vision_start_token;
        for (int i = 0; i < num_image_pad_tokens; i++) {
            expanded_tag += m_vlm_config.image_pad_token;
        }
        expanded_tag += m_vlm_config.vision_end_token;
        unified_prompt.replace(unified_prompt.find(NATIVE_TAG), NATIVE_TAG.length(), expanded_tag);
    }
    return {std::move(unified_prompt), std::move(images_sequence)};
}

ov::Tensor InputsEmbedderQwen2VL::get_inputs_embeds(const std::string& unified_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
    std::vector<std::array<size_t, 3>> images_grid_thw;
    images_grid_thw.reserve(images.size());
    for (const auto& encoded_image : images) {
        size_t grid_t = 1;
        size_t grid_h = encoded_image.resized_source_size.height;
        size_t grid_w = encoded_image.resized_source_size.width;
        images_grid_thw.push_back({grid_t, grid_h, grid_w});
    }

    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    ov::Tensor text_embeds;
    {
        // Acquire request, run inference, then copy the result to safeguard against later reuse
        CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
        EmbeddingsRequest& req = embeddings_request_guard.get();
        ov::Tensor tmp_embeds = m_embedding->infer(req, input_ids);

        // Deep-copy to ensure the data remains valid after the request is released
        text_embeds = ov::Tensor(tmp_embeds.get_element_type(), tmp_embeds.get_shape());
        std::memcpy(text_embeds.data(), tmp_embeds.data(), tmp_embeds.get_byte_size());
    } // Request released here

    auto start_tokenizer_time = std::chrono::steady_clock::now();
    ov::Tensor encoded_vision_start_token = m_tokenizer.encode(m_vlm_config.vision_start_token, ov::genai::add_special_tokens(false)).input_ids;
    ov::Tensor encoded_image_pad_token = m_tokenizer.encode(m_vlm_config.image_pad_token, ov::genai::add_special_tokens(false)).input_ids;
    ov::Tensor encoded_vision_end_token = m_tokenizer.encode(m_vlm_config.vision_end_token, ov::genai::add_special_tokens(false)).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    int64_t vision_start_token_id = encoded_vision_start_token.data<int64_t>()[encoded_vision_start_token.get_size() - 1];
    int64_t image_pad_token_id = encoded_image_pad_token.data<int64_t>()[encoded_image_pad_token.get_size() - 1];
    int64_t vision_end_token_id = encoded_vision_end_token.data<int64_t>()[encoded_vision_end_token.get_size() - 1];

    m_position_ids = create_position_ids(input_ids, images_grid_thw, images_sequence, 0, vision_start_token_id);

    int64_t position_ids_max_element = *std::max_element(m_position_ids.data<int64_t>(), m_position_ids.data<int64_t>() + m_position_ids.get_size());
    m_rope_delta = position_ids_max_element + 1 - static_cast<int64_t>(input_ids.get_shape().at(1));

    if (images.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }
    ov::Tensor merged_image_embeddings_tensor;
    if (recalculate_merged_embeddings) {
        m_merged_image_embeddings = run_image_embeddings_merger(images, images_sequence);
    }
    merged_image_embeddings_tensor = m_merged_image_embeddings;

    // [CDPruner] Apply token pruning if CDPruner is enabled and images are present
    size_t original_visual_tokens = 0;
    size_t pruned_visual_tokens = 0;
    std::vector<size_t> original_tokens_per_region;
    std::vector<size_t> pruned_tokens_per_region;
    std::vector<std::vector<bool>> keep_flags_per_region;

    auto current_pruning_config = m_vision_encoder->get_pruning_config();
    bool pruner_enabled = current_pruning_config.has_value() && current_pruning_config->pruning_ratio > 0;

    if (m_vision_encoder->is_pruning_available() && pruner_enabled && !images.empty()) {
        // Store original visual token count for position adjustment
        original_visual_tokens = merged_image_embeddings_tensor.get_shape()[0];

        // Extract text features for CDPruner using the implemented function
        ov::Tensor text_features = extract_text_features_for_cdpruner(input_ids,
                                                                      image_pad_token_id,
                                                                      vision_start_token_id,
                                                                      vision_end_token_id);

        // Convert visual features for CDPruner using the implemented function
        // [CDPruner] Check enable_frame_chunking to decide chunking strategy
        size_t chunk_count = current_pruning_config->enable_frame_chunking ? images.size() : 1;
        auto visual_features = convert_visual_features_for_cdpruner(merged_image_embeddings_tensor, chunk_count);

        // Apply CDPruner to get pruned visual tokens
        ov::Tensor pruned_visual_features = m_vision_encoder->apply_pruning(visual_features, text_features);
        
        // [CDPruner] Convert back from 3D [1, num_tokens, hidden_size] to 2D [num_tokens, hidden_size]
        // to match the expected input format for merge_text_and_image_embeddings
        ov::Shape pruned_shape = pruned_visual_features.get_shape();
        pruned_visual_tokens = pruned_shape[1];  // num_tokens dimension
        size_t hidden_size = pruned_shape[2];    // hidden_size dimension

        // Create 2D tensor with shape [num_tokens, hidden_size]
        ov::Tensor pruned_2d_tensor(pruned_visual_features.get_element_type(), {pruned_visual_tokens, hidden_size});

        // Copy data from 3D [1, num_tokens, hidden_size] to 2D [num_tokens, hidden_size]
        const float* src_data = pruned_visual_features.data<const float>();
        float* dst_data = pruned_2d_tensor.data<float>();
        size_t total_elements = pruned_visual_tokens * hidden_size;
        std::memcpy(dst_data, src_data, total_elements * sizeof(float));

        merged_image_embeddings_tensor = pruned_2d_tensor;

        // Adjust position_ids if pruning occurred
        if (original_visual_tokens != pruned_visual_tokens) {
            auto kept_indices_per_image = m_vision_encoder->get_last_selected_token_indices();
            m_position_ids = adjust_position_ids_for_pruning(m_position_ids,
                                                             input_ids,
                                                             original_visual_tokens,
                                                             pruned_visual_tokens,
                                                             vision_start_token_id,
                                                             image_pad_token_id,
                                                             images_grid_thw,
                                                             images_sequence,
                                                             0,
                                                             kept_indices_per_image,
                                                             &original_tokens_per_region,
                                                             &pruned_tokens_per_region,
                                                             &keep_flags_per_region);

            size_t total_original_tokens =
                std::accumulate(original_tokens_per_region.begin(), original_tokens_per_region.end(), size_t{0});
            size_t total_pruned_tokens =
                std::accumulate(pruned_tokens_per_region.begin(), pruned_tokens_per_region.end(), size_t{0});
            OPENVINO_ASSERT(total_original_tokens == original_visual_tokens,
                            "Original visual token metadata mismatch after pruning");
            OPENVINO_ASSERT(total_pruned_tokens == pruned_visual_tokens,
                            "Pruned visual token metadata mismatch after pruning");
            OPENVINO_ASSERT(keep_flags_per_region.size() == original_tokens_per_region.size(),
                            "Kept visual token mask count mismatch with vision regions");

            const int64_t* pos_data_begin = m_position_ids.data<const int64_t>();
            int64_t position_ids_max_element =
                *std::max_element(pos_data_begin, pos_data_begin + m_position_ids.get_size());

            size_t new_sequence_length = m_position_ids.get_shape().at(2);
            m_rope_delta = position_ids_max_element + 1 - static_cast<int64_t>(new_sequence_length);
        }
    }

    // [CDPruner] Handle pruned visual tokens case
    if (m_vision_encoder->is_pruning_available() && pruner_enabled && !images.empty() &&
        original_visual_tokens != pruned_visual_tokens) {
        size_t tokens_removed = original_visual_tokens - pruned_visual_tokens;
        size_t new_sequence_length = input_ids.get_shape().at(1) - tokens_removed;
        ov::Tensor pruned_input_ids(ov::element::i64, {1, new_sequence_length});
        const int64_t* input_data = input_ids.data<const int64_t>();
        int64_t* pruned_data = pruned_input_ids.data<int64_t>();
        const size_t seq_len = input_ids.get_shape().at(1);
        const size_t region_count = original_tokens_per_region.size();
        OPENVINO_ASSERT(region_count == keep_flags_per_region.size(),
                        "Vision region metadata not available for pruning");

        size_t computed_pruned_tokens = 0;
        for (const auto& mask : keep_flags_per_region) {
            computed_pruned_tokens += static_cast<size_t>(std::count(mask.begin(), mask.end(), true));
        }
        OPENVINO_ASSERT(computed_pruned_tokens == pruned_visual_tokens,
                        "Kept visual token mask total mismatch with pruned embeddings");

        size_t write_idx = 0;
        bool inside_vision_region = false;
        size_t region_idx = 0;
        size_t pad_index = 0;

        for (size_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
            int64_t token_id = input_data[seq_idx];

            if (token_id == vision_start_token_id) {
                OPENVINO_ASSERT(region_idx < region_count,
                                "Encountered more vision regions than metadata entries while pruning input ids");
                inside_vision_region = true;
                pad_index = 0;
            }

            if (inside_vision_region && token_id == image_pad_token_id) {
                OPENVINO_ASSERT(region_idx < region_count,
                                "Vision region index exceeds metadata size while pruning input ids");
                const auto& keep_mask = keep_flags_per_region.at(region_idx);
                OPENVINO_ASSERT(pad_index < keep_mask.size(),
                                "Visual token index exceeds region token count while pruning input ids");
                if (keep_mask[pad_index]) {
                    OPENVINO_ASSERT(write_idx < new_sequence_length,
                                    "Pruned input ids index exceeds expected sequence length");
                    pruned_data[write_idx++] = token_id;
                }
                ++pad_index;
                continue;
            }

            OPENVINO_ASSERT(write_idx < new_sequence_length,
                            "Pruned input ids index exceeds expected sequence length");
            pruned_data[write_idx++] = token_id;

            if (inside_vision_region && token_id == vision_end_token_id) {
                const auto& keep_mask = keep_flags_per_region.at(region_idx);
                OPENVINO_ASSERT(pad_index == keep_mask.size(),
                                "Mismatch between consumed visual tokens and region metadata while pruning input ids");
                inside_vision_region = false;
                ++region_idx;
            }
        }

        OPENVINO_ASSERT(!inside_vision_region,
                        "Unexpected end of sequence inside a vision region while pruning input ids");
        OPENVINO_ASSERT(region_idx == region_count,
                        "Not all vision regions processed while generating pruned input ids");
        OPENVINO_ASSERT(write_idx == new_sequence_length,
                        "Pruned input ids length mismatch after visual token pruning");

        auto& kv_history = m_kv_cache_state.get_state();
        OPENVINO_ASSERT(kv_history.size() >= input_ids.get_size(),
                        "KV cache history does not contain expected original prompt length");
        OPENVINO_ASSERT(kv_history.size() >= m_prev_hist_length,
                        "KV cache history is shorter than recorded previous history length");
        kv_history.resize(m_prev_hist_length);
        m_kv_cache_state.add_inputs(pruned_input_ids);
        // Visual tokens have been pruned, need to create new merged embeddings with correct dimensions
        return merge_text_and_image_embeddings_with_pruning(input_ids,
                                                            text_embeds,
                                                            merged_image_embeddings_tensor,
                                                            image_pad_token_id,
                                                            vision_start_token_id,
                                                            vision_end_token_id,
                                                            original_tokens_per_region,
                                                            keep_flags_per_region);
    } else {
        // No pruning or no images, use original function
        return qwen2_vl_utils::merge_text_and_image_embeddings(input_ids,
                                                               text_embeds,
                                                               merged_image_embeddings_tensor,
                                                               image_pad_token_id);
    }
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderQwen2VL::get_position_ids(const size_t inputs_embeds_size, const size_t history_size) {
    if (history_size != 0) {
        ov::Tensor position_ids{ov::element::i64, {3, 1, inputs_embeds_size}};
        int64_t new_pos_id = static_cast<int64_t>(history_size + m_rope_delta);
        for (size_t dim = 0; dim < 3; ++dim) {
            int64_t* pos_data = position_ids.data<int64_t>() + dim * inputs_embeds_size;
            std::iota(pos_data, pos_data + inputs_embeds_size, new_pos_id);
        }
        return {position_ids, m_rope_delta};
    }
    return {m_position_ids, m_rope_delta};
}

void InputsEmbedderQwen2VL::start_chat(const std::string& system_message) {
    IInputsEmbedder::start_chat(system_message);
    m_position_ids = ov::Tensor();
    m_rope_delta = 0;
}

void InputsEmbedderQwen2VL::finish_chat() {
    IInputsEmbedder::finish_chat();
    m_position_ids = ov::Tensor();
    m_rope_delta = 0;
    m_merged_image_embeddings = ov::Tensor();
    // [CDPruner] CDPruner is stateless, no cleanup needed
}

ov::Tensor InputsEmbedderQwen2VL::run_image_embeddings_merger(
    const std::vector<EncodedImage>& images,
    const std::vector<size_t>& images_sequence
) {
    auto [reordered_image_embeds, reordered_images_grid_thw] = qwen2_vl_utils::reorder_image_embeds_and_grid_thw(images, images_sequence);

    ov::Tensor concatenated_embeds = qwen2_vl_utils::concatenate_image_embeds(reordered_image_embeds);
    ov::Tensor rotary_pos_emb = get_rotary_pos_emb(reordered_images_grid_thw);

    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_embeddings_merger.get());
    ov::InferRequest& vision_embeddings_merger = infer_request_guard.get();
    vision_embeddings_merger.set_tensor("hidden_states", concatenated_embeds);
    if (m_with_cu_seqlens_input) {
        ov::Tensor cu_seq_lens = qwen2_vl_utils::get_cu_seqlens(reordered_images_grid_thw);
        vision_embeddings_merger.set_tensor("cu_seq_lens", cu_seq_lens);
    } else {
        ov::Tensor attention_mask = qwen2_vl_utils::get_attention_mask(reordered_images_grid_thw);
        vision_embeddings_merger.set_tensor("attention_mask", attention_mask);
    }
    vision_embeddings_merger.set_tensor("rotary_pos_emb", rotary_pos_emb);
    vision_embeddings_merger.infer();
    ov::Tensor processed_vision_embeds = vision_embeddings_merger.get_output_tensor();

    ov::Tensor res = ov::Tensor(processed_vision_embeds.get_element_type(), processed_vision_embeds.get_shape());
    std::memcpy(res.data(), processed_vision_embeds.data(), processed_vision_embeds.get_byte_size());
    return res;
}

ov::Tensor InputsEmbedderQwen2VL::get_rotary_pos_emb(const std::vector<std::array<size_t, 3>>& grids_thw) {
    const size_t spatial_merge_size = m_vision_encoder->get_processor_config().merge_size;

    std::vector<std::vector<size_t>> all_pos_ids;
    size_t max_grid_size = 0;

    for (const auto& grid_thw : grids_thw) {
        size_t t = grid_thw.at(0);
        size_t h = grid_thw.at(1);
        size_t w = grid_thw.at(2);

        max_grid_size = std::max({max_grid_size, h, w});

        // According to spatial merge size, create height & width position IDs
        std::vector<size_t> hpos_ids;
        std::vector<size_t> wpos_ids;
        size_t h_blocks = h / spatial_merge_size;
        size_t w_blocks = w / spatial_merge_size;
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

        // Stack and repeat for each t
        for (size_t i = 0; i < t; ++i) {
            for (size_t j = 0; j < hpos_ids.size(); ++j) {
                all_pos_ids.push_back({hpos_ids[j], wpos_ids[j]});
            }
        }
    }

    // Calculate rotary embeddings for max_grid_size
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_embeddings_merger.get());
    ov::InferRequest& vision_embeddings_merger = infer_request_guard.get();
    const size_t dim = vision_embeddings_merger.get_tensor("rotary_pos_emb").get_shape().at(1);
    const float theta = 10000.0f;
    
    std::vector<float> inv_freq(dim / 2);
    for (size_t i = 0; i < dim / 2; ++i) {
        inv_freq[i] = 1.0f / std::pow(theta, static_cast<float>(i) / static_cast<float>(dim / 2));
    }

    std::vector<std::vector<float>> freqs(max_grid_size);
    for (size_t i = 0; i < max_grid_size; ++i) {
        freqs[i].resize(dim / 2);
        for (size_t j = 0; j < dim / 2; ++j) {
            freqs[i][j] = static_cast<float>(i) * inv_freq[j];
        }
    }

    ov::Tensor rotary_pos_emb(ov::element::f32, {all_pos_ids.size(), dim});
    float* output_data = rotary_pos_emb.data<float>();

    for (size_t i = 0; i < all_pos_ids.size(); ++i) {
        const auto& pos = all_pos_ids.at(i);
        size_t h_idx = pos.at(0);
        size_t w_idx = pos.at(1);
        std::copy_n(freqs[h_idx].begin(), dim / 2, output_data + i * dim);
        std::copy_n(freqs[w_idx].begin(), dim / 2, output_data + i * dim + dim / 2);
    }

    return rotary_pos_emb;
}

ov::Tensor InputsEmbedderQwen2VL::create_position_ids(
    const ov::Tensor& input_ids_tensor,
    const std::vector<std::array<size_t, 3>>& images_grid_thw,
    const std::vector<size_t>& images_sequence,
    const size_t image_id,
    const int64_t vision_start_token_id) {
    const size_t spatial_merge_size = m_vision_encoder->get_processor_config().merge_size;

    std::vector<std::array<size_t, 3>> reordered_images_grid_thw;
    for (size_t new_image_id : images_sequence) {
        reordered_images_grid_thw.push_back(images_grid_thw.at(new_image_id - image_id));
    }
    
    const int64_t* input_ids = input_ids_tensor.data<int64_t>();
    size_t batch_size = input_ids_tensor.get_shape().at(0);
    size_t seq_len = input_ids_tensor.get_shape().at(1);

    std::vector<size_t> vision_start_indices;
    for (size_t i = 0; i < seq_len; ++i) {
        if (input_ids[i] == vision_start_token_id) {
            vision_start_indices.push_back(i);
        }
    }

    ov::Tensor position_ids{ov::element::i64, {3, batch_size, seq_len}};
    int64_t* pos_data = position_ids.data<int64_t>();
    
    size_t st = 0;
    int64_t next_pos = 0;
    size_t grid_idx = 0;

    for (size_t i = 0; i < vision_start_indices.size(); ++i) {
        size_t ed = vision_start_indices.at(i);

        // Process text tokens before image
        if (st < ed) {
            for (size_t pos = st; pos < ed; ++pos) {
                pos_data[pos] = next_pos;               // temporal
                pos_data[seq_len + pos] = next_pos;     // height
                pos_data[2 * seq_len + pos] = next_pos; // width
                next_pos++;
            }
        }

        // Process image start token
        pos_data[ed] = next_pos;               // temporal
        pos_data[seq_len + ed] = next_pos;     // height
        pos_data[2 * seq_len + ed] = next_pos; // width
        next_pos++;
        ed++;

        // Process image token with grid
        if (grid_idx < reordered_images_grid_thw.size()) {
            const auto& grid = reordered_images_grid_thw.at(grid_idx);
            size_t llm_grid_h = grid.at(1) / spatial_merge_size;
            size_t llm_grid_w = grid.at(2) / spatial_merge_size;
            size_t ed_image = ed + llm_grid_h * llm_grid_w;

            // Fill temporal dimension
            std::fill_n(pos_data + ed, llm_grid_h * llm_grid_w, next_pos);

            // Fill height and width dimensions
            int64_t* height_data = pos_data + seq_len + ed;
            int64_t* width_data = pos_data + 2 * seq_len + ed;
            for (size_t h = 0; h < llm_grid_h; ++h) {
                std::fill_n(height_data + h * llm_grid_w, llm_grid_w, next_pos + h);
                for (size_t w = 0; w < llm_grid_w; ++w) {
                    width_data[h * llm_grid_w + w] = next_pos + w;
                }
            }

            next_pos += std::max(llm_grid_h, llm_grid_w);
            st = ed_image;
            grid_idx++;
        }
    }

    // Process remaining text tokens
    if (st < seq_len) {
        for (size_t pos = st; pos < seq_len; ++pos) {
            pos_data[pos] = next_pos;               // temporal
            pos_data[seq_len + pos] = next_pos;     // height
            pos_data[2 * seq_len + pos] = next_pos; // width
            next_pos++;
        }
    }

    return position_ids;
}

// [CDPruner] Text feature extraction functions implementation

std::vector<int64_t> InputsEmbedderQwen2VL::extract_instruction_tokens(const ov::Tensor& input_ids,
                                                                       int64_t image_pad_token_id,
                                                                       int64_t vision_start_token_id,
                                                                       int64_t vision_end_token_id) {
    std::vector<int64_t> instruction_tokens;

    // Get input_ids data
    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    size_t seq_len = input_ids.get_shape()[1];  // [batch_size, seq_len]

    bool inside_vision_region = false;

    // Iterate through the sequence to extract instruction tokens
    for (size_t i = 0; i < seq_len; ++i) {
        int64_t current_token = input_ids_data[i];

        // Check for vision start token
        if (current_token == vision_start_token_id) {
            inside_vision_region = true;
            continue;
        }

        // Check for vision end token
        if (current_token == vision_end_token_id) {
            inside_vision_region = false;
            continue;
        }

        // Skip if inside vision region or if it's an image pad token
        if (inside_vision_region || current_token == image_pad_token_id) {
            continue;
        }

        // This is a valid instruction token
        instruction_tokens.push_back(current_token);
    }

    return instruction_tokens;
}

ov::Tensor InputsEmbedderQwen2VL::extract_text_features_for_cdpruner(const ov::Tensor& input_ids,
                                                                     int64_t image_pad_token_id,
                                                                     int64_t vision_start_token_id,
                                                                     int64_t vision_end_token_id) {
    // Extract instruction tokens
    std::vector<int64_t> instruction_tokens =
        extract_instruction_tokens(input_ids, image_pad_token_id, vision_start_token_id, vision_end_token_id);

    if (instruction_tokens.empty()) {
        // If no instruction tokens found, create a zero tensor with proper dimensions
        ov::Tensor zero_embedding(ov::element::f32, {1, m_vlm_config.hidden_size});
        std::memset(zero_embedding.data<float>(), 0, zero_embedding.get_byte_size());
        return zero_embedding;
    }

    // [OPTIMIZED] Create batch tensor for all instruction tokens
    size_t num_tokens = instruction_tokens.size();
    ov::Tensor instruction_tokens_tensor(ov::element::i64, {1, num_tokens});
    int64_t* tokens_data = instruction_tokens_tensor.data<int64_t>();

    // Copy instruction tokens to tensor
    for (size_t i = 0; i < num_tokens; ++i) {
        tokens_data[i] = instruction_tokens[i];
    }

    // Acquire a fresh EmbeddingsRequest for this independent inference
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& local_req = embeddings_request_guard.get();

    ov::Tensor batch_embeddings = m_embedding->infer(local_req, instruction_tokens_tensor, false);

    // [OPTIMIZED] Calculate average pooling across sequence dimension
    ov::Tensor avg_embedding(ov::element::f32, {1, m_vlm_config.hidden_size});
    float* avg_data = avg_embedding.data<float>();
    const float* batch_data = batch_embeddings.data<const float>();

    // Initialize to zero
    std::memset(avg_data, 0, avg_embedding.get_byte_size());

    // Sum across all tokens: Σ_{j=1 to N} e_j
    for (size_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
        const float* token_embed = batch_data + token_idx * m_vlm_config.hidden_size;
        for (size_t dim = 0; dim < m_vlm_config.hidden_size; ++dim) {
            avg_data[dim] += token_embed[dim];
        }
    }

    // Calculate average: H_q = (1/N) * Σ_{j=1 to N} e_j
    float num_tokens_f = static_cast<float>(num_tokens);
    for (size_t dim = 0; dim < m_vlm_config.hidden_size; ++dim) {
        avg_data[dim] /= num_tokens_f;
    }

    return avg_embedding;
}

// [CDPruner] Position encoding adjustment function for pruning.
// Example: consider a single image whose visual tokens form a 3x3 grid (indices 0-8).
// If CDPruner keeps indices {0, 4, 7} (coords (0,0), (1,1), (2,1)), this helper
// rebuilds the temporal/height/width tables so that:
//   - only tokens at those coordinates are inserted back into the prompt,
//   - each kept token retains its original grid row/col offsets, and
//   - text tokens that follow resume indexing from the last visual position.
// Removed visual tokens are collected and printed for debugging to make pruning effects visible.
ov::Tensor InputsEmbedderQwen2VL::adjust_position_ids_for_pruning(
    const ov::Tensor& original_position_ids,
    const ov::Tensor& input_ids,
    size_t original_visual_tokens,
    size_t pruned_visual_tokens,
    int64_t vision_start_token_id,
    int64_t image_pad_token_id,
    const std::vector<std::array<size_t, 3>>& images_grid_thw,
    const std::vector<size_t>& images_sequence,
    size_t image_id,
    const std::vector<std::vector<size_t>>& kept_indices_per_image,
    std::vector<size_t>* original_tokens_per_region_out,
    std::vector<size_t>* pruned_tokens_per_region_out,
    std::vector<std::vector<bool>>* keep_flags_per_region_out) {
    if (original_visual_tokens == pruned_visual_tokens) {
        return original_position_ids;
    }

    OPENVINO_ASSERT(!images_sequence.empty(), "Image sequence must not be empty when pruning visual tokens");
    OPENVINO_ASSERT(!kept_indices_per_image.empty(), "Kept token indices are missing after pruning");

    const size_t spatial_merge_size = std::max<size_t>(1, m_vision_encoder->get_processor_config().merge_size);

    std::vector<std::array<size_t, 3>> reordered_images_grid_thw;
    reordered_images_grid_thw.reserve(images_sequence.size());
    for (size_t new_image_id : images_sequence) {
        OPENVINO_ASSERT(new_image_id >= image_id && new_image_id - image_id < images_grid_thw.size(),
                        "Image sequence index is out of range");
        reordered_images_grid_thw.push_back(images_grid_thw.at(new_image_id - image_id));
    }

    std::vector<size_t> original_tokens_per_region_local;
    original_tokens_per_region_local.reserve(reordered_images_grid_thw.size());
    for (const auto& grid : reordered_images_grid_thw) {
        size_t grid_t = std::max<size_t>(1, grid.at(0));
        size_t grid_h = grid.at(1);
        size_t grid_w = grid.at(2);
        OPENVINO_ASSERT(grid_h % spatial_merge_size == 0 && grid_w % spatial_merge_size == 0,
                        "Grid dimensions must be divisible by spatial merge size");
        size_t llm_grid_h = grid_h / spatial_merge_size;
        size_t llm_grid_w = grid_w / spatial_merge_size;
        original_tokens_per_region_local.push_back(grid_t * llm_grid_h * llm_grid_w);
    }
    if (original_tokens_per_region_out) {
        *original_tokens_per_region_out = original_tokens_per_region_local;
    }

    std::vector<std::vector<bool>> keep_flags_local;

    ov::Tensor updated_position_ids = update_position_ids(original_position_ids,
                                                          input_ids,
                                                          vision_start_token_id,
                                                          image_pad_token_id,
                                                          reordered_images_grid_thw,
                                                          kept_indices_per_image,
                                                          spatial_merge_size,
                                                          pruned_tokens_per_region_out,
                                                          nullptr,
                                                          keep_flags_per_region_out ? keep_flags_per_region_out
                                                                                    : &keep_flags_local);

    return updated_position_ids;
}

ov::Tensor InputsEmbedderQwen2VL::update_position_ids(
    const ov::Tensor& original_position_ids,
    const ov::Tensor& input_ids,
    int64_t vision_start_token_id,
    int64_t image_pad_token_id,
    const std::vector<std::array<size_t, 3>>& reordered_images_grid_thw,
    const std::vector<std::vector<size_t>>& kept_indices_per_image,
    size_t spatial_merge_size,
    std::vector<size_t>* pruned_tokens_per_region_out,
    std::vector<std::vector<size_t>>* normalized_kept_indices_out,
    std::vector<std::vector<bool>>* keep_flags_out) {
    const ov::Shape& pos_shape = original_position_ids.get_shape();
    OPENVINO_ASSERT(pos_shape.size() == 3, "Position ids tensor must have 3 dimensions");

    const size_t dims = pos_shape.at(0);
    const size_t batch_size = pos_shape.at(1);
    const size_t seq_len = pos_shape.at(2);
    OPENVINO_ASSERT(dims == 3, "Position ids tensor first dimension must be 3");

    struct RegionInfo {
        size_t tokens;
        size_t grid_width;
        size_t spatial_area;
        size_t offset;
    };

    std::vector<RegionInfo> regions;
    regions.reserve(reordered_images_grid_thw.size());

    size_t cumulative_offset = 0;
    for (const auto& grid : reordered_images_grid_thw) {
    size_t grid_t = std::max<size_t>(1, grid.at(0));
    size_t grid_h = grid.at(1);
    size_t grid_w = grid.at(2);
        OPENVINO_ASSERT(grid_h % spatial_merge_size == 0 && grid_w % spatial_merge_size == 0,
                        "Grid dimensions must be divisible by spatial merge size");

        size_t llm_grid_h = grid_h / spatial_merge_size;
        size_t llm_grid_w = grid_w / spatial_merge_size;
    size_t spatial_area = llm_grid_h * llm_grid_w;
    OPENVINO_ASSERT(spatial_area > 0, "Vision region must contain at least one spatial token");
    size_t total_tokens = spatial_area * grid_t;

    regions.push_back({total_tokens,
               llm_grid_w == 0 ? 1 : llm_grid_w,
               spatial_area,
               cumulative_offset});
    cumulative_offset += total_tokens;
    }

    const size_t region_count = regions.size();

    auto normalize_indices = [&](const std::vector<std::vector<size_t>>& raw_indices) {
        std::vector<std::vector<size_t>> normalized;
        if (raw_indices.empty()) {
            normalized.assign(region_count, {});
            return normalized;
        }

        if (raw_indices.size() == region_count) {
            normalized = raw_indices;
            return normalized;
        }

        if (raw_indices.size() == 1 && region_count > 1) {
            normalized.assign(region_count, {});
            const auto& aggregated = raw_indices.front();
            for (size_t kept_idx : aggregated) {
                OPENVINO_ASSERT(kept_idx < cumulative_offset, "Aggregated kept index is out of range");
                size_t image_idx = 0;
                for (; image_idx < region_count; ++image_idx) {
                    size_t region_start = regions[image_idx].offset;
                    size_t region_end = region_start + regions[image_idx].tokens;
                    if (kept_idx >= region_start && kept_idx < region_end) {
                        normalized[image_idx].push_back(kept_idx - region_start);
                        break;
                    }
                }
                OPENVINO_ASSERT(image_idx < region_count, "Aggregated kept index cannot be assigned to a vision region");
            }
            return normalized;
        }

        OPENVINO_THROW("Kept token indices layout does not match vision regions");
    };

    std::vector<std::vector<size_t>> normalized_kept_indices = normalize_indices(kept_indices_per_image);
    for (auto& per_region_indices : normalized_kept_indices) {
        std::sort(per_region_indices.begin(), per_region_indices.end());
        auto unique_end = std::unique(per_region_indices.begin(), per_region_indices.end());
        per_region_indices.erase(unique_end, per_region_indices.end());
    }
    if (normalized_kept_indices_out) {
        *normalized_kept_indices_out = normalized_kept_indices;
    }

    if (pruned_tokens_per_region_out) {
        pruned_tokens_per_region_out->assign(region_count, 0);
    }

    std::vector<std::vector<bool>> local_keep_flags;
    std::vector<std::vector<bool>>& keep_flags = keep_flags_out ? *keep_flags_out : local_keep_flags;
    keep_flags.clear();
    keep_flags.reserve(region_count);

    size_t total_removed = 0;
    for (size_t idx = 0; idx < region_count; ++idx) {
        const auto& region = regions[idx];
        std::vector<bool> flags(region.tokens, false);
        size_t kept_count = 0;
        for (size_t kept_idx : normalized_kept_indices[idx]) {
            OPENVINO_ASSERT(kept_idx < region.tokens, "Kept visual token index is out of range");
            flags[kept_idx] = true;
            ++kept_count;
        }
        if (region.tokens < kept_count) {
            std::cerr << "[CDPruner] Region mismatch -- idx=" << idx << ", tokens=" << region.tokens
                      << ", kept=" << kept_count << ", cumulative_offset=" << regions[idx].offset
                      << std::endl;
            if (idx + 1 < regions.size()) {
                std::cerr << "[CDPruner] Next region offset=" << regions[idx + 1].offset << std::endl;
            }
            OPENVINO_THROW("Number of kept tokens exceeds original visual tokens for image");
        }
        if (pruned_tokens_per_region_out) {
            (*pruned_tokens_per_region_out)[idx] = kept_count;
        }
        total_removed += region.tokens - kept_count;
        keep_flags.push_back(std::move(flags));
    }
    OPENVINO_ASSERT(seq_len >= total_removed, "Sequence length underflow after pruning");
    size_t new_seq_len = seq_len - total_removed;

    ov::Tensor new_position_ids(original_position_ids.get_element_type(), {3, batch_size, new_seq_len});
    int64_t* new_data = new_position_ids.data<int64_t>();
    int64_t* temporal_out = new_data;
    int64_t* height_out = new_data + batch_size * new_seq_len;
    int64_t* width_out = new_data + 2 * batch_size * new_seq_len;
    size_t batch_stride_out = new_seq_len;

    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    const int64_t* original_data = original_position_ids.data<const int64_t>();

    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        size_t seq_idx = 0;
        size_t write_idx = 0;
        size_t image_idx = 0;
        bool inside_vision = false;
        size_t visual_index = 0;
        size_t batch_offset_in = batch_idx * seq_len;
        size_t batch_offset_out = batch_idx * batch_stride_out;
        int64_t next_pos = seq_len > 0 ? original_data[batch_offset_in] : 0;
    int64_t grid_temporal_value = 0;
    int64_t grid_base_value = 0;
    int64_t max_height = next_pos - 1;
    int64_t max_width = next_pos - 1;
    int64_t max_temporal = next_pos - 1;

        while (seq_idx < seq_len) {
            int64_t token_id = input_ids_data[batch_offset_in + seq_idx];

            if (inside_vision && token_id == image_pad_token_id) {
                OPENVINO_ASSERT(image_idx < region_count, "Vision region index out of bounds");
                const auto& region = regions[image_idx];
                const size_t grid_width = region.grid_width > 0 ? region.grid_width : 1;
                const size_t spatial_area = region.spatial_area > 0 ? region.spatial_area : grid_width;
                bool keep_token = (image_idx < keep_flags.size() && visual_index < keep_flags[image_idx].size())
                                      ? keep_flags[image_idx][visual_index]
                                      : false;
                size_t local_index = spatial_area > 0 ? (visual_index % spatial_area) : 0;
                size_t row = grid_width > 0 ? local_index / grid_width : 0;
                size_t col = grid_width > 0 ? local_index % grid_width : 0;
                size_t temporal_index = spatial_area > 0 ? (visual_index / spatial_area) : 0;
                if (keep_token) {
                    int64_t temporal_value = grid_temporal_value + static_cast<int64_t>(temporal_index);
                    int64_t height_value = grid_base_value + static_cast<int64_t>(row);
                    int64_t width_value = grid_base_value + static_cast<int64_t>(col);

                    temporal_out[batch_offset_out + write_idx] = temporal_value;
                    height_out[batch_offset_out + write_idx] = height_value;
                    width_out[batch_offset_out + write_idx] = width_value;

                    max_height = std::max(max_height, height_value);
                    max_width = std::max(max_width, width_value);
                    max_temporal = std::max(max_temporal, temporal_value);

                    ++write_idx;
                }

                ++visual_index;
                ++seq_idx;
                continue;
            }

            if (inside_vision) {
                inside_vision = false;
                int64_t region_max = std::max(std::max(max_height, max_width), max_temporal);
                next_pos = region_max + 1;
                ++image_idx;
                visual_index = 0;
                max_height = next_pos - 1;
                max_width = next_pos - 1;
                max_temporal = next_pos - 1;
            }

            int64_t temporal_value = next_pos;
            temporal_out[batch_offset_out + write_idx] = temporal_value;
            height_out[batch_offset_out + write_idx] = temporal_value;
            width_out[batch_offset_out + write_idx] = temporal_value;

            ++write_idx;
            ++seq_idx;
            ++next_pos;

            if (token_id == vision_start_token_id) {
                inside_vision = true;
                visual_index = 0;
                grid_temporal_value = next_pos;
                grid_base_value = next_pos;
            }
        }

        OPENVINO_ASSERT(!inside_vision, "Unexpected end of sequence inside a vision region");
        OPENVINO_ASSERT(image_idx == keep_flags.size(), "Mismatch between processed vision regions and metadata");
        OPENVINO_ASSERT(write_idx == new_seq_len, "Pruned position ids length mismatch after recomputation");
    }

    return new_position_ids;
}

std::vector<ov::Tensor> InputsEmbedderQwen2VL::convert_visual_features_for_cdpruner(
    const ov::Tensor& merged_image_embeddings,
    size_t image_num) {
    // Convert from [num_patches, embedding_dim] to image_num * [1, num_patches, embedding_dim]
    ov::Shape original_shape = merged_image_embeddings.get_shape();
    size_t num_patches = original_shape[0];
    size_t embedding_dim = original_shape[1];
    size_t new_patches = num_patches / image_num;
    OPENVINO_ASSERT(original_shape[0] == new_patches * image_num, "Inconsistent number of patches per image");

    std::vector<ov::Tensor> visual_features;
    const float* src_data = merged_image_embeddings.data<const float>();
    size_t total_elements = new_patches * embedding_dim;
    for (size_t i = 0; i < image_num; i++) {
        ov::Shape new_shape = {1, new_patches, embedding_dim};
        ov::Tensor features(merged_image_embeddings.get_element_type(), new_shape);
        float* dst_data = features.data<float>();
        std::memcpy(dst_data, src_data + total_elements * i, total_elements * sizeof(float));
        visual_features.push_back(features);
    }
    return visual_features;
}

// [CDPruner] Create merged embeddings for pruned visual tokens
ov::Tensor InputsEmbedderQwen2VL::merge_text_and_image_embeddings_with_pruning(
    const ov::Tensor& input_ids,
    const ov::Tensor& text_embeds,
    const ov::Tensor& pruned_vision_embeds,
    int64_t image_pad_token_id,
    int64_t vision_start_token_id,
    int64_t vision_end_token_id,
    const std::vector<size_t>& original_tokens_per_region,
    const std::vector<std::vector<bool>>& keep_flags_per_region) {
    auto text_embeds_shape = text_embeds.get_shape();
    size_t batch_size = text_embeds_shape.at(0);
    size_t original_seq_length = text_embeds_shape.at(1);
    size_t hidden_size = text_embeds_shape.at(2);

    size_t total_original_visual_tokens = 0;
    for (size_t tokens : original_tokens_per_region) {
        total_original_visual_tokens += tokens;
    }

    size_t pruned_visual_tokens = pruned_vision_embeds.get_shape()[0];
    size_t tokens_removed = total_original_visual_tokens - pruned_visual_tokens;
    size_t new_seq_length = original_seq_length - tokens_removed;

    const size_t region_count = original_tokens_per_region.size();
    OPENVINO_ASSERT(region_count == keep_flags_per_region.size(),
                    "Vision region metadata not available while merging embeddings");

    size_t computed_pruned_tokens = 0;
    for (const auto& mask : keep_flags_per_region) {
        computed_pruned_tokens += static_cast<size_t>(std::count(mask.begin(), mask.end(), true));
    }
    OPENVINO_ASSERT(computed_pruned_tokens == pruned_visual_tokens,
                    "Kept visual token mask total mismatch with pruned embeddings during merge");

    ov::Tensor merged_embeds(text_embeds.get_element_type(), {batch_size, new_seq_length, hidden_size});

    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    const float* text_embeds_data = text_embeds.data<const float>();
    const float* vision_embeds_data = pruned_vision_embeds.data<const float>();
    float* merged_embeds_data = merged_embeds.data<float>();

    size_t vision_embed_idx = 0;

    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        size_t out_seq_idx = 0;
        size_t region_idx = 0;
        size_t pad_index = 0;
        bool inside_vision_region = false;

        for (size_t seq_idx = 0; seq_idx < original_seq_length; ++seq_idx) {
            size_t input_flat_idx = batch_idx * original_seq_length + seq_idx;
            int64_t token_id = input_ids_data[input_flat_idx];

            if (token_id == vision_start_token_id) {
                OPENVINO_ASSERT(region_idx < region_count,
                                "Encountered more vision regions than metadata entries while merging embeddings");
                inside_vision_region = true;
                pad_index = 0;
            }

            if (inside_vision_region && token_id == image_pad_token_id) {
                const auto& keep_mask = keep_flags_per_region.at(region_idx);
                OPENVINO_ASSERT(pad_index < keep_mask.size(),
                                "Visual token index exceeds region token count while merging embeddings");
                if (keep_mask[pad_index]) {
                    OPENVINO_ASSERT(out_seq_idx < new_seq_length,
                                    "Merged embeddings index exceeds expected sequence length");
                    size_t out_flat_idx = batch_idx * new_seq_length + out_seq_idx;
                    std::copy_n(vision_embeds_data + vision_embed_idx * hidden_size,
                                hidden_size,
                                merged_embeds_data + out_flat_idx * hidden_size);
                    ++vision_embed_idx;
                    ++out_seq_idx;
                }
                ++pad_index;
                continue;
            }

            OPENVINO_ASSERT(out_seq_idx < new_seq_length, "Merged embeddings index exceeds expected sequence length");
            size_t out_flat_idx = batch_idx * new_seq_length + out_seq_idx;
            std::copy_n(text_embeds_data + input_flat_idx * hidden_size,
                        hidden_size,
                        merged_embeds_data + out_flat_idx * hidden_size);
            ++out_seq_idx;

            if (inside_vision_region && token_id == vision_end_token_id) {
                const auto& keep_mask = keep_flags_per_region.at(region_idx);
                OPENVINO_ASSERT(pad_index == keep_mask.size(),
                                "Mismatch between consumed visual tokens and region metadata while merging embeddings");
                inside_vision_region = false;
                ++region_idx;
            }
        }

        OPENVINO_ASSERT(!inside_vision_region,
                        "Unexpected end of sequence inside a vision region while merging embeddings");
        OPENVINO_ASSERT(region_idx == region_count, "Not all vision regions processed while merging embeddings");
        OPENVINO_ASSERT(out_seq_idx == new_seq_length, "Merged embeddings sequence length mismatch after pruning");
    }

    OPENVINO_ASSERT(vision_embed_idx == pruned_visual_tokens,
                    "Pruned vision embeddings were not fully consumed during merge");

    return merged_embeds;
}

} // namespace ov::genai
