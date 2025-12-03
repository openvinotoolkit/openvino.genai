
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
#include "openvino/op/if.hpp"
#include "openvino/op/concat.hpp"

#include "visual_language/vl_sdpa_transformations.hpp"

namespace ov::genai {

namespace {

// Chat template hardcodes char sequence instead of referring to tag values, so NATIVE_TAG is hardcoded as well.
const std::string NATIVE_TAG = "<|vision_start|><|image_pad|><|vision_end|>";
const std::string NATIVE_VIDEO_TAG = "<|vision_start|><|video_pad|><|vision_end|>";

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

std::pair<std::shared_ptr<ov::Model>, std::shared_ptr<ov::op::v0::Result>> patch_preprocess_branch_image(
    const std::shared_ptr<ov::op::v0::Parameter>& raw_image_1,
    const std::shared_ptr<ov::op::v0::Parameter>& resize_shape,
    const std::shared_ptr<ov::op::v0::Constant>& image_mean,
    const std::shared_ptr<ov::op::v0::Constant>& image_scale,
    const std::shared_ptr<ov::op::v0::Parameter>& tile_shape) {
    auto img_f32_nchw = create_f32_nchw_input(raw_image_1);
    auto img_resized = create_bicubic_resize(img_f32_nchw, resize_shape);
    auto img_normalized = create_normalization(img_resized, image_mean, image_scale);
    auto temporal_images = std::make_shared<ov::op::v0::Tile>(img_normalized, tile_shape);
    auto results = std::make_shared<ov::op::v0::Result>(temporal_images);
    return {
        std::make_shared<ov::Model>(results, ov::ParameterVector{raw_image_1, resize_shape, tile_shape}, "then_body"),
        results};
}

std::pair<std::shared_ptr<ov::Model>, std::shared_ptr<ov::op::v0::Result>> patch_preprocess_branch_video(
    const std::shared_ptr<ov::op::v0::Parameter>& cond_img_vid,
    const std::shared_ptr<ov::op::v0::Parameter>& raw_frame_1,
    const std::shared_ptr<ov::op::v0::Parameter>& raw_frame_2,
    const std::shared_ptr<ov::op::v0::Parameter>& resize_shape,
    const std::shared_ptr<ov::op::v0::Constant>& image_mean,
    const std::shared_ptr<ov::op::v0::Constant>& image_scale) {
    auto img_f32_nchw_1 = create_f32_nchw_input(raw_frame_1);
    auto img_resized_1 = create_bicubic_resize(img_f32_nchw_1, resize_shape);
    auto img_normalized_1 = create_normalization(img_resized_1, image_mean, image_scale);

    auto img_f32_nchw_2 = create_f32_nchw_input(raw_frame_2);
    auto img_resized_2 = create_bicubic_resize(img_f32_nchw_2, resize_shape);
    auto img_normalized_2 = create_normalization(img_resized_2, image_mean, image_scale);

    int64_t concat_axis = 0;
    ov::OutputVector inputs_to_concat = {img_normalized_1->output(0), img_normalized_2->output(0)};
    auto temporal_images = std::make_shared<ov::op::v0::Concat>(inputs_to_concat, concat_axis);

    auto result_temperal_images = std::make_shared<ov::op::v0::Result>(temporal_images);

    // If node's limitation: condition node must be output.
    auto result_ignore = std::make_shared<ov::op::v0::Result>(cond_img_vid);
    return {std::make_shared<ov::Model>(ov::ResultVector{result_temperal_images, result_ignore},
                                        ov::ParameterVector{cond_img_vid, raw_frame_1, raw_frame_2, resize_shape},
                                        "else_body"),
            result_temperal_images};
}

std::shared_ptr<ov::Model> patch_preprocess_into_model(const std::shared_ptr<ov::Model>& model_org,
                                                       const ov::op::v0::Constant& image_mean_tensor,
                                                       const ov::op::v0::Constant& image_scale_tensor) {
    auto cond_img_vid = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto raw_images_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{-1, -1, -1, -1});
    auto raw_images_2 = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{-1, -1, -1, -1});

    auto resize_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2});
    auto tile_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{4});
    auto reshape_shape8d = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{8});
    auto reshape_shape4d = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{4});
    auto reshape_shape2d = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2});

    cond_img_vid->set_friendly_name("cond_img_vid");
    cond_img_vid->output(0).get_tensor().set_names({"cond_img_vid"});
    raw_images_1->set_friendly_name("raw_images_1");
    raw_images_1->output(0).get_tensor().set_names({"raw_images_1"});
    raw_images_2->set_friendly_name("raw_images_2");
    raw_images_2->output(0).get_tensor().set_names({"raw_images_2"});

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

    // If
    auto then_raw_image_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{-1, -1, -1, -1});
    auto then_resize_target_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2});
    auto then_tile_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{4});
    auto model_then = patch_preprocess_branch_image(then_raw_image_1,
                                                    then_resize_target_shape,
                                                    image_mean,
                                                    image_scale,
                                                    then_tile_shape);

    auto else_video = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto else_raw_frame_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{-1, -1, -1, -1});
    auto else_raw_frame_2 = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{-1, -1, -1, -1});
    auto else_resize_target_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2});
    auto model_else = patch_preprocess_branch_video(else_video,
                                                    else_raw_frame_1,
                                                    else_raw_frame_2,
                                                    else_resize_target_shape,
                                                    image_mean,
                                                    image_scale);

    auto if_op = std::make_shared<ov::op::v8::If>();
    if_op->set_then_body(model_then.first);
    if_op->set_else_body(model_else.first);
    if_op->set_input(cond_img_vid->output(0), nullptr, else_video);
    
    if_op->set_input(raw_images_1->output(0), nullptr, else_raw_frame_1);
    if_op->set_input(raw_images_2->output(0), nullptr, else_raw_frame_2);
    if_op->set_input(resize_shape->output(0), nullptr, else_resize_target_shape);

    if_op->set_input(raw_images_1->output(0), then_raw_image_1, nullptr);
    if_op->set_input(resize_shape->output(0), then_resize_target_shape, nullptr);
    if_op->set_input(tile_shape->output(0), then_tile_shape, nullptr);

    auto temporal_images = if_op->set_output(model_then.second, model_else.second);
    auto img_8d =
        create_transpose_patches(temporal_images.get_node_shared_ptr(),
                                 reshape_shape8d,
                                 std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                                        Shape{8},
                                                                        std::vector<int32_t>{0, 2, 5, 3, 6, 1, 4, 7}));

    auto img_4d = create_transpose_patches(
        std::move(img_8d),
        reshape_shape4d,
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, Shape{4}, std::vector<int32_t>{0, 2, 1, 3}));

    auto img_2d = create_flatten_patches(std::move(img_4d), reshape_shape2d);

    auto params_org = model_org->get_parameters();
    OPENVINO_ASSERT(params_org.size() == 1u);

    ov::replace_node(params_org[0], img_2d);

    auto results = model_org->get_results();
    return std::make_shared<ov::Model>(results,
                                       ov::ParameterVector{cond_img_vid,
                                                           raw_images_1,
                                                           raw_images_2,
                                                           resize_shape,
                                                           tile_shape,
                                                           reshape_shape8d,
                                                           reshape_shape4d,
                                                           reshape_shape2d});
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
        image_embeds.push_back(encoded_image.resized_source);

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

    return {std::move(reordered_image_embeds), std::move(reordered_images_grid_thw)};
}

std::pair<std::vector<ov::Tensor>, std::vector<std::array<size_t, 3>>> reorder_video_embeds_and_grid_thw(
    const std::vector<EncodedVideo>& videos,
    const std::vector<size_t>& videos_sequence
) {
    std::vector<ov::Tensor> video_embeds;
    std::vector<std::array<size_t, 3>> videos_grid_thw;

    for (const auto& encoded_video : videos) {
        video_embeds.push_back(encoded_video.video_features);
        size_t grid_t = encoded_video.frame_num;
        size_t grid_h = encoded_video.resized_source_size.height;
        size_t grid_w = encoded_video.resized_source_size.width;
        videos_grid_thw.push_back({grid_t, grid_h, grid_w});
    }

    std::vector<ov::Tensor> reordered_video_embeds;
    std::vector<std::array<size_t, 3>> reordered_videos_grid_thw;
    for (size_t new_video_id : videos_sequence) {
        reordered_video_embeds.push_back(video_embeds.at(new_video_id));
        reordered_videos_grid_thw.push_back(videos_grid_thw.at(new_video_id));
    }

    return {reordered_video_embeds, reordered_videos_grid_thw};
}

static void calc_cu_seqlens(const std::vector<std::array<size_t, 3>>& reordered_grid_thw,
                            int32_t& cumsum,
                            std::vector<int32_t>& cu_seqlens) {
    for (const auto& grid_thw : reordered_grid_thw) {
        size_t slice_len = grid_thw.at(1) * grid_thw.at(2);
        for (size_t t = 0; t < grid_thw.at(0); ++t) {
            cumsum += slice_len;
            cu_seqlens.push_back(cumsum);
        }
    }
}

ov::Tensor get_attention_mask(const std::vector<std::array<size_t, 3>>& reordered_images_grid_thw, const std::vector<std::array<size_t, 3>>& reordered_videos_grid_thw) {
    // Calculate cumulative sequence lengths for attention mask
    std::vector<int32_t> cu_seqlens;
    cu_seqlens.push_back(0);
    int32_t cumsum = 0;

    calc_cu_seqlens(reordered_videos_grid_thw, cumsum, cu_seqlens);
    calc_cu_seqlens(reordered_images_grid_thw, cumsum, cu_seqlens);

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

ov::Tensor get_cu_seqlens(const std::vector<std::array<size_t, 3>>& reordered_images_grid_thw, const std::vector<std::array<size_t, 3>>& reordered_videos_grid_thw) {
    // Calculate cumulative sequence lengths for attention mask
    std::vector<int32_t> cu_seqlens;
    cu_seqlens.push_back(0);
    int32_t cumsum = 0;
    calc_cu_seqlens(reordered_videos_grid_thw, cumsum, cu_seqlens);
    calc_cu_seqlens(reordered_images_grid_thw, cumsum, cu_seqlens);

    ov::Tensor t_cu_seqlens(ov::element::i32, {cu_seqlens.size()});
    std::memcpy(t_cu_seqlens.data<int32_t>(), cu_seqlens.data(), cu_seqlens.size() * sizeof(int32_t));
    return t_cu_seqlens;
}

ov::Tensor concatenate_video_image_embeds(const std::vector<ov::Tensor>& reordered_video_embeds, const std::vector<ov::Tensor>& reordered_image_embeds) {
    // one image + zero video.
    if (reordered_image_embeds.size() == 1u && reordered_video_embeds.empty()) {
        return reordered_image_embeds.at(0);
    }
    // zero image + one video.
    if (reordered_image_embeds.empty() && reordered_video_embeds.size() == 1u) {
        return reordered_video_embeds.at(0);
    }
    // zero image + zero video.
    if (reordered_image_embeds.empty() && reordered_video_embeds.empty()) {
        return ov::Tensor();
    }

    // multiple image(s) or video(s).
    ov::Tensor concatenated_embeds;
    size_t total_length = 0;
    for (const auto& embed : reordered_video_embeds) {
        total_length += embed.get_shape().at(0);
    }
    for (const auto& embed : reordered_image_embeds) {
        total_length += embed.get_shape().at(0);
    }

    // The video and image embeds features are from same embedded model.
    // So reordered_image_embeds and reordered_video_embeds should have same element type and hidden_dim.
    if (reordered_video_embeds.size() > 0u && reordered_image_embeds.size() > 0u) {
        OPENVINO_ASSERT(reordered_video_embeds.at(0).get_element_type() == reordered_image_embeds.at(0).get_element_type());
        OPENVINO_ASSERT(reordered_video_embeds.at(0).get_shape().at(1) == reordered_image_embeds.at(0).get_shape().at(1));
    }

    size_t hidden_dim;
    ov::element::Type type;
    if (reordered_image_embeds.size() > 0u) {
        hidden_dim = reordered_image_embeds.at(0).get_shape().at(1);
        type = reordered_image_embeds.at(0).get_element_type();
    } else {
        hidden_dim = reordered_video_embeds.at(0).get_shape().at(1);
        type = reordered_video_embeds.at(0).get_element_type();
    }

    concatenated_embeds = ov::Tensor(type, {total_length, hidden_dim});
    uint8_t* concat_data = reinterpret_cast<uint8_t*>(concatenated_embeds.data());

    size_t offset = 0;
    for (const auto& embed : reordered_video_embeds) {
        std::memcpy(concat_data + offset, embed.data(), embed.get_byte_size());
        offset += embed.get_byte_size();
    }
    for (const auto& embed : reordered_image_embeds) {
        std::memcpy(concat_data + offset, embed.data(), embed.get_byte_size());
        offset += embed.get_byte_size();
    }
    return concatenated_embeds;
}

ov::Tensor merge_text_and_video_image_embeddings(
    const ov::Tensor& input_ids,
    const ov::Tensor& text_embeds, 
    const ov::Tensor& processed_image_embeds,
    const ov::Tensor& processed_video_embeds,
    const int64_t image_pad_token_id,
    const int64_t video_pad_token_id
) {
    ov::Tensor merged_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
    std::memcpy(merged_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());

    auto text_embeds_shape = text_embeds.get_shape();
    size_t batch_size = text_embeds_shape.at(0);
    size_t seq_length = text_embeds_shape.at(1);
    size_t hidden_size = text_embeds_shape.at(2);

    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    float* merged_embeds_data = merged_embeds.data<float>();
    const float* image_embeds_data = processed_image_embeds.data<const float>();
    const float* video_embeds_data = processed_video_embeds.data<const float>();

    size_t image_embed_idx = 0;
    size_t video_embed_idx = 0;
    const int64_t img_token = image_pad_token_id;
    const int64_t vid_token = video_pad_token_id;
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        for (size_t seq_idx = 0; seq_idx < seq_length; ++seq_idx) {
            size_t flat_idx = batch_idx * seq_length + seq_idx;
            if (input_ids_data[flat_idx] == vid_token) {
                std::copy_n(video_embeds_data + video_embed_idx * hidden_size,
                            hidden_size,
                            merged_embeds_data + flat_idx * hidden_size);
                ++video_embed_idx;
            } else if (input_ids_data[flat_idx] == img_token) {
                std::copy_n(image_embeds_data + image_embed_idx * hidden_size,
                            hidden_size,
                            merged_embeds_data + flat_idx * hidden_size);
                ++image_embed_idx;
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

    auto image_mean = ov::op::v0::Constant(ov::element::f32, ov::Shape{1, a_image_mean.size(), 1, 1}, a_image_mean.data());
    auto image_scale = ov::op::v0::Constant(ov::element::f32, ov::Shape{1, a_image_scale.size(), 1, 1}, a_image_scale.data());

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
void VisionEncoderQwen2VL::encode_with_imagepreprocess_cpp(const std::vector<ov::Tensor>& images,
                                                           const ov::AnyMap& config_map,
                                                           ov::Tensor& out_tensor,
                                                           ImageSize& out_rsz_size,
                                                           size_t frame_num,
                                                           size_t frame_id) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    // The default value of temporal_patch_size for original QWen2-VL and QWen2.5-VL is 2.
    // If images.size() == 1: means processing image.
    // If images.size() == 2: means processing video.
    // If images.size() == others: undefined behaviour. so the following check is required.
    // NOTE: The following assertion enforces that temporal_patch_size == 2.
    // This is a limitation of the current model architectures (QWen2-VL and QWen2.5-VL), which are designed to process
    // either a single image or a pair of video frames (temporal_patch_size = 2). The code and model are not guaranteed
    // to work correctly for other values of temporal_patch_size. If support for more frames or different patch sizes is
    // required in the future, both the model and this preprocessing logic will need to be updated accordingly.
    OPENVINO_ASSERT(config.temporal_patch_size == 2u, "temporal_patch_size != 2.");
    if (images.size() > 1)
        OPENVINO_ASSERT(config.temporal_patch_size == images.size(), "temporal_patch_size != images.size()");

    ov::Shape orig_shape = images[0].get_shape();
    ImageSize target_image_size = qwen2_vl_utils::smart_resize(orig_shape.at(1),
                                                               orig_shape.at(2),
                                                               config.patch_size * config.merge_size,
                                                               config.min_pixels,
                                                               config.max_pixels);

    ov::Tensor tiled_patches(ov::element::f32,
                             {config.temporal_patch_size, 3, target_image_size.height, target_image_size.width});

    for (size_t i = 0; i < config.temporal_patch_size; i++) {
        const auto& image = images.size() > i ? images[i] : images[0];

        clip_image_u8 input_image = tensor_to_clip_image_u8(image);
        clip_image_u8 resized_image;
        bicubic_resize(input_image, resized_image, target_image_size.width, target_image_size.height);

        clip_ctx ctx;
        std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
        std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);
        clip_image_f32 normalized_image = clip_image_preprocess(ctx, resized_image);

        auto patch = clip_image_f32_to_tensor(normalized_image);

        std::memcpy(tiled_patches.data<float>() + i * patch.get_byte_size() / sizeof(float),
                    patch.data<float>(),
                    patch.get_byte_size());
    }
    auto patches = std::move(tiled_patches);

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
    // Just avoid to multiple copy.
    if (frame_id == 0u) {
        auto out_shape = infer_output.get_shape();
        out_shape[0] = out_shape[0] * frame_num;
        out_tensor = ov::Tensor(infer_output.get_element_type(), out_shape);
    }

    std::memcpy(reinterpret_cast<uint8_t*>(out_tensor.data()) + frame_id * infer_output.get_byte_size(),
                infer_output.data(),
                infer_output.get_byte_size());
    out_rsz_size = ImageSize{grid_h, grid_w};
}

/**
 * @brief Encode image or video frames. here is the OV version of encode_with_imagepreprocess_cpp
 * @param images size == 1 means image input;  size == 2, means 2 frames from video
 */
void VisionEncoderQwen2VL::encode_with_imagepreprocess_ov(const std::vector<ov::Tensor>& images,
                                                          const ov::AnyMap& config_map,
                                                          ov::Tensor& out_tensor,
                                                          ImageSize& out_rsz_size,
                                                          size_t frame_num,
                                                          size_t frame_id) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    OPENVINO_ASSERT(images.size() == 1 || images.size() == 2);
    if (images.size() == 2) {
        OPENVINO_ASSERT(images[0].get_shape() == images[1].get_shape(), "Video frames should have same layout.");
    }

    ov::Shape image_shape = images[0].get_shape();
    auto original_height = image_shape.at(1);
    auto original_width = image_shape.at(2);

    ImageSize target_image_size = qwen2_vl_utils::smart_resize(
        original_height, 
        original_width, 
        config.patch_size * config.merge_size,
        config.min_pixels,
        config.max_pixels
    );

    // The default value of temporal_patch_size for original QWen2-VL and QWen2.5-VL is 2.
    // In this model, Only 2 frames are processed at a time, so the following check is required.
    // If cond_img_vid = 1: means image branch, just duplicating input_image_1 as input_image_2
    // If cond_img_vid = 0: means video branch, processing adjacent frames.
    OPENVINO_ASSERT(config.temporal_patch_size == 2u, "temporal_patch_size != 2.");
    const float VIDEO_BRANCH_CONDITION = 0.f;
    const float IMAGE_BRANCH_CONDITION = 1.f;
    std::vector<float> cond_img_vid_data{images.size() == 2u ? VIDEO_BRANCH_CONDITION : IMAGE_BRANCH_CONDITION};
    ov::Tensor cond_img_vid(ov::element::f32, ov::Shape{1}, cond_img_vid_data.data());
    // const_cast is safe as ov::Tensor only views the data and doesn't modify it.
    ov::Tensor input_image_1(
        ov::element::u8, 
        image_shape, 
        const_cast<uint8_t*>(images[0].data<uint8_t>())
    );
    ov::Tensor input_image_2(
        ov::element::u8,
        image_shape,
        const_cast<uint8_t*>(images.size() == 2 ? images[1].data<uint8_t>() : images[0].data<uint8_t>())
    );

    uint64_t a_target_shape[2] = {target_image_size.height, target_image_size.width};
    ov::Tensor target_shape(ov::element::i64, ov::Shape{2}, a_target_shape);

    auto patches_shape = images[0].get_shape();
    size_t temporal_patch_size = std::max(static_cast<size_t>(patches_shape.at(0)), static_cast<size_t>(config.temporal_patch_size));
    size_t channel = image_shape.at(3);

    size_t grid_t = temporal_patch_size / config.temporal_patch_size;
    size_t grid_h = target_image_size.height / config.patch_size;
    size_t grid_w = target_image_size.width / config.patch_size;

    size_t repeats = 1;
    if (patches_shape.at(0) == 1) {
        repeats = config.temporal_patch_size;
    }
    uint64_t a_tile_shape[4] = {static_cast<size_t>(repeats), 1, 1, 1};

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
    ov::Tensor tile_shape(ov::element::i64, ov::Shape{4}, a_tile_shape);
    ov::Tensor reshape_shape8d(ov::element::i64, ov::Shape{8}, a_temp_shape8d);
    ov::Tensor reshape_shape4d(ov::element::i64, ov::Shape{4}, a_temp_shape4d);
    ov::Tensor reshape_shape2d(ov::element::i64, ov::Shape{2}, last_output_shape);

    encoder.set_tensor("cond_img_vid", cond_img_vid);
    encoder.set_tensor("raw_images_1", input_image_1);
    encoder.set_tensor("raw_images_2", input_image_2);
    encoder.set_tensor("resize_shape", target_shape);
    encoder.set_tensor("tile_shape", tile_shape);
    encoder.set_tensor("reshape_shape8d", reshape_shape8d);
    encoder.set_tensor("reshape_shape4d", reshape_shape4d);
    encoder.set_tensor("reshape_shape2d", reshape_shape2d);

    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();

    // Just avoid to multiple copy.
    if (frame_id == 0u) {
        auto out_shape = infer_output.get_shape();
        out_shape[0] = out_shape[0] * frame_num;
        out_tensor = ov::Tensor(infer_output.get_element_type(), out_shape);
    }

    std::memcpy(reinterpret_cast<uint8_t*>(out_tensor.data()) + frame_id * infer_output.get_byte_size(),
                infer_output.data(),
                infer_output.get_byte_size());
    out_rsz_size = ImageSize{grid_h, grid_w};
}

EncodedImage VisionEncoderQwen2VL::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    EncodedImage encoded_img;
    if (use_ov_image_preprocess == false) {
        encode_with_imagepreprocess_cpp({image}, config_map, encoded_img.resized_source, encoded_img.resized_source_size);
        return encoded_img;
    }
    encode_with_imagepreprocess_ov({image}, config_map, encoded_img.resized_source, encoded_img.resized_source_size);
    return encoded_img;
}

EncodedVideo VisionEncoderQwen2VL::encode_frames(const std::vector<ov::Tensor>& frames, const ov::AnyMap& config_map) {
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);
    EncodedVideo encoded_video;
    size_t i = 0;
    size_t image_num = frames.size();

    size_t frame_id = 0;
    encoded_video.frame_num = (image_num + config.temporal_patch_size - 1) / config.temporal_patch_size;

    using EncodeFunc = std::function<void(const std::vector<ov::Tensor>&, const ov::AnyMap&, ov::genai::EncodedVideo&, size_t, size_t)>;
    EncodeFunc encode_func;
    if (use_ov_image_preprocess == false) {
        encode_func = [this](const std::vector<ov::Tensor>& image, const ov::AnyMap& config_map, ov::genai::EncodedVideo& encoded_video, size_t frm_num, size_t frm_id) {
            this->encode_with_imagepreprocess_cpp(image, config_map, encoded_video.video_features, encoded_video.resized_source_size, frm_num, frm_id);
        };
    } else {
        encode_func = [this](const std::vector<ov::Tensor>& image, const ov::AnyMap& config_map, ov::genai::EncodedVideo& encoded_video, size_t frm_num, size_t frm_id) {
            this->encode_with_imagepreprocess_ov(image, config_map, encoded_video.video_features, encoded_video.resized_source_size, frm_num, frm_id);
        };
    }

    // Regarding Qwen-VL's video processing, it needs to merge `config.temporal_patch_size` adjacent frames for processing.
    // For video frames that are fewer than `config.temporal_patch_size`, they will be processed like images.
    for (; i + config.temporal_patch_size <= image_num; i += config.temporal_patch_size) {
        encode_func(std::vector<ov::Tensor>(frames.begin() + i, frames.begin() + i + config.temporal_patch_size),
                    config_map, encoded_video, encoded_video.frame_num, frame_id);
        frame_id++;
    }
    for (; i < image_num; i++) {
        encode_func({frames[i]}, config_map, encoded_video, encoded_video.frame_num, frame_id);
        frame_id++;
    }

    return encoded_video;
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

    encode_vision_placeholder_tokens();

    m_merge_length = std::pow(m_vision_encoder->get_processor_config().merge_size, 2);
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

    encode_vision_placeholder_tokens();

    m_merge_length = std::pow(m_vision_encoder->get_processor_config().merge_size, 2);
}

void InputsEmbedderQwen2VL::encode_vision_placeholder_tokens() {
    auto encoded_vision_tokens = m_tokenizer.encode(m_vlm_config.vision_start_token + m_vlm_config.vision_end_token +
                                                    m_vlm_config.image_pad_token + m_vlm_config.video_pad_token,
                                                    ov::genai::add_special_tokens(false));
    m_vision_token_ids["vision_start"] = encoded_vision_tokens.input_ids.data<int64_t>()[0];
    m_vision_token_ids["vision_end"] = encoded_vision_tokens.input_ids.data<int64_t>()[1];
    m_vision_token_ids["image_pad"] = encoded_vision_tokens.input_ids.data<int64_t>()[2];
    m_vision_token_ids["video_pad"] = encoded_vision_tokens.input_ids.data<int64_t>()[3];
}

size_t InputsEmbedderQwen2VL::calc_tokens_num(size_t grid_t, size_t grid_h, size_t grid_w) const {
    return grid_t * grid_h * grid_w / m_merge_length;
}

size_t InputsEmbedderQwen2VL::calc_vec_tokens_num(const std::vector<std::array<size_t, 3UL>>& vec_grid_thw) const {
    size_t token_num = 0;
    for (auto grid_thw : vec_grid_thw) {
        token_num += calc_tokens_num(grid_thw[0], grid_thw[1], grid_thw[2]);
    }
    return token_num;
};

NormalizedPrompt InputsEmbedderQwen2VL::normalize_prompt(const std::string& prompt,
                                                         size_t image_base_id,
                                                         size_t video_base_id,
                                                         const std::vector<EncodedImage>& images,
                                                         const std::vector<EncodedVideo>& videos) const {
    // Images
    auto [unified_prompt, images_sequence] = normalize(prompt, NATIVE_TAG, NATIVE_TAG, image_base_id, images.size());
    std::vector<std::array<size_t, 3>> images_grid_thw;
    images_grid_thw.reserve(images.size());

    for (const auto& encoded_image : images) {
        size_t grid_t = 1;
        size_t grid_h = encoded_image.resized_source_size.height;
        size_t grid_w = encoded_image.resized_source_size.width;
        images_grid_thw.push_back({grid_t, grid_h, grid_w});
    }

    for (size_t new_image_id : images_sequence) {
        auto [grid_t, grid_h, grid_w] = images_grid_thw.at(new_image_id - image_base_id);
        const size_t num_image_pad_tokens = calc_tokens_num(grid_t, grid_h, grid_w);

        std::string expanded_tag;
        expanded_tag.reserve(m_vlm_config.vision_start_token.length() +
                             m_vlm_config.image_pad_token.length() * num_image_pad_tokens +
                             m_vlm_config.vision_end_token.length());
        expanded_tag.append(m_vlm_config.vision_start_token);
        for (int i = 0; i < num_image_pad_tokens; ++i) {
            expanded_tag.append(m_vlm_config.image_pad_token);
        }
        expanded_tag.append(m_vlm_config.vision_end_token);

        unified_prompt.replace(unified_prompt.find(NATIVE_TAG), NATIVE_TAG.length(), expanded_tag);
    }

    // Video
    std::vector<size_t> videos_sequence;
    std::tie(unified_prompt, videos_sequence) =
        normalize(unified_prompt, NATIVE_VIDEO_TAG, NATIVE_VIDEO_TAG, video_base_id, videos.size());
    std::vector<std::array<size_t, 3>> video_grid_thw;
    video_grid_thw.reserve(videos.size());

    for (const auto& encoded_vd : videos) {
        size_t grid_t = encoded_vd.frame_num;
        OPENVINO_ASSERT(grid_t > 0, "Video input must contain at least one frame.");
        size_t grid_h = encoded_vd.resized_source_size.height;
        size_t grid_w = encoded_vd.resized_source_size.width;
        video_grid_thw.push_back({grid_t, grid_h, grid_w});
    }

    for (size_t new_image_id : videos_sequence) {
        auto [grid_t, grid_h, grid_w] = video_grid_thw.at(new_image_id - video_base_id);
        const size_t num_video_pad_tokens = calc_tokens_num(grid_t, grid_h, grid_w);

        std::string expanded_tag;
        expanded_tag.reserve(m_vlm_config.vision_start_token.length() +
                             m_vlm_config.video_pad_token.length() * num_video_pad_tokens +
                             m_vlm_config.vision_end_token.length());
        expanded_tag.append(m_vlm_config.vision_start_token);
        for (size_t i = 0; i < num_video_pad_tokens; ++i) {
            expanded_tag.append(m_vlm_config.video_pad_token);
        }
        expanded_tag.append(m_vlm_config.vision_end_token);

        unified_prompt.replace(unified_prompt.find(NATIVE_VIDEO_TAG), NATIVE_VIDEO_TAG.length(), expanded_tag);
    }

    return {std::move(unified_prompt), std::move(images_sequence), std::move(videos_sequence)};
}

ov::Tensor InputsEmbedderQwen2VL::get_inputs_embeds(const std::string& unified_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
    return get_inputs_embeds(unified_prompt, images, {}, metrics, recalculate_merged_embeddings, images_sequence, {});
}

ov::Tensor InputsEmbedderQwen2VL::get_inputs_embeds(const std::string& unified_prompt,
                                                    const std::vector<ov::genai::EncodedImage>& images,
                                                    const std::vector<ov::genai::EncodedVideo>& videos,
                                                    ov::genai::VLMPerfMetrics& metrics,
                                                    bool recalculate_merged_embeddings,
                                                    const std::vector<size_t>& images_sequence,
                                                    const std::vector<size_t>& videos_sequence,
                                                    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    std::vector<std::array<size_t, 3>> images_grid_thw;
    images_grid_thw.reserve(images.size());
    for (const auto& encoded_image : images) {
        size_t grid_t = 1;
        size_t grid_h = encoded_image.resized_source_size.height;
        size_t grid_w = encoded_image.resized_source_size.width;
        images_grid_thw.push_back({grid_t, grid_h, grid_w});
    }

    std::vector<std::array<size_t, 3>> video_grid_thw;
    video_grid_thw.reserve(videos.size());
    for (const auto& encoded_video : videos) {
        size_t grid_t = encoded_video.frame_num;
        OPENVINO_ASSERT(grid_t > 0, "Video input must contain at least one frame.");
        size_t grid_h = encoded_video.resized_source_size.height;
        size_t grid_w = encoded_video.resized_source_size.width;
        video_grid_thw.push_back({grid_t, grid_h, grid_w});
    }

    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    ov::Tensor text_embeds;
    {
        // Acquire request, run inference, then copy the result to safeguard against later reuse
        CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
        EmbeddingsRequest& req = embeddings_request_guard.get();
        ov::Tensor tmp_embeds = m_embedding->infer(req, input_ids);

        // Deep-copy necessary: Returned InferRequest's internal memory will be reused in
        // extract_text_features_for_cdpruner() that acquires a request from the same queue.
        // Without deep-copy, the second inference would overwrite this data, corrupting text_embeds.
        text_embeds = ov::Tensor(tmp_embeds.get_element_type(), tmp_embeds.get_shape());
        std::memcpy(text_embeds.data(), tmp_embeds.data(), tmp_embeds.get_byte_size());
    } // Request released here

    int64_t vision_start_token_id = m_vision_token_ids["vision_start"];
    int64_t vision_end_token_id = m_vision_token_ids["vision_end"];
    int64_t image_pad_token_id = m_vision_token_ids["image_pad"];
    int64_t video_pad_token_id = m_vision_token_ids["video_pad"];

    m_position_ids = create_position_ids(input_ids, images_grid_thw, images_sequence, 0, video_grid_thw, videos_sequence, 0, vision_start_token_id, history_vision_count);

    int64_t position_ids_max_element = *std::max_element(m_position_ids.data<int64_t>(), m_position_ids.data<int64_t>() + m_position_ids.get_size());
    m_rope_delta = position_ids_max_element + 1 - static_cast<int64_t>(input_ids.get_shape().at(1));

    if (images.empty() && videos.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }
    ov::Tensor merged_video_embeddings_tensor;
    ov::Tensor merged_image_embeddings_tensor;
    if (recalculate_merged_embeddings) {
        std::tie(m_merged_video_embeddings, m_merged_image_embeddings) = run_video_image_embeddings_merger(images, images_sequence, videos, videos_sequence);
    }
    merged_video_embeddings_tensor = m_merged_video_embeddings;
    merged_image_embeddings_tensor = m_merged_image_embeddings;

    // [CDPruner] Check if pruning is active and execute pipeline if applicable
    if (is_cdpruner_active(images)) {
        PruningResult pruning_result = execute_cdpruner_pipeline(input_ids,
                                                                 text_embeds,
                                                                 merged_image_embeddings_tensor,
                                                                 images,
                                                                 images_grid_thw,
                                                                 images_sequence,
                                                                 image_pad_token_id,
                                                                 vision_start_token_id,
                                                                 vision_end_token_id);

        return merge_text_and_image_embeddings_with_pruning(input_ids,
                                                            text_embeds,
                                                            pruning_result.pruned_embeddings,
                                                            image_pad_token_id,
                                                            vision_start_token_id,
                                                            vision_end_token_id,
                                                            pruning_result.keep_flags_per_region);
    }

    return qwen2_vl_utils::merge_text_and_video_image_embeddings(input_ids,
                                                                 text_embeds,
                                                                 merged_image_embeddings_tensor,
                                                                 merged_video_embeddings_tensor,
                                                                 image_pad_token_id,
                                                                 video_pad_token_id);
}

std::vector<ov::genai::EncodedVideo> InputsEmbedderQwen2VL::encode_videos(const std::vector<ov::Tensor>& videos) {
    std::vector<EncodedVideo> embeds;
    for (const ov::Tensor& single_video : videos) {
        std::vector<ov::Tensor> single_frames = to_single_image_tensors({single_video});
        auto encoded_video = m_vision_encoder->encode_frames(single_frames);
        embeds.emplace_back(encoded_video);
    }
    return embeds;
}

std::vector<ov::genai::EncodedImage> InputsEmbedderQwen2VL::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    for (ov::Tensor& image : single_images) {
        cvt_to_3_chn_image(image);
        embeds.emplace_back(m_vision_encoder->encode(image));
    }
    return embeds;
}

void InputsEmbedderQwen2VL::cvt_to_3_chn_image(ov::Tensor& image) {
    auto shape = image.get_shape();
    OPENVINO_ASSERT(shape.size() == 4);
    OPENVINO_ASSERT(image.get_element_type() == ov::element::u8);
    auto channels = image.get_shape().at(3);
    if (channels == 3) {
        return;
    } else if (channels == 1) {
        shape[3] = 3;
        auto new_img = ov::Tensor(image.get_element_type(), shape);

        const uint8_t* in_data = image.data<const uint8_t>();
        uint8_t* out_data = new_img.data<uint8_t>();

        for (size_t i = 0; i < shape[0] * shape[1] * shape[2]; ++i) {
            auto gray_val = *(in_data + i);
            *(out_data + i * 3 + 0) = gray_val;
            *(out_data + i * 3 + 1) = gray_val;
            *(out_data + i * 3 + 2) = gray_val;
        }
        image = new_img;
    } else if (channels == 4) {
        shape[3] = 3;
        auto new_img = ov::Tensor(image.get_element_type(), shape);

        const uint8_t* in_data = image.data<const uint8_t>();
        uint8_t* out_data = new_img.data<uint8_t>();

        for (size_t i = 0; i < shape[0] * shape[1] * shape[2]; ++i) {
            std::memcpy(out_data + i * 3, in_data + i * 4, 3 * sizeof(uint8_t));
        }
        image = new_img;
    }
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderQwen2VL::get_position_ids(const size_t inputs_embeds_size, const size_t history_size) {
    if (history_size != 0) {
        return get_generation_phase_position_ids(inputs_embeds_size, history_size, m_rope_delta);
    }
    return {m_position_ids, m_rope_delta};
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderQwen2VL::get_generation_phase_position_ids(const size_t inputs_embeds_size, const size_t history_size, int64_t rope_delta) {
    OPENVINO_ASSERT(history_size != 0, "get_generation_phase_position_ids() should only be called when history_size is non-zero (generation phase).");
    ov::Tensor position_ids{ov::element::i64, {3, 1, inputs_embeds_size}};
    int64_t new_pos_id = static_cast<int64_t>(history_size + rope_delta);
    for (size_t dim = 0; dim < 3; ++dim) {
        int64_t* pos_data = position_ids.data<int64_t>() + dim * inputs_embeds_size;
        std::iota(pos_data, pos_data + inputs_embeds_size, new_pos_id);
    }
    return {position_ids, rope_delta};
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
    m_merged_video_embeddings = ov::Tensor();
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderQwen2VL::run_video_image_embeddings_merger(
    const std::vector<EncodedImage>& images,
    const std::vector<size_t>& images_sequence,
    const std::vector<EncodedVideo>& videos,
    const std::vector<size_t>& videos_sequence
) {
    auto [reordered_image_embeds, reordered_images_grid_thw] = qwen2_vl_utils::reorder_image_embeds_and_grid_thw(images, images_sequence);
    auto [reordered_video_embeds, reordered_videos_grid_thw] = qwen2_vl_utils::reorder_video_embeds_and_grid_thw(videos, videos_sequence);

    ov::Tensor concatenated_embeds = qwen2_vl_utils::concatenate_video_image_embeds(reordered_video_embeds, reordered_image_embeds);

    std::vector<std::array<size_t, 3>> reordered_vision_grid_thw;
    reordered_vision_grid_thw.reserve(reordered_videos_grid_thw.size() + reordered_images_grid_thw.size());
    reordered_vision_grid_thw.insert(reordered_vision_grid_thw.end(), reordered_videos_grid_thw.begin(), reordered_videos_grid_thw.end());
    reordered_vision_grid_thw.insert(reordered_vision_grid_thw.end(), reordered_images_grid_thw.begin(), reordered_images_grid_thw.end());

    ov::Tensor rotary_pos_emb = get_rotary_pos_emb(reordered_vision_grid_thw);

    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_embeddings_merger.get());
    ov::InferRequest& vision_embeddings_merger = infer_request_guard.get();
    vision_embeddings_merger.set_tensor("hidden_states", concatenated_embeds);
    if (m_with_cu_seqlens_input) {
        ov::Tensor cu_seq_lens = qwen2_vl_utils::get_cu_seqlens(reordered_images_grid_thw, reordered_videos_grid_thw);
        vision_embeddings_merger.set_tensor("cu_seq_lens", cu_seq_lens);
    } else {
        ov::Tensor attention_mask = qwen2_vl_utils::get_attention_mask(reordered_images_grid_thw, reordered_videos_grid_thw);
        vision_embeddings_merger.set_tensor("attention_mask", attention_mask);
    }
    vision_embeddings_merger.set_tensor("rotary_pos_emb", rotary_pos_emb);
    vision_embeddings_merger.infer();
    ov::Tensor processed_vision_embeds = vision_embeddings_merger.get_output_tensor();

    auto out_vision_shape = processed_vision_embeds.get_shape();

    // Split Video and Image's features.
    auto video_fea_num = calc_vec_tokens_num(reordered_videos_grid_thw);
    auto image_fea_num = calc_vec_tokens_num(reordered_images_grid_thw);
    size_t video_fea_count = 0;
    if ((video_fea_num + image_fea_num) != 0) {
        video_fea_count = out_vision_shape.at(0) * video_fea_num / (video_fea_num + image_fea_num);
    }

    ov::Shape video_fea_shape = ov::Shape({video_fea_count, out_vision_shape.at(1)});
    ov::Tensor res_video = ov::Tensor(processed_vision_embeds.get_element_type(), video_fea_shape);
    OPENVINO_ASSERT(processed_vision_embeds.get_byte_size() >= res_video.get_byte_size(), "Vision embeds size should >= video embeds size.");
    std::memcpy(res_video.data(), processed_vision_embeds.data(), res_video.get_byte_size());

    ov::Shape image_fea_shape = ov::Shape({out_vision_shape.at(0) - video_fea_count, out_vision_shape.at(1)});
    ov::Tensor res_image = ov::Tensor(processed_vision_embeds.get_element_type(), image_fea_shape);
    OPENVINO_ASSERT(processed_vision_embeds.get_byte_size() == res_image.get_byte_size() + res_video.get_byte_size(),
                    "Vision embeds size should == image + video embeds size.");
    std::memcpy(res_image.data(), reinterpret_cast<uint8_t*>(processed_vision_embeds.data()) + res_video.get_byte_size(), res_image.get_byte_size());
    return {res_video, res_image};
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
    const std::vector<std::array<size_t, 3>>& videos_grid_thw,
    const std::vector<size_t>& videos_sequence,
    const size_t video_id,
    const int64_t vision_start_token_id,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    const size_t spatial_merge_size = m_vision_encoder->get_processor_config().merge_size;
    const size_t tokens_per_second = m_vlm_config.vision_config_tokens_per_second;
    std::vector<std::array<size_t, 3>> reordered_images_grid_thw;

    if (history_vision_count.size() > 0) {
        size_t vid_idx = 0;
        size_t img_idx = 0;
        for (size_t i = 0; i < history_vision_count.size(); i++) {
            size_t ed = vid_idx + history_vision_count[i].first;
            for (; vid_idx < ed; vid_idx++) {
                reordered_images_grid_thw.push_back(videos_grid_thw.at(vid_idx - video_id));
            }
            ed = img_idx + history_vision_count[i].second;
            for (; img_idx < ed; img_idx++) {
                reordered_images_grid_thw.push_back(images_grid_thw.at(img_idx - image_id));
            }
        }
    } else {
        for (size_t new_frame_id : videos_sequence) {
            reordered_images_grid_thw.push_back(videos_grid_thw.at(new_frame_id - video_id));
        }
        for (size_t new_image_id : images_sequence) {
            reordered_images_grid_thw.push_back(images_grid_thw.at(new_image_id - image_id));
        }
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
            size_t llm_grid_t = grid.at(0);
            size_t llm_grid_h = grid.at(1) / spatial_merge_size;
            size_t llm_grid_w = grid.at(2) / spatial_merge_size;
            size_t llm_grid_sz = llm_grid_h * llm_grid_w;
            size_t ed_image = ed + llm_grid_t * llm_grid_sz;

            // Fill temporal dimension
            for (size_t t = 0; t < llm_grid_t; t++) {
                std::fill_n(pos_data + ed + t * llm_grid_sz, llm_grid_sz, next_pos + t * tokens_per_second);
            }

            // Fill height and width dimensions
            int64_t* height_data = pos_data + seq_len + ed;
            int64_t* width_data = pos_data + 2 * seq_len + ed;
            for (size_t t = 0; t < llm_grid_t; t++) {
                size_t offset_sz = t * llm_grid_sz;
                for (size_t h = 0; h < llm_grid_h; ++h) {
                    size_t offset = h * llm_grid_w + offset_sz;
                    std::fill_n(height_data + offset, llm_grid_w, next_pos + h);
                    for (size_t w = 0; w < llm_grid_w; ++w) {
                        width_data[offset + w] = next_pos + w;
                    }
                }
            }

            next_pos += std::max(((llm_grid_t - 1) * tokens_per_second + 1), std::max(llm_grid_h, llm_grid_w));
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
                                                                       int64_t vision_end_token_id) const {
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

// [CDPruner] Override: Text feature extraction for relevance calculation
ov::Tensor InputsEmbedderQwen2VL::extract_text_features_for_pruning(const ov::Tensor& text_embeds,
                                                                    const ov::Tensor& input_ids,
                                                                    int64_t vision_start_token_id,
                                                                    int64_t vision_end_token_id) const {
    // Qwen2VL uses image_pad_token_id which we need to get from vision_token_ids
    int64_t image_pad_token_id = m_vision_token_ids.at("image_pad");

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
// [CDPruner] Override: Position encoding adjustment function for pruning
void InputsEmbedderQwen2VL::adjust_position_ids_after_pruning(
    ov::Tensor& position_ids_inout,
    const ov::Tensor& input_ids,
    int64_t vision_start_token_id,
    int64_t image_pad_token_id,
    const std::vector<std::array<size_t, 3>>& images_grid_thw,
    const std::vector<size_t>& images_sequence,
    std::vector<std::vector<bool>>& keep_flags_per_region_out) const {
    auto kept_indices_per_image = m_vision_encoder->get_last_selected_token_indices();
    OPENVINO_ASSERT(!images_sequence.empty(), "Image sequence must not be empty when pruning visual tokens");
    OPENVINO_ASSERT(!kept_indices_per_image.empty(), "Kept token indices are missing after pruning");

    const size_t spatial_merge_size = std::max<size_t>(1, m_vision_encoder->get_processor_config().merge_size);

    std::vector<std::array<size_t, 3>> reordered_images_grid_thw;
    reordered_images_grid_thw.reserve(images_sequence.size());
    for (size_t new_image_id : images_sequence) {
        OPENVINO_ASSERT(new_image_id < images_grid_thw.size(), "Image sequence index is out of range");
        // Indices are expected to be normalized to the current batch ordering.
        reordered_images_grid_thw.push_back(images_grid_thw.at(new_image_id));
    }

    // Directly modify position_ids_inout in-place
    position_ids_inout = update_position_ids(position_ids_inout,
                                             input_ids,
                                             vision_start_token_id,
                                             image_pad_token_id,
                                             reordered_images_grid_thw,
                                             kept_indices_per_image,
                                             spatial_merge_size,
                                             keep_flags_per_region_out);
}

ov::Tensor InputsEmbedderQwen2VL::update_position_ids(
    const ov::Tensor& original_position_ids,
    const ov::Tensor& input_ids,
    int64_t vision_start_token_id,
    int64_t image_pad_token_id,
    const std::vector<std::array<size_t, 3>>& reordered_images_grid_thw,
    const std::vector<std::vector<size_t>>& kept_indices_per_image,
    size_t spatial_merge_size,
    std::vector<std::vector<bool>>& keep_flags_out) const {
    const ov::Shape& pos_shape = original_position_ids.get_shape();
    OPENVINO_ASSERT(pos_shape.size() == 3 && pos_shape[0] == 3, "Position ids must be [3, batch, seq_len]");

    const size_t batch_size = pos_shape[1];
    const size_t seq_len = pos_shape[2];
    const size_t region_count = reordered_images_grid_thw.size();

    // Build region metadata
    struct RegionInfo {
        size_t tokens, grid_width, spatial_area, offset;
    };

    std::vector<RegionInfo> regions;
    regions.reserve(region_count);
    size_t cumulative_offset = 0;

    for (const auto& [grid_t, grid_h, grid_w] : reordered_images_grid_thw) {
        OPENVINO_ASSERT(grid_h % spatial_merge_size == 0 && grid_w % spatial_merge_size == 0,
                        "Grid dimensions must be divisible by spatial merge size");

        size_t llm_grid_h = grid_h / spatial_merge_size;
        size_t llm_grid_w = grid_w / spatial_merge_size;
        size_t spatial_area = llm_grid_h * llm_grid_w;
        OPENVINO_ASSERT(spatial_area > 0, "Vision region must contain at least one spatial token");

        size_t t = std::max<size_t>(1, grid_t);
        size_t total_tokens = spatial_area * t;
        regions.push_back({total_tokens, std::max<size_t>(1, llm_grid_w), spatial_area, cumulative_offset});
        cumulative_offset += total_tokens;
    }

    // Normalize kept indices: convert aggregated indices to per-region format if needed
    auto normalize_kept_indices = [&]() {
        if (kept_indices_per_image.empty())
            return std::vector<std::vector<size_t>>(region_count);

        if (kept_indices_per_image.size() == region_count)
            return kept_indices_per_image;

        // Handle single aggregated vector case
        OPENVINO_ASSERT(kept_indices_per_image.size() == 1 && region_count > 1,
                        "Kept token indices layout does not match vision regions");

        std::vector<std::vector<size_t>> normalized(region_count);
        for (size_t kept_idx : kept_indices_per_image[0]) {
            OPENVINO_ASSERT(kept_idx < cumulative_offset, "Aggregated kept index out of range");
            for (size_t img_idx = 0; img_idx < region_count; ++img_idx) {
                if (kept_idx >= regions[img_idx].offset &&
                    kept_idx < regions[img_idx].offset + regions[img_idx].tokens) {
                    normalized[img_idx].push_back(kept_idx - regions[img_idx].offset);
                    break;
                }
            }
        }
        return normalized;
    };

    auto normalized_indices = normalize_kept_indices();

    // Sort and deduplicate each region's indices
    for (auto& indices : normalized_indices) {
        std::sort(indices.begin(), indices.end());
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
    }

    // Build keep flags and calculate new sequence length
    keep_flags_out.clear();
    keep_flags_out.reserve(region_count);
    size_t total_removed = 0;

    for (size_t idx = 0; idx < region_count; ++idx) {
        std::vector<bool> flags(regions[idx].tokens, false);
        for (size_t kept_idx : normalized_indices[idx]) {
            OPENVINO_ASSERT(kept_idx < regions[idx].tokens, "Kept token index out of range");
            flags[kept_idx] = true;
        }
        size_t kept_count = std::count(flags.begin(), flags.end(), true);
        OPENVINO_ASSERT(kept_count <= regions[idx].tokens, "Kept tokens exceed region size");
        total_removed += regions[idx].tokens - kept_count;
        keep_flags_out.push_back(std::move(flags));
    }

    OPENVINO_ASSERT(seq_len >= total_removed, "Sequence length underflow after pruning");
    size_t new_seq_len = seq_len - total_removed;

    // Allocate new position IDs tensor
    ov::Tensor new_position_ids(original_position_ids.get_element_type(), {3, batch_size, new_seq_len});
    int64_t* pos_data[3] = {new_position_ids.data<int64_t>(),
                            new_position_ids.data<int64_t>() + batch_size * new_seq_len,
                            new_position_ids.data<int64_t>() + 2 * batch_size * new_seq_len};

    const int64_t* input_ids_data = input_ids.data<const int64_t>();

    // Process each batch
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        size_t write_idx = 0, image_idx = 0, visual_idx = 0;
        bool inside_vision = false;
        int64_t next_pos = 0, grid_base = 0;
        int64_t max_pos[3] = {-1, -1, -1};  // temporal, height, width
        size_t batch_offset = batch_idx * seq_len;
        size_t out_offset = batch_idx * new_seq_len;

        for (size_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
            int64_t token_id = input_ids_data[batch_offset + seq_idx];

            // Handle image pad tokens inside vision region
            if (inside_vision && token_id == image_pad_token_id) {
                OPENVINO_ASSERT(image_idx < region_count, "Vision region index out of bounds");
                if (keep_flags_out[image_idx][visual_idx]) {
                    const auto& region = regions[image_idx];
                    size_t local_idx = visual_idx % region.spatial_area;
                    size_t temporal_idx = visual_idx / region.spatial_area;
                    size_t row = local_idx / region.grid_width;
                    size_t col = local_idx % region.grid_width;

                    int64_t pos_vals[3] = {grid_base + static_cast<int64_t>(temporal_idx),
                                           grid_base + static_cast<int64_t>(row),
                                           grid_base + static_cast<int64_t>(col)};

                    for (int dim = 0; dim < 3; ++dim) {
                        pos_data[dim][out_offset + write_idx] = pos_vals[dim];
                        max_pos[dim] = std::max(max_pos[dim], pos_vals[dim]);
                    }
                    ++write_idx;
                }
                ++visual_idx;
                continue;
            }

            // Handle end of vision region
            if (inside_vision) {
                inside_vision = false;
                next_pos = std::max({max_pos[0], max_pos[1], max_pos[2]}) + 1;
                ++image_idx;
                visual_idx = 0;
                std::fill(max_pos, max_pos + 3, next_pos - 1);
            }

            // Write text token position
            for (int dim = 0; dim < 3; ++dim) {
                pos_data[dim][out_offset + write_idx] = next_pos;
            }
            ++write_idx;
            ++next_pos;

            // Handle start of vision region
            if (token_id == vision_start_token_id) {
                inside_vision = true;
                visual_idx = 0;
                grid_base = next_pos;
            }
        }

        OPENVINO_ASSERT(!inside_vision, "Unexpected end of sequence inside vision region");
        OPENVINO_ASSERT(image_idx == region_count, "Not all vision regions processed");
        OPENVINO_ASSERT(write_idx == new_seq_len, "Output sequence length mismatch");
    }

    return new_position_ids;
}

// [CDPruner] Override: Convert visual features to CDPruner format
std::vector<ov::Tensor> InputsEmbedderQwen2VL::convert_visual_features_for_pruning(
    const ov::Tensor& merged_image_embeddings,
    size_t chunk_count) const {
    // Convert from [num_patches, embedding_dim] to chunk_count * [1, num_patches, embedding_dim]
    ov::Shape original_shape = merged_image_embeddings.get_shape();
    size_t num_patches = original_shape[0];
    size_t embedding_dim = original_shape[1];
    size_t new_patches = num_patches / chunk_count;
    OPENVINO_ASSERT(original_shape[0] == new_patches * chunk_count, "Inconsistent number of patches per image");

    std::vector<ov::Tensor> visual_features;
    const float* src_data = merged_image_embeddings.data<const float>();
    size_t total_elements = new_patches * embedding_dim;
    for (size_t i = 0; i < chunk_count; i++) {
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
    const std::vector<std::vector<bool>>& keep_flags_per_region) const {
    auto text_embeds_shape = text_embeds.get_shape();
    size_t batch_size = text_embeds_shape.at(0);
    size_t original_seq_length = text_embeds_shape.at(1);
    size_t hidden_size = text_embeds_shape.at(2);

    size_t total_original_visual_tokens = 0;
    for (const auto& mask : keep_flags_per_region) {
        total_original_visual_tokens += mask.size();
    }

    size_t pruned_visual_tokens = pruned_vision_embeds.get_shape()[0];
    size_t tokens_removed = total_original_visual_tokens - pruned_visual_tokens;
    size_t new_seq_length = original_seq_length - tokens_removed;

    const size_t region_count = keep_flags_per_region.size();
    OPENVINO_ASSERT(region_count > 0, "Vision region metadata not available while merging embeddings");

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

// [CDPruner] Generate pruned input_ids based on keep_flags
ov::Tensor InputsEmbedderQwen2VL::generate_pruned_input_ids(const ov::Tensor& input_ids,
                                                            const std::vector<std::vector<bool>>& keep_flags_per_region,
                                                            int64_t image_pad_token_id,
                                                            int64_t vision_start_token_id,
                                                            int64_t vision_end_token_id) const {
    size_t original_seq_len = input_ids.get_shape().at(1);

    // Calculate total tokens to remove
    size_t tokens_to_remove = 0;
    for (const auto& mask : keep_flags_per_region) {
        tokens_to_remove += static_cast<size_t>(std::count(mask.begin(), mask.end(), false));
    }

    size_t new_sequence_length = original_seq_len - tokens_to_remove;
    ov::Tensor pruned_input_ids(ov::element::i64, {1, new_sequence_length});

    const int64_t* input_data = input_ids.data<const int64_t>();
    int64_t* pruned_data = pruned_input_ids.data<int64_t>();

    size_t write_idx = 0;
    bool inside_vision_region = false;
    size_t region_idx = 0;
    size_t pad_index = 0;
    size_t region_count = keep_flags_per_region.size();

    for (size_t seq_idx = 0; seq_idx < original_seq_len; ++seq_idx) {
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

        OPENVINO_ASSERT(write_idx < new_sequence_length, "Pruned input ids index exceeds expected sequence length");
        pruned_data[write_idx++] = token_id;

        if (inside_vision_region && token_id == vision_end_token_id) {
            const auto& keep_mask = keep_flags_per_region.at(region_idx);
            OPENVINO_ASSERT(pad_index == keep_mask.size(),
                            "Mismatch between consumed visual tokens and region metadata while pruning input ids");
            inside_vision_region = false;
            ++region_idx;
        }
    }

    OPENVINO_ASSERT(!inside_vision_region, "Unexpected end of sequence inside a vision region while pruning input ids");
    OPENVINO_ASSERT(region_idx == region_count, "Not all vision regions processed while generating pruned input ids");
    OPENVINO_ASSERT(write_idx == new_sequence_length, "Pruned input ids length mismatch after visual token pruning");

    return pruned_input_ids;
}

} // namespace ov::genai
