// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/minicpm/classes.hpp"

#include "visual_language/clip.hpp"

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
#include "openvino/op/constant.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/squeeze.hpp"

namespace ov::genai {

namespace {

std::string NATIVE_TAG = "<image>./</image>";

/**
* @brief Represents the result of slicing an image into smaller patches.
*
* This struct is used in miniCPM inputs embedder to store the sliced image patches
* and the target size of the processed image.
*
* @param slices A tensor containing the sliced image patches.
* @param target_size The desired size of the image after processing.
*/
struct ImageSliceResult {
    ov::Tensor slices;
    ImageSize target_size;
};

int ensure_divide(int length, int patch_size) {
    return std::max(static_cast<int>(std::round(static_cast<float>(length) / patch_size) * patch_size), patch_size);
}

std::pair<int, int> find_best_resize(std::pair<int, int> original_size, int scale_resolution, int patch_size, bool allow_upscale=false) {
    int width = original_size.first;
    int height = original_size.second;
    if ((width * height > scale_resolution * scale_resolution) || allow_upscale) {
        float r = static_cast<float>(width) / height;
        height = static_cast<int>(scale_resolution / std::sqrt(r));
        width = static_cast<int>(height * r);
    }
    int best_width = ensure_divide(width, patch_size);
    int best_height = ensure_divide(height, patch_size);
    return std::make_pair(best_width, best_height);
}

std::pair<int, int> get_refine_size(std::pair<int, int> original_size, std::pair<int, int> grid, int scale_resolution, int patch_size, bool allow_upscale) {
    int width, height;
    std::tie(width, height) = original_size;
    int grid_x, grid_y;
    std::tie(grid_x, grid_y) = grid;

    int refine_width = ensure_divide(width, grid_x);
    int refine_height = ensure_divide(height, grid_y);

    int grid_width = refine_width / grid_x;
    int grid_height = refine_height / grid_y;

    auto best_grid_size = find_best_resize(std::make_pair(grid_width, grid_height), scale_resolution, patch_size, allow_upscale);
    int best_grid_width, best_grid_height;
    std::tie(best_grid_width, best_grid_height) = best_grid_size;

    std::pair<int, int> refine_size = std::make_pair(best_grid_width * grid_x, best_grid_height * grid_y);
    return refine_size;
}

// Create a model that implements the image preprocessing operations
std::shared_ptr<ov::Model> patch_preprocess_into_model(std::shared_ptr<ov::Model> model_org) {
    // Input parameters for the preprocessing model
    auto raw_images = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{-1, -1, -1, -1});
    auto resize_target_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2});
    auto image_mean = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 1, 1});
    auto image_scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 1, 1});
    auto patch_size = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1});
    auto kernel_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2});

    // Set friendly names for inputs
    raw_images->set_friendly_name("raw_images");
    raw_images->output(0).get_tensor().set_names({"raw_images"});
    resize_target_shape->set_friendly_name("resize_target_shape");
    resize_target_shape->output(0).get_tensor().set_names({"resize_target_shape"});
    image_mean->set_friendly_name("image_mean");
    image_mean->output(0).get_tensor().set_names({"image_mean"});
    image_scale->set_friendly_name("image_scale");
    image_scale->output(0).get_tensor().set_names({"image_scale"});
    patch_size->set_friendly_name("patch_size");
    patch_size->output(0).get_tensor().set_names({"patch_size"});
    kernel_shape->set_friendly_name("kernel_shape");
    kernel_shape->output(0).get_tensor().set_names({"kernel_shape"});

    // Convert image to float32
    auto raw_images_f32 = std::make_shared<ov::op::v0::Convert>(raw_images, ov::element::f32);
    
    // Transpose from NHWC to NCHW
    auto img_trans = std::make_shared<ov::op::v1::Transpose>(raw_images_f32,
        std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{0, 3, 1, 2})
    );

    // Resize image using bicubic interpolation
    ov::op::v0::Interpolate::Attributes attrs = { };
    attrs.axes = {2, 3};
    attrs.mode = "cubic";
    attrs.antialias = true;
    attrs.align_corners = true;
    auto img_resized = std::make_shared<ov::op::v0::Interpolate>(img_trans, resize_target_shape, attrs);
    
    // Round to nearest even
    auto img_resized_rnd = std::make_shared<ov::op::v5::Round>(img_resized, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
    
    // Clamp values between 0 and 255
    auto resized_images_f32_planar = std::make_shared<ov::op::v0::Clamp>(img_resized_rnd, 0, 255);
    
    // Normalize image by subtracting mean and scaling
    auto resized_images_m = std::make_shared<ov::op::v1::Subtract>(resized_images_f32_planar, image_mean);
    auto resized_images_s = std::make_shared<ov::op::v1::Multiply>(resized_images_m, image_scale);

    // Extract shape information for unfold operation
    auto shape = std::make_shared<ov::op::v0::ShapeOf>(resized_images_s);
    auto batch_size = std::make_shared<ov::op::v0::Gather>(shape, 
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}),
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0})
    );
    auto channels = std::make_shared<ov::op::v0::Gather>(shape, 
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}),
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0})
    );
    auto height = std::make_shared<ov::op::v0::Gather>(shape, 
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2}),
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0})
    );
    auto width = std::make_shared<ov::op::v0::Gather>(shape, 
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{3}),
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0})
    );

    // Calculate output dimensions for unfold
    auto output_h = std::make_shared<ov::op::v1::Divide>(
        std::make_shared<ov::op::v1::Subtract>(height, patch_size),
        patch_size
    );
    auto output_h_add = std::make_shared<ov::op::v1::Add>(
        output_h,
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1})
    );
    auto output_w = std::make_shared<ov::op::v1::Divide>(
        std::make_shared<ov::op::v1::Subtract>(width, patch_size),
        patch_size
    );
    auto output_w_add = std::make_shared<ov::op::v1::Add>(
        output_w,
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1})
    );
    auto kernels_per_plane = std::make_shared<ov::op::v1::Multiply>(output_h_add, output_w_add);
    
    // Calculate new channel dimension for unfolded tensor
    auto patch_size_squared = std::make_shared<ov::op::v1::Multiply>(patch_size, patch_size);
    auto new_c = std::make_shared<ov::op::v1::Multiply>(channels, patch_size_squared);
    
    // Create unfold operation using extract_image_patches
    auto extract_patches = std::make_shared<ov::op::v3::ExtractImagePatches>(
        resized_images_s,
        kernel_shape,
        std::vector<int64_t>{patch_size->get_output_element_type(0) == ov::element::i64 ? 
                            static_cast<int64_t>(patch_size->get_output_shape(0)[0]) : 
                            static_cast<int64_t>(14)},
        std::vector<int64_t>{patch_size->get_output_element_type(0) == ov::element::i64 ? 
                            static_cast<int64_t>(patch_size->get_output_shape(0)[0]) : 
                            static_cast<int64_t>(14)},
        ov::op::PadType::VALID
    );
    
    // Reshape to match the expected output format
    auto reshape_to_2d = std::make_shared<ov::op::v1::Reshape>(
        extract_patches,
        std::make_shared<ov::op::v0::Concat>(
            std::vector<std::shared_ptr<ov::Node>>{
                batch_size,
                new_c,
                kernels_per_plane
            },
            0
        ),
        true
    );

    // Replace the original model's input with our preprocessing pipeline
    auto params_org = model_org->get_parameters();
    OPENVINO_ASSERT(params_org.size() == 1);
    ov::replace_node(params_org[0], reshape_to_2d);

    auto results = model_org->get_results();

    return std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{
            raw_images,
            resize_target_shape,
            image_mean,
            image_scale,
            patch_size,
            kernel_shape
        }
    );
}

// torch.bucketize(fractional_coords, boundaries, right=True)
std::vector<int64_t> bucket_size_right(const std::vector<float>& fractional_coords, const std::vector<float>& boundaries) {
    std::vector<int64_t> bucket_coords(fractional_coords.size());
    std::transform(fractional_coords.begin(), fractional_coords.end(), bucket_coords.begin(), [&boundaries](float fractional_coord) {
        return std::distance(boundaries.begin(), std::upper_bound(boundaries.begin(), boundaries.end(), fractional_coord));
    });
    return bucket_coords;
}

ov::Tensor prepare_vis_position_ids(
    const ov::Tensor& pixel_values,
    const ov::Tensor& patch_attention_mask,
    const std::vector<ImageSize> tgt_sizes,
    size_t patch_size,
    size_t num_patches_per_side) {
    size_t batch_size = pixel_values.get_shape().at(0);
    size_t max_im_h = pixel_values.get_shape().at(2), max_im_w = pixel_values.get_shape().at(3);
    size_t max_nb_patches_h = max_im_h / patch_size, max_nb_patches_w = max_im_w / patch_size;
    std::vector<float> boundaries(1.0f * num_patches_per_side - 1);
    std::generate(boundaries.begin(), boundaries.end(), [num_patches_per_side, val = 0.0f]() mutable {
        val += 1.0f / num_patches_per_side;
        return val;
    });
    size_t position_ids_batch_elem = max_nb_patches_h * max_nb_patches_w;
    ov::Tensor position_ids{ov::element::i64, {batch_size, position_ids_batch_elem}};
    int64_t* res_data = position_ids.data<int64_t>();
    std::fill_n(res_data, position_ids.get_size(), 0);

    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        size_t nb_patches_h = tgt_sizes.at(batch_idx).height;
        size_t nb_patches_w = tgt_sizes.at(batch_idx).width;

        std::vector<float> fractional_coords_h(nb_patches_h);
        std::generate(fractional_coords_h.begin(), fractional_coords_h.end(), [nb_patches_h, val = -1.0f / nb_patches_h]() mutable {
            val += 1.0f / nb_patches_h;
            return val;
        });
        std::vector<float> fractional_coords_w(nb_patches_w);
        std::generate(fractional_coords_w.begin(), fractional_coords_w.end(), [nb_patches_w, val = -1.0f / nb_patches_w]() mutable {
            val += 1.0f / nb_patches_w;
            return val;
        });

        std::vector<int64_t> bucket_coords_h = bucket_size_right(fractional_coords_h, boundaries);
        std::vector<int64_t> bucket_coords_w = bucket_size_right(fractional_coords_w, boundaries);

        std::vector<int64_t> pos_ids(bucket_coords_h.size() * bucket_coords_w.size());
        for (size_t col = 0; col < bucket_coords_h.size(); ++col) {
            for (size_t row = 0; row < bucket_coords_w.size(); ++row) {;
                pos_ids.at(col * bucket_coords_w.size() + row) = bucket_coords_h.at(col) * num_patches_per_side + bucket_coords_w.at(row);
            }
        }
        std::copy(pos_ids.begin(), pos_ids.end(), res_data + batch_idx * position_ids_batch_elem);
    }
    return position_ids;
}

std::pair<EncodedImage, ImageSliceResult> llava_image_embed_make_with_bytes_slice(clip_ctx& ctx_clip, const ov::Tensor& img, ov::InferRequest& encoder, int max_slice_nums, int scale_resolution, size_t patch_size, bool never_split) {
    // Extract image dimensions
    ov::Shape image_shape = img.get_shape();
    auto original_height = image_shape.at(1);
    auto original_width = image_shape.at(2);
    auto original_size = std::make_pair(original_width, original_height);
    
    // Calculate best resize dimensions
    auto best_size = find_best_resize(original_size, scale_resolution, patch_size, true);
    
    // Set up input tensors for the encoder
    ov::Tensor raw_images(ov::element::u8, image_shape, img.data<uint8_t>());
    
    uint64_t a_target_shape[2] = {static_cast<uint64_t>(best_size.second), static_cast<uint64_t>(best_size.first)};
    ov::Tensor target_shape(ov::element::i64, ov::Shape{2}, a_target_shape);
    
    std::vector<float> a_image_mean(ctx_clip.image_mean, ctx_clip.image_mean + 3);
    std::vector<float> a_image_scale(ctx_clip.image_std, ctx_clip.image_std + 3);
    for(auto& v : a_image_mean) v *= 255.0f;
    for(auto& v : a_image_scale) v = 1.0f / (v * 255.0f);
    
    ov::Tensor image_mean(ov::element::f32, ov::Shape{1, 3, 1, 1}, a_image_mean.data());
    ov::Tensor image_scale(ov::element::f32, ov::Shape{1, 3, 1, 1}, a_image_scale.data());
    
    uint64_t a_patch_size[1] = {patch_size};
    ov::Tensor patch_size_tensor(ov::element::i64, ov::Shape{1}, a_patch_size);
    
    uint64_t a_kernel_shape[2] = {patch_size, patch_size};
    ov::Tensor kernel_shape(ov::element::i64, ov::Shape{2}, a_kernel_shape);
    
    // Set input tensors to the encoder
    encoder.set_tensor("raw_images", raw_images);
    encoder.set_tensor("resize_target_shape", target_shape);
    encoder.set_tensor("image_mean", image_mean);
    encoder.set_tensor("image_scale", image_scale);
    encoder.set_tensor("patch_size", patch_size_tensor);
    encoder.set_tensor("kernel_shape", kernel_shape);
    
    // Run inference
    encoder.infer();
    
    // Get output tensor
    const ov::Tensor& output_tensor = encoder.get_output_tensor();
    
    // Create encoded image result
    ImageSize resized_source_size{best_size.second / patch_size, best_size.first / patch_size};
    ov::Tensor resized_source{ov::element::f32, output_tensor.get_shape()};
    output_tensor.copy_to(resized_source);
    
    // Create empty image slice result
    ImageSliceResult image_slice_result;
    
    return {{std::move(resized_source), resized_source_size}, std::move(image_slice_result)};
}

} // namespace

EncodedImage VisionEncoderMiniCPM::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    clip_ctx ctx_clip;
    ctx_clip.image_size = config.image_size;
    std::copy(config.norm_mean.begin(), config.norm_mean.end(), ctx_clip.image_mean);
    std::copy(config.norm_std.begin(), config.norm_std.end(), ctx_clip.image_std);

    auto [encoded_image, image_slice_result] = llava_image_embed_make_with_bytes_slice(ctx_clip, image, encoder, config.max_slice_nums, config.scale_resolution, config.patch_size, 0 == config.max_slice_nums);
    encoded_image.resampled_image = resample_encoded_image(encoded_image, image_slice_result.slices, image_slice_result.target_size);
    if (image_slice_result.slices) {
        encoded_image.slices_shape = image_slice_result.slices.get_shape();
    }
    return encoded_image;
}

ResampledImage VisionEncoderMiniCPM::resample_encoded_image(const EncodedImage& encoded_image, const ov::Tensor& slices, const ImageSize& target_size) {
    const ov::Tensor& resampled_source = resample(encoded_image.resized_source, {encoded_image.resized_source_size});
    std::vector<std::vector<ov::Tensor>> vision_embed_tensors;
    if (slices) {
        size_t token_idx = 0;
        const ov::Shape& slices_shape = slices.get_shape();
        vision_embed_tensors.resize(slices_shape.at(0));
        for (size_t i = 0; i < slices_shape.at(0); ++i) {
            std::vector<ov::Tensor> vision_embeds;
            vision_embeds.resize(slices_shape.at(1));
            for (size_t ja = 0; ja < slices_shape.at(1); ++ja) {
                size_t d2 = slices_shape.at(2);
                size_t d3 = slices_shape.at(3);
                ov::Tensor encoded_view{ov::element::f32, {1, d2, d3}, slices.data<float>() + (i * slices_shape.at(1) + ja) * d2 * d3};
                vision_embeds[ja] = resample(encoded_view, {target_size});
            }
            vision_embed_tensors[i] = vision_embeds;
        }
    }
    return {resampled_source, vision_embed_tensors};
}

namespace {

ov::Tensor concatenate_last_dim(const ov::Tensor& first, const ov::Tensor& second) {
    size_t res_d_0 = first.get_shape().at(0);
    size_t res_d_1 = first.get_shape().at(1);
    OPENVINO_ASSERT(second.get_shape().at(0) == res_d_0);
    OPENVINO_ASSERT(second.get_shape().at(1) == res_d_1);
    size_t res_d_2 = first.get_shape().at(2) + second.get_shape().at(2);
    ov::Tensor res{first.get_element_type(), {res_d_0, res_d_1, res_d_2}};
    auto first_data = first.data<float>();
    auto second_data = second.data<float>();
    float* res_data = res.data<float>();
    for (size_t i = 0; i < res_d_0; ++i) {
        for (size_t j = 0; j < res_d_1; ++j) {
            size_t k = 0;
            for (; k < first.get_shape().at(2); ++k) {
                res_data[i * res_d_1 * res_d_2 + j * res_d_2 + k]
                    = first_data[i * res_d_1 * first.get_shape().at(2) + j * first.get_shape().at(2) + k];
            }
            for (size_t l = 0; l < second.get_shape().at(2); ++l, ++k) {
                res_data[i * res_d_1 * res_d_2 + j * res_d_2 + k]
                    = second_data[i * res_d_1 * second.get_shape().at(2) + j * second.get_shape().at(2) + l];
            }
        }
    }
    return res;
}

/// embed_dim: output dimension for each position
/// pos: a list of positions to be encoded: size (H, W)
/// out: (H, W, D)
ov::Tensor get_1d_sincos_pos_embed_from_grid_new(size_t embed_dim, const ov::Tensor& pos) {
    OPENVINO_ASSERT(embed_dim % 2 == 0);
    ov::Shape pos_shape = pos.get_shape();
    size_t H = pos_shape[0];
    size_t W = pos_shape[1];

    std::vector<float> omega(embed_dim / 2);
    for (size_t i = 0; i < omega.size(); ++i) {
        omega[i] = 1.0f / std::pow(10000.0f, float(i) / (embed_dim / 2));
    }

    std::vector<size_t> out_shape = {H, W, embed_dim};
    ov::Tensor emb(ov::element::f32, out_shape);

    auto pos_data = pos.data<float>();
    auto emb_data = emb.data<float>();

    size_t counter = 0;
    for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
            for (size_t d = 0; d < embed_dim / 2; ++d) {
                // Correctly access the 2D position grid
                float value = omega[d] * pos_data[h * W + w];
                emb_data[h * W * embed_dim + w * embed_dim + d] = std::sin(value);
                emb_data[h * W * embed_dim + w * embed_dim + d + (embed_dim / 2)] = std::cos(value);
            }
        }
    }
    return emb;
}

ov::Tensor get_2d_sincos_pos_embed_from_grid(size_t embed_dim, const ov::Tensor& grid) {
    OPENVINO_ASSERT(embed_dim % 2 == 0);
    ov::Shape grid_shape = grid.get_shape();
    auto grid_data = grid.data<float>();
    ov::Shape plane_shape{grid_shape.at(1), grid_shape.at(2)};
    ov::Tensor emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, ov::Tensor{
        ov::element::f32,
        plane_shape,
        grid_data
    }); // (H, W, D/2)
    ov::Tensor emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, ov::Tensor{
        ov::element::f32,
        plane_shape,
        grid_data + plane_shape.at(0) * plane_shape.at(1)
    }); // (H, W, D/2)
    return concatenate_last_dim(emb_h, emb_w);
}

/// image_size: image_size or (image_height, image_width)
/// return:
/// pos_embed: [image_height, image_width, embed_dim]
ov::Tensor get_2d_sincos_pos_embed(size_t embed_dim, const ImageSize& image_size) {
    size_t grid_h_size = image_size.height, grid_w_size = image_size.width;
    ov::Tensor grid(ov::element::f32, {2, grid_h_size, grid_w_size});
    float* data = grid.data<float>();
    for (size_t y = 0; y < grid_h_size; ++y) {
        std::iota(data, data + grid_w_size, 0.0f);
        data += grid_w_size;
    }
for (float y = 0.0f; y < grid_h_size; ++y) {
        std::fill(data, data + grid_w_size, y);
        data += grid_w_size;
    }
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid);
}

void adjust_pos_cache(
    const std::vector<ImageSize>& target_sizes,
    size_t hidden_size,
    ov::Tensor& pos_embed_cache) {
    size_t max_h = std::max_element(target_sizes.begin(), target_sizes.end(), [](const ImageSize& left, const ImageSize& right) {
        return left.height < right.height;
    })->height;
    size_t max_w = std::max_element(target_sizes.begin(), target_sizes.end(), [](const ImageSize& left, const ImageSize& right) {
        return left.width < right.width;
    })->width;
    size_t allocated_height, allocated_width;
    if (pos_embed_cache) {
        const ov::Shape& allocated_shape = pos_embed_cache.get_shape();
        allocated_height = allocated_shape.at(0);
        allocated_width = allocated_shape.at(1);
    } else {
        allocated_height = allocated_width = 70;
    }
    if (max_h > allocated_height || max_w > allocated_width) {
        allocated_height = std::max(max_h, allocated_height);
        allocated_width = std::max(max_w, allocated_width);
        pos_embed_cache = get_2d_sincos_pos_embed(
            hidden_size, {allocated_height, allocated_width}
        );
    }
}

} // namespace

std::pair<std::string, std::vector<size_t>> InputsEmbedderMiniCPM::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {

    auto [unified_prompt, image_sequence] = normalize(
        prompt,
        NATIVE_TAG,
        '(' + NATIVE_TAG + ")\n",
        base_id,
        images.size()
    );
    std::string unk64;

    for (size_t idx = 0; idx < m_vlm_config.query_num; ++idx) {
        unk64 += m_vlm_config.unk;
    }
    for (size_t new_image_id : image_sequence) {
        const EncodedImage& encoded_image = images.at(new_image_id - base_id);
        std::string expanded_tag;
        if (m_vlm_config.use_image_id) {
            expanded_tag += m_vlm_config.im_id_start + std::to_string(new_image_id) + m_vlm_config.im_id_end;
        }
        expanded_tag += m_vlm_config.im_start + unk64 + m_vlm_config.im_end;
        ov::Shape slices_shape = encoded_image.slices_shape;
        if (slices_shape.size()) {
            for (size_t row_idx = 0; row_idx < slices_shape.at(0); ++row_idx) {
                for (size_t col_idx = 0; col_idx < slices_shape.at(1); ++col_idx) {
                    expanded_tag += m_vlm_config.slice_start + unk64 + m_vlm_config.slice_end;
                }
                expanded_tag += '\n';
            }
            expanded_tag.pop_back(); // Equivalent of python "\n".join(slices).
        }
        unified_prompt.replace(unified_prompt.find(NATIVE_TAG), NATIVE_TAG.length(), expanded_tag);
    }

    return {std::move(unified_prompt), std::move(image_sequence)};
}

ov::Tensor InputsEmbedderMiniCPM::get_inputs_embeds(const std::string& unified_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
    std::string unk64;
    ov::Tensor encoded_input = get_encoded_input_ids(unified_prompt, metrics);

    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor inputs_embeds = m_embedding->infer(req, encoded_input);
    OPENVINO_ASSERT(
        m_vlm_config.hidden_size == inputs_embeds.get_shape().at(2),
        "Unexpected embedding size"
    );
    auto start_tokenizer_time = std::chrono::steady_clock::now();
    ov::Tensor special_tokens = m_tokenizer.encode(
        m_vlm_config.im_start
        + m_vlm_config.im_end
        + m_vlm_config.slice_start
        + m_vlm_config.slice_end,
        ov::genai::add_special_tokens(false)
    ).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    OPENVINO_ASSERT(
        4 == special_tokens.get_shape().at(1),
        "Every special token must be represented with a single int."
    );
    int64_t im_start_id = special_tokens.data<int64_t>()[0];
    int64_t im_end_id = special_tokens.data<int64_t>()[1];
    int64_t slice_start_id = special_tokens.data<int64_t>()[2];
    int64_t slice_end_id = special_tokens.data<int64_t>()[3];
    int64_t im_start_pos = 0, slice_start_pos = 0;
    int64_t* begin = encoded_input.data<int64_t>();
    int64_t* ids = begin;
    size_t encoded_input_size = encoded_input.get_size();
    int64_t* end = ids + encoded_input_size;
    float* inputs_embeds_data = inputs_embeds.data<float>();
    for (size_t image_id : images_sequence) {
        const EncodedImage& encoded_image = images.at(image_id);
        const ov::Tensor& resampled_source = encoded_image.resampled_image.resampled_source;
        auto emb = resampled_source.data<float>();
        ids = std::find(ids, end, im_start_id);
        OPENVINO_ASSERT(end != ids);
        ++ids;
        std::copy_n(emb, resampled_source.get_size(), inputs_embeds_data + std::distance(begin, ids) * m_vlm_config.hidden_size);
        ids += m_vlm_config.query_num;
        ov::Shape slices_shape = encoded_image.slices_shape;
        if (slices_shape.size()) {
            size_t token_idx = 0;
            for (size_t i = 0; i < slices_shape.at(0); ++i) {
                for (size_t ja = 0; ja < slices_shape.at(1); ++ja) {
                    const ov::Tensor& vision_embed_tensor_i_j = encoded_image.resampled_image.vision_embed_tensors[i][ja];
                    ids = std::find(ids, end, slice_start_id);
                    OPENVINO_ASSERT(end != ids);
                    ++ids;
                    std::copy_n(vision_embed_tensor_i_j.data<float>(), vision_embed_tensor_i_j.get_size(), inputs_embeds_data + std::distance(begin, ids) * m_vlm_config.hidden_size);
                    ids += m_vlm_config.query_num;
                }
            }
        }
    }

    // inputs_embeds is bound to infer request that can be used by another thread after leaving this scope
    // so we need to return a copy to make sure data does not get corrupted
    ov::Tensor inputs_embeds_copy(inputs_embeds.get_element_type(), inputs_embeds.get_shape());
    std::memcpy(inputs_embeds_copy.data(), inputs_embeds.data(), inputs_embeds.get_byte_size());
    return inputs_embeds_copy;
}

ov::Tensor VisionEncoderMiniCPM::resample(const ov::Tensor& encoded_image, const std::vector<ImageSize>& target_sizes) {
    size_t bs = encoded_image.get_shape().at(0);
    std::vector<size_t> patch_len{target_sizes.size()};
    std::transform(target_sizes.begin(), target_sizes.end(), patch_len.begin(), [](const ImageSize& height_width) {
        return height_width.height * height_width.width;
    });
    adjust_pos_cache(
        target_sizes,
        m_vlm_config.hidden_size,
        m_pos_embed_cache
    );
    size_t max_patch_len = *std::max_element(patch_len.begin(), patch_len.end());
    ov::Tensor key_padding_mask(ov::element::f32, {bs, max_patch_len});
    float* mask_data = key_padding_mask.data<float>();
    size_t embed_len = m_pos_embed_cache.get_shape().at(2);
    ov::Tensor pos_embed(ov::element::f32, {max_patch_len, bs, embed_len}); // BLD => L * B * D
    float* pos_embed_data = pos_embed.data<float>();
    float* cache_data = m_pos_embed_cache.data<float>();
    size_t _d0 = m_pos_embed_cache.get_shape().at(0);
    size_t _d1 = m_pos_embed_cache.get_shape().at(1);
    for (size_t i = 0; i < bs; ++i) {
        size_t target_h = target_sizes.at(i).height;
        size_t target_w = target_sizes.at(i).width;
        for (size_t h_idx = 0; h_idx < target_h; ++h_idx) {
            for (size_t w_idx = 0; w_idx < target_w; ++w_idx) {
                std::copy_n(
                    cache_data + (h_idx * _d1 + w_idx) * embed_len,
                    embed_len,
                    pos_embed_data + (h_idx * target_w + w_idx) * bs * embed_len + i * embed_len
                );
            }
        }
        for (size_t flat = target_h * target_w; flat < max_patch_len; ++flat) {
            std::fill_n(pos_embed_data + flat * bs * embed_len + i * embed_len, embed_len, 0.0f);
        }
        std::fill_n(mask_data + i * max_patch_len, patch_len[i], 0.0f);
        std::fill_n(mask_data + i * max_patch_len + patch_len[i], max_patch_len - patch_len[i], 1.0f);
    }
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_resampler.get());
    ov::InferRequest& resampler = infer_request_guard.get();
    resampler.set_tensor("image_feature", encoded_image); // [N, H*W, old_hidden_size]
    resampler.set_tensor("pos_embed", pos_embed); // [H*W, N, new_hidden_size]
    resampler.set_tensor("key_padding_mask", key_padding_mask); // [N, H*W]
    resampler.infer();
    auto resampler_out = resampler.get_output_tensor();
    // resampler_out is bound to infer request and the data may become corrupted after next resampler inference
    // so we need to return a copy to make sure data does not get corrupted
    ov::Tensor res(resampler_out.get_element_type(), resampler_out.get_shape());
    std::memcpy(res.data(), resampler_out.data(), resampler_out.get_byte_size());
    return res; // [N, query_num, new_hidden_size]
}

VisionEncoderMiniCPM::VisionEncoderMiniCPM(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap properties) : VisionEncoder{model_dir, device, properties} {
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_dir, "config.json");
    
    // Read and patch the vision encoder model
    auto model_org = utils::singleton_core().read_model(model_dir / "openvino_vision_embeddings_model.xml");
    auto model = patch_preprocess_into_model(model_org);
    auto compiled_model = utils::singleton_core().compile_model(model, device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM vision embeddings model");
    m_ireq_queue_vision_encoder = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    
    // Load and compile the resampler model
    auto compiled_resampler_model = utils::singleton_core().compile_model(model_dir / "openvino_resampler_model.xml", device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_resampler_model, "VLM resampler model");
    m_ireq_queue_resampler = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_resampler_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_resampler_model]() -> ov::InferRequest {
            return compiled_resampler_model.create_infer_request();
        });
    m_pos_embed_cache = get_2d_sincos_pos_embed(m_vlm_config.hidden_size, {70, 70});
}

VisionEncoderMiniCPM::VisionEncoderMiniCPM(
    const ModelsMap& models_map,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) : VisionEncoder{models_map, config_dir_path, device, device_config} {
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");
    
    // Read and patch the vision encoder model
    const auto& vision_encoder_model = utils::get_model_weights_pair(models_map, "vision_embeddings").first;
    const auto& vision_encoder_weights = utils::get_model_weights_pair(models_map, "vision_embeddings").second;
    auto model_org = utils::singleton_core().read_model(vision_encoder_model, vision_encoder_weights);
    auto model = patch_preprocess_into_model(model_org);
    auto compiled_model = utils::singleton_core().compile_model(model, device, device_config);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM vision embeddings model");
    m_ireq_queue_vision_encoder = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    
    // Load and compile the resampler model
    const auto& resampler_model = utils::get_model_weights_pair(models_map, "resampler").first;
    const auto& resampler_weights = utils::get_model_weights_pair(models_map, "resampler").second;
    auto compiled_resampler_model = utils::singleton_core().compile_model(resampler_model, resampler_weights, device, device_config);
    ov::genai::utils::print_compiled_model_properties(compiled_resampler_model, "VLM resampler model");
    m_ireq_queue_resampler = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_resampler_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_resampler_model]() -> ov::InferRequest {
            return compiled_resampler_model.create_infer_request();
        });
    m_pos_embed_cache = get_2d_sincos_pos_embed(m_vlm_config.hidden_size, {70, 70});
}

InputsEmbedderMiniCPM::InputsEmbedderMiniCPM(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, device, device_config) {}

InputsEmbedderMiniCPM::InputsEmbedderMiniCPM(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

} // namespace ov::genai