
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/minicpm/classes.hpp"

#include "visual_language/clip.hpp"
#include "visual_language/vision_encoder.hpp"

#include "utils.hpp"

namespace ov::genai {

namespace {

std::string NATIVE_TAG = "<image>./</image>";
constexpr char VIDEO_GROUP_MARKER[] = "<ov_genai_minicpm_video_group>";

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

std::vector<std::vector<clip_image_u8>> slice_image(const clip_image_u8& img, const int max_slice_nums, const int scale_resolution, const int patch_size, const bool never_split) {
    const std::pair<int, int> original_size{img.nx, img.ny};
    const int original_width = img.nx;
    const int original_height = img.ny;
    const float log_ratio = logf(1.0f * original_width / original_height);
    const float ratio = 1.0f * original_width * original_height / (scale_resolution * scale_resolution);
    const int multiple = std::min(int(ceil(ratio)), max_slice_nums);

    std::vector<std::vector<clip_image_u8>> images;
    images.push_back(std::vector<clip_image_u8>{});

    if (multiple <= 1) {
        auto best_size = find_best_resize(original_size, scale_resolution, patch_size, true);
        images.back().push_back(clip_image_u8{});
        bicubic_resize(img, images.back().back(), best_size.first, best_size.second);
    }
    else if (multiple > 1) {

        std::vector<int> candidate_split_grids_nums;
        for (int i : {multiple - 1, multiple, multiple + 1}) {
            if (i == 1 || i > max_slice_nums) {
                continue;
            }
            candidate_split_grids_nums.push_back(i);
        }

        auto best_size = find_best_resize(original_size, scale_resolution, patch_size);
        images.back().push_back(clip_image_u8{});
        bicubic_resize(img, images.back().back(), best_size.first, best_size.second);

        std::vector<std::pair<int, int>> candidate_grids;

        for (int split_grids_nums : candidate_split_grids_nums) {
            int m = 1;
            while (m <= split_grids_nums) {
                if (split_grids_nums % m == 0) {
                    candidate_grids.emplace_back(m, split_grids_nums / m);
                }
                ++m;
            }
        }

        std::pair<int, int> best_grid{ 1, 1 };
        float min_error = std::numeric_limits<float>::infinity();

        for (const auto& grid : candidate_grids) {
            float error = std::abs(log_ratio - std::log(1.0f * grid.first / grid.second));
            if (error < min_error) {
                best_grid = grid;
                min_error = error;
            }
        }
        auto refine_size = get_refine_size(original_size, best_grid, scale_resolution, patch_size, true);
        clip_image_u8 refine_image;
        bicubic_resize(img, refine_image, refine_size.first, refine_size.second);

        // split_to_patches
        int width = refine_image.nx;
        int height = refine_image.ny;
        int grid_x = int(width / best_grid.first);
        int grid_y = int(height / best_grid.second);
        for (int patches_i = 0, ic = 0; patches_i < height && ic < best_grid.second; patches_i += grid_y, ic += 1) {
            images.push_back(std::vector<clip_image_u8>{});
            for (int patches_j = 0, jc = 0; patches_j < width && jc < best_grid.first; patches_j += grid_x, jc += 1) {
                images.back().push_back(clip_image_u8{});
                clip_image_u8& patch = images.back().back();
                patch.nx = grid_x;
                patch.ny = grid_y;
                patch.buf.resize(3 * patch.nx * patch.ny);
                for (int y = patches_i; y < patches_i + grid_y; ++y) {
                    for (int x = patches_j; x < patches_j + grid_x; ++x) {
                        const int i = 3 * (y * refine_image.nx + x);
                        const int j = 3 * ((y - patches_i) * patch.nx + (x - patches_j));
                        patch.buf[j] = refine_image.buf[i];
                        patch.buf[j + 1] = refine_image.buf[i + 1];
                        patch.buf[j + 2] = refine_image.buf[i + 2];
                    }
                }
            }
        }
    }

    return images;
}

// Reimplemented https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold
// in shape [NCHW], out shape: [N, C*kernel*kernel, H*W/kernel/kernel]
ov::Tensor unfold(const ov::Tensor& images_tensor, size_t kernel) {
    ov::Shape images_shape = images_tensor.get_shape();

    OPENVINO_ASSERT(4 == images_shape.size(), "Input tensor must be 4D (NCHW).");

    const size_t bs = images_shape.at(0);
    const size_t images_c = images_shape.at(1);
    const size_t images_h = images_shape.at(2);
    const size_t images_w = images_shape.at(3);

    OPENVINO_ASSERT(images_h >= kernel && images_w >= kernel, "Input height and width must be greater than or equal to kernel size.");

    const size_t new_c = images_c * kernel * kernel;
    const size_t output_h = (images_h - kernel) / kernel + 1;
    const size_t output_w = (images_w - kernel) / kernel + 1;
    const size_t kernels_per_plane = output_h * output_w;

    ov::Tensor unfolded_tensor(ov::element::f32, {bs, new_c, kernels_per_plane});
    const float* images = images_tensor.data<float>();
    float* unfolded = unfolded_tensor.data<float>();
    for (size_t batch_idx = 0; batch_idx < bs; ++batch_idx) {
        for (size_t c_idx = 0; c_idx < images_c; ++c_idx) {
            for (size_t h_out = 0; h_out < output_h; ++h_out) {
                for (size_t w_out = 0; w_out < output_w; ++w_out) {
                    size_t h_idx = h_out * kernel;  // Calculate input height index
                    size_t w_idx = w_out * kernel;  // Calculate input width index

                    for (size_t kh = 0; kh < kernel; ++kh) {
                        for (size_t kw = 0; kw < kernel; ++kw) {
                            size_t input_idx = (batch_idx * images_c * images_h * images_w) +
                                                (c_idx * images_h * images_w) +
                                                ((h_idx + kh) * images_w) +
                                                (w_idx + kw);

                            size_t unfolded_c_idx = (c_idx * kernel * kernel) + (kh * kernel) + kw;
                            size_t unfolded_idx = (batch_idx * new_c * kernels_per_plane) +
                                                    unfolded_c_idx * kernels_per_plane +
                                                    (h_out * output_w + w_out);

                            unfolded[unfolded_idx] = images[input_idx];
                        }
                    }
                }
            }
        }
    }
    return unfolded_tensor;
}

ov::Tensor preprocess_for_encoder(const ov::Tensor& images, size_t kernel) {
    ov::Shape images_shape = images.get_shape();
    OPENVINO_ASSERT(4 == images_shape.size());
    ov::Tensor unfolded_tensor = unfold(images, kernel);
    const ov::Shape& unfolded_shape = unfolded_tensor.get_shape();  // [N, C*kernel*kernel, H*W/kernel/kernel]
    const size_t bs = unfolded_shape[0];
    const size_t d1 = unfolded_shape[1];
    const size_t d2 = unfolded_shape[2];
    const size_t channels = 3;
    const size_t new_len = d2 * kernel;

    ov::Tensor permuted_tensor{ov::element::f32, {bs, channels, kernel, new_len}};
    const float* unfolded = unfolded_tensor.data<float>();
    float* permuted = permuted_tensor.data<float>();
    for (size_t b_idx = 0; b_idx < bs; ++b_idx) {
        for (size_t c_idx = 0; c_idx < channels; ++c_idx) {
            for (size_t k1_idx = 0; k1_idx < kernel; ++k1_idx) {
                for (size_t d2_idx = 0; d2_idx < d2; ++d2_idx) {
                    for (size_t k2_idx = 0; k2_idx < kernel; ++k2_idx) {
                        size_t unfolded_idx = b_idx * d1 * d2 +
                                            (c_idx * kernel * kernel + k1_idx * kernel + k2_idx) * d2 +
                                            d2_idx;
                        size_t permuted_idx = b_idx * channels * kernel * new_len +
                                            c_idx * kernel * new_len +
                                            k1_idx * new_len +
                                            d2_idx * kernel + k2_idx;
                        permuted[permuted_idx] = unfolded[unfolded_idx];
                    }
                }
            }
        }
    }
    return permuted_tensor;
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
    std::vector<float> boundaries(num_patches_per_side - 1);
    std::iota(boundaries.begin(), boundaries.end(), 1.0f);
    std::transform(boundaries.begin(), boundaries.end(), boundaries.begin(), [num_patches_per_side](float val) {
        return val / num_patches_per_side;
    });
    size_t position_ids_batch_elem = max_nb_patches_h * max_nb_patches_w;
    ov::Tensor position_ids{ov::element::i64, {batch_size, position_ids_batch_elem}};
    int64_t* res_data = position_ids.data<int64_t>();
    std::fill_n(res_data, position_ids.get_size(), 0);

    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        size_t nb_patches_h = tgt_sizes.at(batch_idx).height;
        size_t nb_patches_w = tgt_sizes.at(batch_idx).width;

        std::vector<float> fractional_coords_h(nb_patches_h);
        std::iota(fractional_coords_h.begin(), fractional_coords_h.end(), 0.0f);
        std::transform(fractional_coords_h.begin(), fractional_coords_h.end(), fractional_coords_h.begin(), [nb_patches_h](float val) {
            return val / nb_patches_h;
        });
        std::vector<float> fractional_coords_w(nb_patches_w);
        std::iota(fractional_coords_w.begin(), fractional_coords_w.end(), 0.0f);
        std::transform(fractional_coords_w.begin(), fractional_coords_w.end(), fractional_coords_w.begin(), [nb_patches_w](float val) {
            return val / nb_patches_w;
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
    clip_image_u8 source = tensor_to_clip_image_u8(img);
    std::vector<std::vector<clip_image_u8>> imgs = slice_image(source, max_slice_nums, scale_resolution, patch_size, never_split);
    const size_t channels = 3;

    std::vector<std::vector<clip_image_f32>> preprocessed{imgs.size()};
    size_t n_images = 0, max_size = 0;
    std::transform(imgs.begin(), imgs.end(), preprocessed.begin(), [&ctx_clip, &max_size, &n_images](const std::vector<clip_image_u8>& row) {
        std::vector<clip_image_f32> processed_row{row.size()};
        std::transform(row.begin(), row.end(), processed_row.begin(), [&ctx_clip, &max_size, &n_images](const clip_image_u8& raw) {
            clip_image_f32 im = clip_image_preprocess(ctx_clip, raw);
            if (size_t(im.ny) * size_t(im.nx) > max_size) {
                max_size = size_t(im.ny) * size_t(im.nx);
            }
            ++n_images;
            return im;
        });
        return processed_row;
    });

    ov::Tensor pixel_values{ov::element::f32, {n_images, channels, patch_size, max_size / patch_size}};
    size_t d3_all_pixel = pixel_values.get_shape().at(3);
    float* pixel_value_data = pixel_values.data<float>();

    //image chw to 1*c*kernel*hw/kernel and padding zero
    clip_image_f32& resized_preprocessed = preprocessed.at(0).at(0);
    size_t img_h = resized_preprocessed.ny;
    size_t img_w = resized_preprocessed.nx;
    ov::Tensor clip_img{ov::element::f32, {1, channels, img_h, img_w}, resized_preprocessed.buf.data()};
    ov::Tensor clip_pixel_values = preprocess_for_encoder(clip_img, patch_size);

    float* clip_value_data = clip_pixel_values.data<float>();
    size_t batch_pixel = 1;
    size_t d3_clip_pixel = clip_pixel_values.get_shape().at(3);
    for (size_t c_idx = 0; c_idx < channels; ++c_idx) {
        for (size_t k_idx = 0; k_idx < patch_size; k_idx++) {
            std::copy(clip_value_data, clip_value_data + d3_clip_pixel, pixel_value_data);
            // pixel_values and patch_attention_mask are multiplied instead of ignoring the corresponding pixel_values resulting in NaN instead of 0.0f if pixel_values has NaN.
            memset(pixel_value_data + d3_clip_pixel, 0, (d3_all_pixel - d3_clip_pixel) * sizeof(float));
            clip_value_data += d3_clip_pixel;
            pixel_value_data += d3_all_pixel;
        }
    }

    if (1 < preprocessed.size()) {
        for (size_t row = 1; row < preprocessed.size(); ++row) {
            size_t n_slices = preprocessed.at(row).size();
            for (size_t col = 0; col < n_slices; ++col) {
                clip_image_f32& elem = preprocessed.at(row).at(col);
                img_h = elem.ny;
                img_w = elem.nx;
                ov::Tensor clip_img{ov::element::f32, {1, channels, img_h, img_w}, elem.buf.data()};
                ov::Tensor clip_pixel_values = preprocess_for_encoder(clip_img, patch_size);

                d3_clip_pixel = clip_pixel_values.get_shape().at(3);
                clip_value_data = clip_pixel_values.data<float>();
                pixel_value_data = pixel_values.data<float>() + batch_pixel * channels * patch_size * d3_all_pixel;
                for (size_t c_idx = 0; c_idx < channels; ++c_idx) {
                    for (size_t k_idx = 0; k_idx < patch_size; k_idx++) {
                        std::copy(clip_value_data, clip_value_data + d3_clip_pixel, pixel_value_data);
                        // pixel_values and patch_attention_mask are multiplied instead of ignoring the corresponding pixel_values resulting in NaN instead of 0.0f if pixel_values has NaN.
                        memset(pixel_value_data + d3_clip_pixel, 0, (d3_all_pixel - d3_clip_pixel) * sizeof(float));
                        clip_value_data += d3_clip_pixel;
                        pixel_value_data += d3_all_pixel;
                    }
                }
                batch_pixel++;
            }
        }
    }
    encoder.set_tensor("pixel_values", pixel_values);

    ov::Tensor patch_attention_mask{ov::element::f32, {pixel_values.get_shape().at(0), 1, max_size / patch_size / patch_size}};
    float* attention_data = patch_attention_mask.data<float>();
    std::fill_n(attention_data, patch_attention_mask.get_size(), 0.0f);
    std::fill_n(attention_data, resized_preprocessed.ny / patch_size * resized_preprocessed.nx / patch_size, 1.0f);
    if (1 < preprocessed.size()) {
        for (size_t row = 1; row < preprocessed.size(); ++row) {
            size_t n_slices = preprocessed.at(row).size();
            for (size_t col = 0; col < n_slices; ++col) {
                const clip_image_f32& elem = preprocessed.at(row).at(col);
                std::fill_n(attention_data + ((row - 1) * n_slices + col + 1) * max_size / patch_size / patch_size, elem.ny / patch_size * elem.nx / patch_size, 1.0f);
            }
        }
    }
    encoder.set_tensor("patch_attention_mask", patch_attention_mask);

    ImageSize resized_source_size{resized_preprocessed.ny / patch_size, resized_preprocessed.nx / patch_size};
    std::vector<ImageSize> tgt_sizes{resized_source_size};
    if (1 < preprocessed.size()) {
        for (auto row = preprocessed.begin() + 1; row != preprocessed.end(); ++row) {
            for (const clip_image_f32& elem : *row) {
                tgt_sizes.push_back({elem.ny / patch_size, elem.nx / patch_size});
            }
        }
    }
    ImageSliceResult image_slice_result;
    ov::Tensor position_ids = prepare_vis_position_ids(pixel_values, patch_attention_mask, tgt_sizes, patch_size, ctx_clip.image_size / patch_size);
    encoder.set_tensor("position_ids", position_ids);
    encoder.infer();
    const ov::Tensor& output_tensor = encoder.get_output_tensor();

    if (1 == preprocessed.size()) {
        ov::Tensor resized_source{ov::element::f32, output_tensor.get_shape()};
        output_tensor.copy_to(resized_source);
        return {{std::move(resized_source), resized_source_size}, std::move(image_slice_result)};
    }

    size_t old_hidden_size = output_tensor.get_shape().at(2);
    const float* out = output_tensor.data<float>();
    size_t n_patches = max_size / patch_size / patch_size;
    ov::Tensor resized_source{ov::element::f32, {1, n_patches, old_hidden_size}};
    std::copy_n(out, resized_source.get_size(), resized_source.data<float>());

    image_slice_result.slices = ov::Tensor{ov::element::f32, {preprocessed.size() - 1, preprocessed.at(1).size(), n_patches, old_hidden_size}};
    for (size_t col = 0; col < preprocessed.size() - 1; ++col) {
        for (size_t row = 0; row < preprocessed.at(1).size(); ++row) {
            std::copy_n(out + (col * preprocessed.at(1).size() + row + 1) * n_patches * old_hidden_size, n_patches * old_hidden_size, image_slice_result.slices.data<float>() + (col * preprocessed.at(1).size() + row) * n_patches * old_hidden_size);
        }
    }
    image_slice_result.target_size = tgt_sizes.at(1);
    return {{std::move(resized_source), resized_source_size}, std::move(image_slice_result)};
}

} // namespace

EncodedImage VisionEncoderMiniCPM::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);

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

ov::Tensor VisionEncoderMiniCPM::resample_image(const EncodedImage& image, size_t pad_to_max) {
    return resample(image.resized_source, image.resized_source_size, pad_to_max);
}

ResampledImage VisionEncoderMiniCPM::resample_encoded_image(const EncodedImage& encoded_image, const ov::Tensor& slices, const ImageSize& target_size) {
    size_t pad_to_max = encoded_image.resized_source_size.height * encoded_image.resized_source_size.width;
    if (slices) {
        pad_to_max = std::max(pad_to_max, target_size.height * target_size.width);
    }
    const ov::Tensor& resampled_source = resample(encoded_image.resized_source, encoded_image.resized_source_size, pad_to_max);
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
                // const_cast is safe as ov::Tensor only views the data and doesn't modify it.
                ov::Tensor encoded_view{
                    ov::element::f32, 
                    {1, d2, d3}, 
                    const_cast<float*>(slices.data<float>()) + (i * slices_shape.at(1) + ja) * d2 * d3
                };
                vision_embeds[ja] = resample(encoded_view, target_size, pad_to_max);
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

    for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
            for (size_t d = 0; d < embed_dim / 2; ++d) {
                float value = omega[d] * pos_data[h * W + w];
                // sin() and cos() are the source of difference with Python until a newer C++ standard is used.
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
    });  // (H, W, D/2)
    ov::Tensor emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, ov::Tensor{
        ov::element::f32,
        plane_shape,
        grid_data + plane_shape.at(0) * plane_shape.at(1)
    });  // (H, W, D/2)
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

NormalizedPrompt InputsEmbedderMiniCPM::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    
    auto [unified_prompt, image_sequence] = normalize(
        prompt,
        NATIVE_TAG,
        NATIVE_TAG + '\n',
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
            expanded_tag.pop_back();  // Equivalent of python "\n".join(slices).
        }
        unified_prompt.replace(unified_prompt.find(NATIVE_TAG), NATIVE_TAG.length(), expanded_tag);
    }

    return {std::move(unified_prompt), std::move(image_sequence), {}};
}

std::vector<EncodedImage> InputsEmbedderMiniCPM::encode_images(
    const std::vector<ov::Tensor>& images,
    bool has_video_inputs
) {
    if (!has_video_inputs) {
        return IInputsEmbedder::encode_images(images);
    }

    std::vector<EncodedImage> encoded_images;
    const std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    encoded_images.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        encoded_images.emplace_back(m_vision_encoder->encode(image, {{"max_slice_nums", 1}}));
    }
    OPENVINO_ASSERT(images.size() == encoded_images.size(), "Input images size and encoded images size mismatch!");
    return encoded_images;
}

NormalizedPrompt InputsEmbedderMiniCPM::normalize_prompt(const std::string& prompt,
                                                         size_t base_image_id,
                                                         size_t base_video_id,
                                                         const std::vector<EncodedImage>& images,
                                                         const std::vector<EncodedVideo>& videos) const {
    if (videos.empty()) {
        return normalize_prompt(prompt, base_image_id, images);
    }

    auto [unified_prompt, video_sequence] = normalize(
        prompt,
        NATIVE_TAG,
        NATIVE_TAG + '\n',
        base_video_id,
        videos.size(),
        ModalityType::VIDEO
    );

    std::string unk64;
    unk64.reserve(m_vlm_config.query_num * m_vlm_config.unk.size());
    for (size_t idx = 0; idx < m_vlm_config.query_num; ++idx) {
        unk64 += m_vlm_config.unk;
    }

    size_t search_offset = 0;
    for (size_t video_id : video_sequence) {
        const EncodedVideo& encoded_video = videos.at(video_id - base_video_id);
        OPENVINO_ASSERT(encoded_video.num_video_tokens % m_vlm_config.query_num == 0,
                        "Unexpected MiniCPM video token count.");
        const size_t groups_num = encoded_video.num_video_tokens / m_vlm_config.query_num;
        std::string expanded_tag;
        for (size_t group_idx = 0; group_idx < groups_num; ++group_idx) {
            expanded_tag += VIDEO_GROUP_MARKER + m_vlm_config.im_start + unk64 + m_vlm_config.im_end;
        }
        const size_t position = unified_prompt.find(NATIVE_TAG, search_offset);
        OPENVINO_ASSERT(position != std::string::npos, "Failed to find MiniCPM video placeholder in prompt.");
        unified_prompt.replace(position, NATIVE_TAG.length(), expanded_tag);
        search_offset = position + expanded_tag.size();
    }

    auto [normalized_prompt, image_sequence] = normalize(
        unified_prompt,
        NATIVE_TAG,
        NATIVE_TAG + '\n',
        base_image_id,
        images.size()
    );
    for (size_t image_id : image_sequence) {
        images.at(image_id - base_image_id);
        normalized_prompt.replace(normalized_prompt.find(NATIVE_TAG),
                                  NATIVE_TAG.length(),
                                  m_vlm_config.im_start + unk64 + m_vlm_config.im_end);
    }

    return {std::move(normalized_prompt),
            std::move(image_sequence),
            std::move(video_sequence)};
}

ov::Tensor InputsEmbedderMiniCPM::get_inputs_embeds(const std::string& unified_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
    std::string unk64;
    ov::Tensor encoded_input = get_encoded_input_ids(unified_prompt, metrics);

    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor inputs_embeds = get_text_embedding(req, encoded_input, metrics);

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

ov::Tensor InputsEmbedderMiniCPM::get_inputs_embeds(
    const std::string& unified_prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    const std::vector<ov::genai::EncodedVideo>& videos,
    ov::genai::VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& images_sequence,
    const std::vector<size_t>& videos_sequence,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    std::vector<bool> video_groups;
    size_t search_offset = 0;
    size_t im_start_pos = unified_prompt.find(m_vlm_config.im_start, search_offset);
    while (im_start_pos != std::string::npos) {
        const bool is_video_group = im_start_pos >= sizeof(VIDEO_GROUP_MARKER) - 1 &&
                                    unified_prompt.compare(im_start_pos - (sizeof(VIDEO_GROUP_MARKER) - 1),
                                                           sizeof(VIDEO_GROUP_MARKER) - 1,
                                                           VIDEO_GROUP_MARKER) == 0;
        video_groups.push_back(is_video_group);
        search_offset = im_start_pos + m_vlm_config.im_start.size();
        im_start_pos = unified_prompt.find(m_vlm_config.im_start, search_offset);
    }
    // Needed for SDPA
    size_t current_video_groups = 0;
    for (const size_t video_id : videos_sequence) {
        const EncodedVideo& encoded_video = videos.at(video_id);
        OPENVINO_ASSERT(encoded_video.num_video_tokens % m_vlm_config.query_num == 0,
                        "Unexpected MiniCPM video token count.");
        current_video_groups += encoded_video.num_video_tokens / m_vlm_config.query_num;
    }
    const size_t current_visual_groups = images_sequence.size() + current_video_groups;
    OPENVINO_ASSERT(video_groups.size() >= current_visual_groups,
                    "MiniCPM prompt contains fewer visual groups than provided media.");
    video_groups.erase(video_groups.begin(), video_groups.end() - current_visual_groups);

    std::string prompt = unified_prompt;
    size_t marker_pos = prompt.find(VIDEO_GROUP_MARKER);
    while (marker_pos != std::string::npos) {
        prompt.erase(marker_pos, sizeof(VIDEO_GROUP_MARKER) - 1);
        marker_pos = prompt.find(VIDEO_GROUP_MARKER, marker_pos);
    }

    ov::Tensor encoded_input = get_encoded_input_ids(prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor inputs_embeds = get_text_embedding(req, encoded_input, metrics);

    OPENVINO_ASSERT(m_vlm_config.hidden_size == inputs_embeds.get_shape().at(2),
                    "Unexpected MiniCPM embedding size.");
    const auto start_tokenizer_time = std::chrono::steady_clock::now();
    const ov::Tensor special_tokens = m_tokenizer.encode(
        m_vlm_config.im_start + m_vlm_config.im_end + m_vlm_config.slice_start + m_vlm_config.slice_end,
        ov::genai::add_special_tokens(false)).input_ids;
    const auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(!metrics.raw_metrics.tokenization_durations.empty());
    metrics.raw_metrics.tokenization_durations.back() += ov::genai::MicroSeconds(
        PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    OPENVINO_ASSERT(special_tokens.get_shape().at(1) == 4,
                    "Every MiniCPM vision boundary token must be represented with a single int.");
    const int64_t im_start_id = special_tokens.data<int64_t>()[0];
    const int64_t slice_start_id = special_tokens.data<int64_t>()[2];
    int64_t* begin = encoded_input.data<int64_t>();
    int64_t* ids = begin;
    int64_t* end = ids + encoded_input.get_size();
    float* inputs_embeds_data = inputs_embeds.data<float>();

    size_t image_sequence_idx = 0;
    size_t video_sequence_idx = 0;
    size_t video_group_idx = 0;
    for (const bool is_video_group : video_groups) {
        ids = std::find(ids, end, im_start_id);
        OPENVINO_ASSERT(ids != end, "Not enough MiniCPM vision placeholders for visual embeddings.");
        ++ids;

        if (is_video_group) {
            OPENVINO_ASSERT(video_sequence_idx < videos_sequence.size(),
                            "MiniCPM prompt contains more video groups than provided videos.");
            const EncodedVideo& encoded_video = videos.at(videos_sequence[video_sequence_idx]);
            OPENVINO_ASSERT(encoded_video.num_video_tokens % m_vlm_config.query_num == 0,
                            "Unexpected MiniCPM video token count.");
            OPENVINO_ASSERT(encoded_video.video_features.get_shape().at(1) == encoded_video.num_video_tokens,
                            "Unexpected MiniCPM video feature shape.");
            OPENVINO_ASSERT(static_cast<size_t>(std::distance(ids, end)) >= m_vlm_config.query_num,
                            "MiniCPM video group exceeds the prompt token count.");
            const size_t group_size = m_vlm_config.query_num * m_vlm_config.hidden_size;
            std::copy_n(encoded_video.video_features.data<float>() + video_group_idx * group_size,
                        group_size,
                        inputs_embeds_data + std::distance(begin, ids) * m_vlm_config.hidden_size);
            ids += m_vlm_config.query_num;

            ++video_group_idx;
            if (video_group_idx == encoded_video.num_video_tokens / m_vlm_config.query_num) {
                video_group_idx = 0;
                ++video_sequence_idx;
            }
            continue;
        }

        OPENVINO_ASSERT(image_sequence_idx < images_sequence.size(),
                        "MiniCPM prompt contains more image groups than provided images.");
        const EncodedImage& encoded_image = images.at(images_sequence[image_sequence_idx++]);
        ov::Tensor aligned_resampled_source;
        const ov::Tensor* resampled_source = &encoded_image.resampled_image.resampled_source;
        if (!videos.empty()) {
            size_t pad_to_max = encoded_image.resized_source_size.height * encoded_image.resized_source_size.width;
            for (const EncodedVideo& video : videos) {
                pad_to_max = std::max(pad_to_max,
                                      video.resized_source_size.height * video.resized_source_size.width);
            }
            aligned_resampled_source = std::static_pointer_cast<VisionEncoderMiniCPM>(m_vision_encoder)
                                           ->resample_image(encoded_image, pad_to_max);
            resampled_source = &aligned_resampled_source;
        }
        OPENVINO_ASSERT(resampled_source->get_size() == m_vlm_config.query_num * m_vlm_config.hidden_size,
                        "Unexpected MiniCPM image feature shape.");
        OPENVINO_ASSERT(static_cast<size_t>(std::distance(ids, end)) >= m_vlm_config.query_num,
                        "MiniCPM image group exceeds the prompt token count.");
        std::copy_n(resampled_source->data<float>(),
                resampled_source->get_size(),
                    inputs_embeds_data + std::distance(begin, ids) * m_vlm_config.hidden_size);
        ids += m_vlm_config.query_num;

        const ov::Shape& slices_shape = encoded_image.slices_shape;
        if (videos.empty() && !slices_shape.empty()) {
            for (size_t row_idx = 0; row_idx < slices_shape.at(0); ++row_idx) {
                for (size_t col_idx = 0; col_idx < slices_shape.at(1); ++col_idx) {
                    const ov::Tensor& slice = encoded_image.resampled_image.vision_embed_tensors[row_idx][col_idx];
                    ids = std::find(ids, end, slice_start_id);
                    OPENVINO_ASSERT(ids != end, "Not enough MiniCPM slice placeholders for image embeddings.");
                    ++ids;
                    OPENVINO_ASSERT(slice.get_size() == m_vlm_config.query_num * m_vlm_config.hidden_size,
                                    "Unexpected MiniCPM image slice feature shape.");
                    std::copy_n(slice.data<float>(),
                                slice.get_size(),
                                inputs_embeds_data + std::distance(begin, ids) * m_vlm_config.hidden_size);
                    ids += m_vlm_config.query_num;
                }
            }
        }
    }

    OPENVINO_ASSERT(image_sequence_idx == images_sequence.size(),
                    "MiniCPM prompt contains fewer image groups than provided images.");
    OPENVINO_ASSERT(video_sequence_idx == videos_sequence.size() && video_group_idx == 0,
                    "MiniCPM prompt contains fewer video groups than provided videos.");

    ov::Tensor result(inputs_embeds.get_element_type(), inputs_embeds.get_shape());
    inputs_embeds.copy_to(result);
    return result;
}

std::vector<ov::genai::EncodedVideo> InputsEmbedderMiniCPM::encode_videos(
    const std::vector<ov::Tensor>& videos,
    const std::vector<VideoMetadata>& videos_metadata) {
    return encode_videos(videos, videos_metadata, {});
}

std::vector<ov::genai::EncodedVideo> InputsEmbedderMiniCPM::encode_videos(
    const std::vector<ov::Tensor>& videos,
    const std::vector<VideoMetadata>& videos_metadata,
    const std::vector<EncodedImage>& images) {
    OPENVINO_ASSERT(videos_metadata.empty() || videos.size() == videos_metadata.size(),
                    "Number of videos and video metadata entries must match.");

    constexpr size_t max_frames_per_group = 6;
    auto vision_encoder = std::static_pointer_cast<VisionEncoderMiniCPM>(m_vision_encoder);
    size_t pad_to_max = 0;
    for (const EncodedImage& image : images) {
        pad_to_max = std::max(pad_to_max, image.resized_source_size.height * image.resized_source_size.width);
    }
    std::vector<EncodedVideo> encoded_videos;
    encoded_videos.reserve(videos.size());

    for (size_t video_idx = 0; video_idx < videos.size(); ++video_idx) {
        VideoMetadata metadata = videos_metadata.empty() ? VideoMetadata{} : videos_metadata[video_idx];
        const ov::Tensor sampled_video = sample_video_if_needed(videos[video_idx], metadata);
        std::vector<ov::Tensor> frames = to_single_image_tensors({sampled_video});
        OPENVINO_ASSERT(!frames.empty(), "MiniCPM video must contain at least one frame.");

        std::vector<size_t> temporal_ids;
        temporal_ids.reserve(frames.size());
        for (size_t frame_idx = 0; frame_idx < frames.size(); ++frame_idx) {
            const size_t source_frame_idx = metadata.frames_indices.empty() ? frame_idx : metadata.frames_indices[frame_idx];
            temporal_ids.push_back(metadata.fps > 0.0f
                                       ? static_cast<size_t>(std::round(source_frame_idx / metadata.fps * 10.0f))
                                       : source_frame_idx);
        }

        std::vector<ov::Tensor> group_features;
        group_features.reserve((frames.size() + max_frames_per_group - 1) / max_frames_per_group);
        ImageSize video_frame_size;
        for (size_t group_start = 0; group_start < frames.size(); group_start += max_frames_per_group) {
            const size_t group_end = std::min(group_start + max_frames_per_group, frames.size());
            std::vector<EncodedImage> group_frames;
            std::vector<size_t> group_temporal_ids;
            group_frames.reserve(group_end - group_start);
            group_temporal_ids.reserve(group_end - group_start);
            for (size_t frame_idx = group_start; frame_idx < group_end; ++frame_idx) {
                group_frames.push_back(m_vision_encoder->encode(frames[frame_idx], {{"max_slice_nums", 1}}));
                group_temporal_ids.push_back(temporal_ids[frame_idx]);
            }
            if (group_start == 0) {
                video_frame_size = group_frames.front().resized_source_size;
            }
            group_features.push_back(vision_encoder->resample_video(group_frames, group_temporal_ids, pad_to_max));
        }

        const ov::Shape& group_shape = group_features.front().get_shape();
        ov::Tensor video_features(ov::element::f32,
                                  {1, group_features.size() * group_shape[1], group_shape[2]});
        float* destination = video_features.data<float>();
        for (const ov::Tensor& group_feature : group_features) {
            std::copy_n(group_feature.data<float>(), group_feature.get_size(), destination);
            destination += group_feature.get_size();
        }

        EncodedVideo encoded_video;
        encoded_video.video_features = std::move(video_features);
        encoded_video.num_video_tokens = group_features.size() * m_vlm_config.query_num;
        encoded_video.resized_source_size = video_frame_size;
        encoded_video.frame_num = frames.size();
        encoded_video.metadata = std::move(metadata);
        encoded_videos.push_back(std::move(encoded_video));
    }
    return encoded_videos;
}

ov::Tensor VisionEncoderMiniCPM::resample(const ov::Tensor& encoded_image, const ImageSize& target_size, size_t pad_to_max) {
    size_t bs = encoded_image.get_shape().at(0);
    size_t patch_len = target_size.height * target_size.width;
    OPENVINO_ASSERT(patch_len <= pad_to_max, "MiniCPM resampler padding cannot be smaller than the patch count.");
    adjust_pos_cache(
        {target_size},
        m_vlm_config.hidden_size,
        m_pos_embed_cache
    );
    ov::Tensor key_padding_mask(ov::element::f32, {bs, pad_to_max});
    float* mask_data = key_padding_mask.data<float>();
    size_t embed_len = m_pos_embed_cache.get_shape().at(2);
    ov::Tensor pos_embed(ov::element::f32, {pad_to_max, bs, embed_len});  // BLD => L * B * D
    float* pos_embed_data = pos_embed.data<float>();
    float* cache_data = m_pos_embed_cache.data<float>();
    size_t _d0 = m_pos_embed_cache.get_shape().at(0);
    size_t _d1 = m_pos_embed_cache.get_shape().at(1);
    for (size_t i = 0; i < bs; ++i) {
        size_t target_h = target_size.height;
        size_t target_w = target_size.width;
        for (size_t h_idx = 0; h_idx < target_h; ++h_idx) {
            for (size_t w_idx = 0; w_idx < target_w; ++w_idx) {
                std::copy_n(
                    cache_data + (h_idx * _d1 + w_idx) * embed_len,
                    embed_len,
                    pos_embed_data + (h_idx * target_w + w_idx) * bs * embed_len + i * embed_len
                );
            }
        }
        for (size_t flat = target_h * target_w; flat < pad_to_max; ++flat) {
            std::fill_n(pos_embed_data + flat * bs * embed_len + i * embed_len, embed_len, 0.0f);
        }
        std::fill_n(mask_data + i * pad_to_max, patch_len, 0.0f);
        std::fill_n(mask_data + i * pad_to_max + patch_len, pad_to_max - patch_len, 1.0f);
    }
    ov::Tensor padded_image(encoded_image.get_element_type(), {bs, pad_to_max, encoded_image.get_shape().at(2)});
    std::memset(padded_image.data(), 0, padded_image.get_byte_size());
    const size_t feature_size = encoded_image.get_shape().at(2);
    for (size_t batch_idx = 0; batch_idx < bs; ++batch_idx) {
        std::memcpy(padded_image.data<float>() + batch_idx * pad_to_max * feature_size,
                    encoded_image.data<float>() + batch_idx * patch_len * feature_size,
                    patch_len * feature_size * sizeof(float));
    }
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_resampler.get());
    ov::InferRequest& resampler = infer_request_guard.get();
    resampler.set_tensor("image_feature", padded_image);  // [N, H*W, old_hidden_size]
    resampler.set_tensor("pos_embed", pos_embed);  // [H*W, N, new_hidden_size]
    resampler.set_tensor("key_padding_mask", key_padding_mask);  // [N, H*W]
    resampler.infer();
    auto resampler_out = resampler.get_output_tensor();
    // resampler_out is bound to infer request and the data may become corrupted after next resampler inference 
    // so we need to return a copy to make sure data does not get corrupted 
    ov::Tensor res(resampler_out.get_element_type(), resampler_out.get_shape());
    std::memcpy(res.data(), resampler_out.data(), resampler_out.get_byte_size());
    return res;  // [N, query_num, new_hidden_size]
}

ov::Tensor VisionEncoderMiniCPM::resample_video(const std::vector<EncodedImage>& frames,
                                                const std::vector<size_t>& temporal_ids,
                                                size_t pad_to_max) {
    OPENVINO_ASSERT(!frames.empty(), "MiniCPM video must contain at least one frame.");
    OPENVINO_ASSERT(frames.size() == temporal_ids.size(), "MiniCPM video frame and temporal ID counts must match.");

    const ov::Tensor& first_features = frames.front().resized_source;
    const ov::Shape& first_shape = first_features.get_shape();
    OPENVINO_ASSERT(first_shape.size() == 3 && first_shape[0] == 1,
                    "Unexpected MiniCPM video frame feature shape.");
    const size_t patches_per_frame = first_shape[1];
    pad_to_max = std::max(pad_to_max, patches_per_frame);
    const size_t vision_hidden_size = first_shape[2];
    const ImageSize target_size = frames.front().resized_source_size;
    for (const EncodedImage& frame : frames) {
        OPENVINO_ASSERT(frame.resized_source.get_shape() == first_shape,
                        "All MiniCPM frames in a temporal group must have the same shape.");
        OPENVINO_ASSERT(frame.resized_source_size.height == target_size.height &&
                        frame.resized_source_size.width == target_size.width,
                        "All MiniCPM frames in a temporal group must have the same patch grid.");
    }

    const size_t group_size = frames.size();
    const size_t combined_patches = group_size * pad_to_max;
    ov::Tensor image_feature(ov::element::f32, {1, combined_patches, vision_hidden_size});
    std::memset(image_feature.data(), 0, image_feature.get_byte_size());
    float* image_feature_data = image_feature.data<float>();
    for (size_t frame_idx = 0; frame_idx < group_size; ++frame_idx) {
        std::copy_n(frames[frame_idx].resized_source.data<float>(),
                    frames[frame_idx].resized_source.get_size(),
                    image_feature_data + frame_idx * pad_to_max * vision_hidden_size);
    }

    adjust_pos_cache({target_size}, m_vlm_config.hidden_size, m_pos_embed_cache);
    const size_t embedding_size = m_pos_embed_cache.get_shape().at(2);
    const size_t cache_width = m_pos_embed_cache.get_shape().at(1);
    ov::Tensor pos_embed(ov::element::f32, {combined_patches, 1, embedding_size});
    std::memset(pos_embed.data(), 0, pos_embed.get_byte_size());
    float* pos_embed_data = pos_embed.data<float>();
    const float* spatial_pos_data = m_pos_embed_cache.data<float>();
    for (size_t frame_idx = 0; frame_idx < group_size; ++frame_idx) {
        const float temporal_position = static_cast<float>(temporal_ids[frame_idx]);
        for (size_t patch_idx = 0; patch_idx < patches_per_frame; ++patch_idx) {
            const size_t row = patch_idx / target_size.width;
            const size_t column = patch_idx % target_size.width;
            float* position = pos_embed_data + (frame_idx * pad_to_max + patch_idx) * embedding_size;
            const float* spatial_position = spatial_pos_data + (row * cache_width + column) * embedding_size;
            for (size_t dim = 0; dim < embedding_size / 2; ++dim) {
                const float omega = 1.0f / std::pow(10000.0f, static_cast<float>(dim) / (embedding_size / 2));
                const float value = omega * temporal_position;
                position[dim] = spatial_position[dim] + std::sin(value);
                position[dim + embedding_size / 2] = spatial_position[dim + embedding_size / 2] + std::cos(value);
            }
        }
    }

    ov::Tensor key_padding_mask(ov::element::f32, {1, combined_patches});
    std::fill_n(key_padding_mask.data<float>(), combined_patches, 1.0f);
    for (size_t frame_idx = 0; frame_idx < group_size; ++frame_idx) {
        std::fill_n(key_padding_mask.data<float>() + frame_idx * pad_to_max, patches_per_frame, 0.0f);
    }
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(m_ireq_queue_resampler.get());
    ov::InferRequest& resampler = infer_request_guard.get();
    resampler.set_tensor("image_feature", image_feature);
    resampler.set_tensor("pos_embed", pos_embed);
    resampler.set_tensor("key_padding_mask", key_padding_mask);
    resampler.infer();
    const ov::Tensor& output = resampler.get_output_tensor();
    ov::Tensor result(output.get_element_type(), output.get_shape());
    output.copy_to(result);
    return result;
}

VisionEncoderMiniCPM::VisionEncoderMiniCPM(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap properties) : VisionEncoder{model_dir, device, properties} {
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_dir, "config.json");
    auto compiled_model = utils::singleton_core().compile_model(
        model_dir / "openvino_resampler_model.xml", device,
        utils::get_model_properties(properties, "resampler", device));
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM resampler model");
    m_ireq_queue_resampler = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        }); 
    m_pos_embed_cache = get_2d_sincos_pos_embed(m_vlm_config.hidden_size, {70, 70});
}

VisionEncoderMiniCPM::VisionEncoderMiniCPM(
        const ModelsMap& models_map,
        const std::filesystem::path& config_dir_path,
        const std::string& device,
        const ov::AnyMap device_config) : VisionEncoder{models_map, config_dir_path, device, device_config} {
    const auto& resampler_model = utils::get_model_weights_pair(models_map, "resampler").first;
    const auto& resampler_weights = utils::get_model_weights_pair(models_map, "resampler").second;
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");
    auto compiled_model = utils::singleton_core().compile_model(
        resampler_model, resampler_weights, device,
        utils::get_model_properties(device_config, "resampler", device));
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM resampler model");
    m_ireq_queue_resampler = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        }); 
    m_pos_embed_cache = get_2d_sincos_pos_embed(m_vlm_config.hidden_size, {70, 70});
}


InputsEmbedderMiniCPM::InputsEmbedderMiniCPM(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) {}

InputsEmbedderMiniCPM::InputsEmbedderMiniCPM(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

} // namespace ov::genai
