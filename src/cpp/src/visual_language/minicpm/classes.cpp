
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/minicpm/classes.hpp"

#include "visual_language/clip.hpp"

#include "utils.hpp"

namespace ov::genai {

namespace {

std::string NATIVE_TAG = "<image>./</image>";

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

EncodedImage llava_image_embed_make_with_bytes_slice(clip_ctx& ctx_clip, const ov::Tensor& img, ov::InferRequest& encoder, int max_slice_nums, int scale_resolution, size_t patch_size, bool never_split) {
    clip_image_u8 source = tensor_to_clip_image_u8(img);
    std::vector<std::vector<clip_image_u8>> imgs = slice_image(source, max_slice_nums, scale_resolution, patch_size, never_split);
    std::vector<std::vector<ov::Tensor>> results;
    std::vector<std::vector<ImageSize>> sizes;
    const size_t channels = 3;

    std::vector<std::vector<clip_image_f32>> preprocessed{imgs.size()};
    size_t max_h = 0, max_w = 0, n_images = 0, max_size = 0;
    std::transform(imgs.begin(), imgs.end(), preprocessed.begin(), [&ctx_clip, &max_h, &max_w, &max_size, &n_images](const std::vector<clip_image_u8>& row) {
        std::vector<clip_image_f32> processed_row{row.size()};
        std::transform(row.begin(), row.end(), processed_row.begin(), [&ctx_clip, &max_h, &max_w, &max_size, &n_images](const clip_image_u8& raw) {
            clip_image_f32 im = clip_image_preprocess(ctx_clip, raw);
            if (size_t(im.ny) * size_t(im.nx) > max_size) {
                max_size = size_t(im.ny) * size_t(im.nx);
                max_h = size_t(im.ny);
                max_w = size_t(im.nx);
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
                        clip_value_data += d3_clip_pixel;
                        pixel_value_data += d3_all_pixel;
                    }
                }
                batch_pixel++;
            }
        }
    }
    encoder.set_tensor("pixel_values", pixel_values);

    ov::Tensor patch_attention_mask{ov::element::f32, {pixel_values.get_shape().at(0), 1, max_h / patch_size * max_w / patch_size}};
    float* attention_data = patch_attention_mask.data<float>();
    std::fill_n(attention_data, patch_attention_mask.get_size(), 0.0f);
    std::fill_n(attention_data, resized_preprocessed.ny / patch_size * resized_preprocessed.nx / patch_size, 1.0f);
    if (1 < preprocessed.size()) {
        for (size_t row = 1; row < preprocessed.size(); ++row) {
            size_t n_slices = preprocessed.at(row).size();
            for (size_t col = 0; col < n_slices; ++col) {
                const clip_image_f32& elem = preprocessed.at(row).at(col);
                std::fill_n(attention_data + ((row - 1) * n_slices + col + 1) * max_h / patch_size * max_w / patch_size, elem.ny / patch_size * elem.nx / patch_size, 1.0f);
            }
        }
    }
    encoder.set_tensor("patch_attention_mask", patch_attention_mask);

    ImageSize resized_source_size{resized_preprocessed.ny / patch_size, resized_preprocessed.nx / patch_size};
    std::vector<ImageSize> tgt_sizes{resized_source_size};
    if (1 < preprocessed.size()) {
        for (const std::vector<clip_image_f32>& row : preprocessed) {
            for (const clip_image_f32& elem : row) {
                tgt_sizes.push_back({elem.ny / patch_size, elem.nx / patch_size});
            }
        }
    }
    ov::Tensor position_ids = prepare_vis_position_ids(pixel_values, patch_attention_mask, tgt_sizes, patch_size, ctx_clip.image_size / patch_size);
    encoder.set_tensor("position_ids", position_ids);
    encoder.start_async();
    encoder.wait();
    const ov::Tensor& output_tensor = encoder.get_output_tensor();

    if (1 == preprocessed.size()) {
        ov::Tensor resized_source{ov::element::f32, output_tensor.get_shape()};
        output_tensor.copy_to(resized_source);
        return {std::move(resized_source), resized_source_size};
    }

    size_t old_hidden_size = output_tensor.get_shape().at(2);
    const float* out = output_tensor.data<float>();
    ov::Tensor resized_source{ov::element::f32, {1, resized_source_size.height * resized_source_size.width, old_hidden_size}};
    std::copy_n(out, resized_source.get_size(), resized_source.data<float>());

    size_t n_patches = tgt_sizes.at(1).height * tgt_sizes.at(1).width;
    ov::Tensor encoded_slices{ov::element::f32, {preprocessed.size() - 1, preprocessed.at(1).size(), n_patches, old_hidden_size}};
    for (size_t col = 0; col < preprocessed.size() - 1; ++col) {
        for (size_t row = 0; row < preprocessed.at(1).size(); ++row) {
            std::copy_n(out + (col * preprocessed.at(1).size() + row + 1) * n_patches * old_hidden_size, n_patches * old_hidden_size, encoded_slices.data<float>() + (col * preprocessed.at(1).size() + row) * n_patches * old_hidden_size);
        }
    }
    return {resized_source, resized_source_size, encoded_slices, tgt_sizes.at(1)};
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
    return llava_image_embed_make_with_bytes_slice(ctx_clip, image, encoder, config.max_slice_nums, config.scale_resolution, config.patch_size, 0 == config.max_slice_nums);
}

namespace {

ov::Tensor concatenate_last_dim(const ov::Tensor& first, const ov::Tensor& second) {
    size_t res_d_0 = first.get_shape().at(0);
    size_t res_d_1 = first.get_shape().at(1);
    OPENVINO_ASSERT(second.get_shape().at(0) == res_d_0);
    OPENVINO_ASSERT(second.get_shape().at(1) == res_d_1);
    size_t res_d_2 = first.get_shape().at(2) + second.get_shape().at(2);
    ov::Tensor res{first.get_element_type(), {res_d_0, res_d_1, res_d_2}};
    float* first_data = first.data<float>();
    float* second_data = second.data<float>();
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

    float* pos_data = pos.data<float>();
    float* emb_data = emb.data<float>();

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
    float* grid_data = grid.data<float>();
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

InputsEmbedderMiniCPM::InputsEmbedderMiniCPM(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, device, device_config) {
    auto compiled_model =
        utils::singleton_core().compile_model(model_dir / "openvino_resampler_model.xml", device, device_config);
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
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    auto compiled_model = utils::singleton_core().compile_model(
        utils::get_model_weights_pair(models_map, "resampler").first,
        utils::get_model_weights_pair(models_map, "resampler").second,
        device,
        device_config);
    m_ireq_queue_resampler = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    m_pos_embed_cache = get_2d_sincos_pos_embed(m_vlm_config.hidden_size, {70, 70});
}

ov::Tensor InputsEmbedderMiniCPM::get_inputs_embeds(const std::string& prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics) {
    auto [unified_prompt, images_sequence] = normalize_prompt(
        prompt,
        NATIVE_TAG,
        '(' + NATIVE_TAG + ")\n",
        m_image_id,
        images.size()
    );

    std::string unk64;
    for (size_t idx = 0; idx < m_vlm_config.query_num; ++idx) {
        unk64 += m_vlm_config.unk;
    }

    for (size_t new_image_id : images_sequence) {
        const EncodedImage& encoded_image = images.at(new_image_id - m_prev_image_id);
        std::string expanded_tag;
        if (m_vlm_config.use_image_id) {
            expanded_tag += m_vlm_config.im_id_start + std::to_string(new_image_id) + m_vlm_config.im_id_end;
        }
        expanded_tag += m_vlm_config.im_start + unk64 + m_vlm_config.im_end;
        if (encoded_image.slices) {
            ov::Shape slices_shape = encoded_image.slices.get_shape();
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
    m_image_id = images_sequence.empty() ? m_image_id : *std::max_element(images_sequence.begin(), images_sequence.end()) + 1;

    ov::Tensor encoded_input = get_encoded_input_ids(unified_prompt, metrics);

    ov::Tensor inputs_embeds = m_embedding->infer(encoded_input);
    OPENVINO_ASSERT(
        m_vlm_config.hidden_size == inputs_embeds.get_shape().at(2),
        "Unexpected embedding size"
    );
    auto start_tokenizer_time = std::chrono::steady_clock::now();
    ov::Tensor special_tokens = m_tokenizer.encode(
        m_vlm_config.im_start
        + m_vlm_config.im_end
        + m_vlm_config.slice_start
        + m_vlm_config.slice_end
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
        const EncodedImage& encoded_image = images.at(image_id - m_prev_image_id);
        const ov::Tensor& resampled_source = resample(encoded_image.resized_source, {encoded_image.resized_source_size});
        float* emb = resampled_source.data<float>();
        ids = std::find(ids, end, im_start_id);
        OPENVINO_ASSERT(end != ids);
        ++ids;
        std::copy_n(emb, resampled_source.get_size(), inputs_embeds_data + std::distance(begin, ids) * m_vlm_config.hidden_size);
        ids += m_vlm_config.query_num;
        if (encoded_image.slices) {
            size_t token_idx = 0;
            const ov::Shape& slices_shape = encoded_image.slices.get_shape();
            for (size_t i = 0; i < slices_shape.at(0); ++i) {
                for (size_t ja = 0; ja < slices_shape.at(1); ++ja) {
                    size_t d2 = slices_shape.at(2);
                    size_t d3 = slices_shape.at(3);
                    ov::Tensor encoded_view{ov::element::f32, {1, d2, d3}, encoded_image.slices.data<float>() + (i * slices_shape.at(1) + ja) * d2 * d3};
                    const ov::Tensor& vision_embed_tensor_i_j = resample(encoded_view, {encoded_image.slices_size});
                    ids = std::find(ids, end, slice_start_id);
                    OPENVINO_ASSERT(end != ids);
                    ++ids;
                    std::copy_n(vision_embed_tensor_i_j.data<float>(), vision_embed_tensor_i_j.get_size(), inputs_embeds_data + std::distance(begin, ids) * m_vlm_config.hidden_size);
                    ids += m_vlm_config.query_num;
                }
            }
        }
    }

    if (!m_is_chat_conversation) {
        m_image_id = 0;
        m_prev_image_id = 0;
    }
    return inputs_embeds;
}

void InputsEmbedderMiniCPM::update_chat_history(const std::string& decoded_results, const ov::genai::GenerationStatus generation_finish_status) {
    IInputsEmbedder::update_chat_history(decoded_results, generation_finish_status);
    if (generation_finish_status == ov::genai::GenerationStatus::CANCEL)
        m_image_id = m_prev_image_id;
    else
        m_prev_image_id = m_image_id;
}

void InputsEmbedderMiniCPM::start_chat(const std::string& system_message) {
    IInputsEmbedder::start_chat(system_message);
    m_prev_image_id = 0;
}

void InputsEmbedderMiniCPM::finish_chat() {
    IInputsEmbedder::finish_chat();
    m_prev_image_id = 0;
}

bool InputsEmbedderMiniCPM::prompt_has_image_tag(const std::string& prompt) const {
    return IInputsEmbedder::prompt_has_image_tag(prompt) || prompt.find(NATIVE_TAG) != std::string::npos;
}

ov::Tensor InputsEmbedderMiniCPM::resample(const ov::Tensor& encoded_image, const std::vector<ImageSize>& target_sizes) {
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
    ov::Tensor pos_embed(ov::element::f32, {max_patch_len, bs, embed_len});  // BLD => L * B * D
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
    resampler.set_tensor("image_feature", encoded_image);  // [N, H*W, old_hidden_size]
    resampler.set_tensor("pos_embed", pos_embed);  // [H*W, N, new_hidden_size]
    resampler.set_tensor("key_padding_mask", key_padding_mask);  // [N, H*W]
    resampler.start_async();
    resampler.wait();
    return resampler.get_output_tensor();  // [N, query_num, new_hidden_size]
}

} // namespace ov::genai