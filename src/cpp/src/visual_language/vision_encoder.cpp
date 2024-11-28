// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vision_encoder.hpp"
#include "visual_language/clip.hpp"
#include "utils.hpp"

using namespace ov::genai;

namespace {
/**
 * @brief Converts an OpenVINO image tensor (1HWC) to a clip_image_u8 structure.
 *
 * @param image_tensor An OpenVINO tensor (1HWC) containing the image data.
 * @return A clip_image_u8 structure containing the image data.
 */
clip_image_u8 tensor_to_clip_image_u8(const ov::Tensor& image_tensor) {
    clip_image_u8 image{
        int(image_tensor.get_shape().at(2)),
        int(image_tensor.get_shape().at(1)),
        {image_tensor.data<uint8_t>(), image_tensor.data<uint8_t>() + image_tensor.get_size()}
    };
    return image;
}

/**
 * @brief Converts a clip_image_f32 structure to an OpenVINO image tensor (1CHW).
 *
 * @param image A clip_image_f32 structure containing the image data.
 * @return An OpenVINO tensor containing the image data (1CHW).
 */
ov::Tensor clip_image_f32_to_tensor(const clip_image_f32& image) {
    ov::Tensor image_tensor{
        ov::element::f32,
        {1, 3, static_cast<size_t>(image.ny), static_cast<size_t>(image.nx)}
    };
    std::memcpy(image_tensor.data<float>(), image.buf.data(), image.buf.size() * sizeof(float));
    return image_tensor;
}

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
    const float log_ratio = log(1.0f * original_width / original_height);
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
    size_t num_patches_per_side
) {
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
    std::vector<std::vector<clip_image_u8>> imgs = ::slice_image(source, max_slice_nums, scale_resolution, patch_size, never_split);
    std::vector<std::vector<ov::Tensor>> results;
    std::vector<std::vector<ImageSize>> sizes;

    std::vector<std::vector<clip_image_f32>> preprocessed{imgs.size()};
    size_t max_h = 0, max_w = 0, n_images = 0;
    std::transform(imgs.begin(), imgs.end(), preprocessed.begin(), [&ctx_clip, &max_h, &max_w, &n_images](const std::vector<clip_image_u8>& row) {
        std::vector<clip_image_f32> processed_row{row.size()};
        std::transform(row.begin(), row.end(), processed_row.begin(), [&ctx_clip, &max_h, &max_w, &n_images](const clip_image_u8& raw) {
            clip_image_f32 im = clip_image_preprocess(ctx_clip, raw);
            max_h = std::max(size_t(im.ny), max_h);
            max_w = std::max(size_t(im.nx), max_w);
            ++n_images;
            return im;
        });
        return processed_row;
    });

    ov::Tensor batched_images{ov::element::f32, {n_images, 3, max_h, max_w}};
    float* batched_data = batched_images.data<float>();
    const clip_image_f32& resized_preprocessed = preprocessed.at(0).at(0);
    std::copy(resized_preprocessed.buf.begin(), resized_preprocessed.buf.end(), batched_data);
    if (1 < preprocessed.size()) {
        for (size_t row = 1; row < preprocessed.size(); ++row) {
            size_t n_slices = preprocessed.at(row).size();
            for (size_t col = 0; col < n_slices; ++col) {
                const clip_image_f32& elem = preprocessed.at(row).at(col);
                std::copy(elem.buf.begin(), elem.buf.end(), batched_data + ((row - 1) * n_slices + col + 1) * 3 * max_h * max_w);
            }
        }
    }
    ov::Tensor pixel_values = preprocess_for_encoder(batched_images, patch_size);
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
    encoder.infer();
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

ProcessorConfig from_any_map(
    const ov::AnyMap& config_map,
    const ProcessorConfig& initial
) {
    auto iter = config_map.find("processor_config");
    ProcessorConfig extracted_config = config_map.end() != iter ?
        iter->second.as<ProcessorConfig>() : initial;
    using utils::read_anymap_param;
    read_anymap_param(config_map, "patch_size", extracted_config.patch_size);
    read_anymap_param(config_map, "scale_resolution", extracted_config.scale_resolution);
    read_anymap_param(config_map, "max_slice_nums", extracted_config.max_slice_nums);
    read_anymap_param(config_map, "norm_mean", extracted_config.norm_mean);
    read_anymap_param(config_map, "norm_std", extracted_config.norm_std);
    return extracted_config;
}

clip_image_f32 preprocess_clip_image_llava(const clip_image_u8& image, const ProcessorConfig& config) {
    bool do_resize = true;
    bool do_center_crop = true;

    // Resize
    clip_image_u8 resized_image;
    if (do_resize) {
        int target_size = config.size_shortest_edge;
        float scale = static_cast<float>(target_size) / std::min(image.nx, image.ny);
        int new_width = static_cast<int>(image.nx * scale);
        int new_height = static_cast<int>(image.ny * scale);
        bicubic_resize(image, resized_image, new_width, new_height);
    } else {
        resized_image = image;
    }

    // Center crop
    clip_image_u8 cropped_image;
    if (do_center_crop) {
        int crop_height = config.crop_size_height;
        int crop_width = config.crop_size_width;
        int start_x = (resized_image.nx - crop_width) / 2;
        int start_y = (resized_image.ny - crop_height) / 2;

        cropped_image.nx = crop_width;
        cropped_image.ny = crop_height;
        cropped_image.buf.resize(3 * crop_width * crop_height);

        for (int y = 0; y < crop_height; ++y) {
            for (int x = 0; x < crop_width; ++x) {
                for (int c = 0; c < 3; ++c) {
                    cropped_image.buf[(y * crop_width + x) * 3 + c] = 
                        resized_image.buf[((start_y + y) * resized_image.nx + (start_x + x)) * 3 + c];
                }
            }
        }
    } else {
        cropped_image = resized_image;
    }

    // Normalize
    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);

    clip_image_f32 normalized_image = clip_image_preprocess(ctx, cropped_image);
    return normalized_image;
}

ov::Tensor get_pixel_values_llava(const ov::Tensor& image, const ProcessorConfig& config) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    clip_image_f32 preprocessed_image = preprocess_clip_image_llava(input_image, config);
    return clip_image_f32_to_tensor(preprocessed_image);
}

ov::Tensor get_pixel_values_llava_next(const ov::Tensor& image, const ProcessorConfig& config) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);

    std::pair<int, int> size{config.size_shortest_edge, config.size_shortest_edge};
    auto patch_size = config.crop_size_height;
    auto image_patches = get_image_patches(input_image, config.image_grid_pinpoints, size, patch_size);

    // Preprocess image patches
    std::vector<clip_image_f32> processed_patches;
    processed_patches.reserve(image_patches.size());

    for (const auto& patch : image_patches) {
        processed_patches.push_back(preprocess_clip_image_llava(patch, config));
    }

    size_t num_patches = processed_patches.size();
    size_t channels = 3;
    size_t height = processed_patches[0].ny;
    size_t width = processed_patches[0].nx;

    ov::Tensor concatenated_tensor(ov::element::f32, {num_patches, channels, height, width});
    float* tensor_data = concatenated_tensor.data<float>();

    // Fill the tensor with the preprocessed patches data (each patch layout is [C * H * W])
    for (size_t i = 0; i < num_patches; ++i) {
        const auto& img = processed_patches[i];
        std::copy(img.buf.begin(), img.buf.end(), tensor_data + i * channels * height * width);
    }

    return concatenated_tensor;
}

std::vector<clip_image_u8> split_image_internvl(
    const clip_image_u8& image,
    int image_size,
    int min_num = 1,
    int max_num = 12,
    bool use_thumbnail = true
) {
    int orig_width = image.nx;
    int orig_height = image.ny;
    float aspect_ratio = static_cast<float>(orig_width) / orig_height;

    std::vector<std::pair<int, int>> target_ratios;
    for (int n = min_num; n <= max_num; ++n) {
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= n; ++j) {
                if (i * j <= max_num && i * j >= min_num) {
                    target_ratios.emplace_back(i, j);
                }
            }
        }
    }
    std::sort(target_ratios.begin(), target_ratios.end(),
        [](const auto& a, const auto& b) { return a.first * a.second < b.first * b.second; });

    auto find_closest_aspect_ratio = [&](float ar, const std::vector<std::pair<int, int>>& ratios) {
        float best_ratio_diff = std::numeric_limits<float>::max();
        std::pair<int, int> best_ratio = {1, 1};
        int area = orig_width * orig_height;

        for (const auto& ratio : ratios) {
            float target_ar = static_cast<float>(ratio.first) / ratio.second;
            float ratio_diff = std::abs(ar - target_ar);
            if (ratio_diff < best_ratio_diff) {
                best_ratio_diff = ratio_diff;
                best_ratio = ratio;
            } else if (ratio_diff == best_ratio_diff && area > 0.5 * image_size * image_size * ratio.first * ratio.second) {
                best_ratio = ratio;
            }
        }
        return best_ratio;
    };

    auto target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios);

    int target_width = image_size * target_aspect_ratio.first;
    int target_height = image_size * target_aspect_ratio.second;
    int blocks = target_aspect_ratio.first * target_aspect_ratio.second;

    clip_image_u8 resized_img;
    bicubic_resize(image, resized_img, target_width, target_height);

    std::vector<clip_image_u8> processed_images;
    for (int i = 0; i < blocks; ++i) {
        int x = (i % (target_width / image_size)) * image_size;
        int y = (i / (target_width / image_size)) * image_size;

        clip_image_u8 split_img;
        split_img.nx = image_size;
        split_img.ny = image_size;
        split_img.buf.resize(3 * image_size * image_size);

        for (int dy = 0; dy < image_size; ++dy) {
            for (int dx = 0; dx < image_size; ++dx) {
                for (int c = 0; c < 3; ++c) {
                    int src_idx = ((y + dy) * target_width + (x + dx)) * 3 + c;
                    int dst_idx = (dy * image_size + dx) * 3 + c;
                    split_img.buf[dst_idx] = resized_img.buf[src_idx];
                }
            }
        }

        processed_images.push_back(std::move(split_img));
    }

    if (use_thumbnail && processed_images.size() != 1) {
        clip_image_u8 thumbnail_img;
        bicubic_resize(image, thumbnail_img, image_size, image_size);
        processed_images.push_back(std::move(thumbnail_img));
    }

    return processed_images;
}

ov::Tensor get_pixel_values_internvl(const ov::Tensor& image, const ProcessorConfig& config) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);

    const size_t image_size = config.size_shortest_edge;

    clip_ctx ctx;
    ctx.image_size = image_size;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);

    std::vector<clip_image_u8> splitted_images = split_image_internvl(input_image, image_size);

    std::vector<clip_image_f32> processed_images;
    processed_images.reserve(splitted_images.size());
    for (const auto& image : splitted_images) {
        processed_images.push_back(clip_image_preprocess(ctx, image));
    }

    size_t batch_size = processed_images.size();
    size_t channels = 3;
    size_t height = processed_images[0].ny;
    size_t width = processed_images[0].nx;

    ov::Tensor output_tensor(ov::element::f32, {batch_size, channels, height, width});
    float* output_data = output_tensor.data<float>();

    for (size_t i = 0; i < batch_size; ++i) {
        const auto& img = processed_images[i];
        std::copy(img.buf.begin(), img.buf.end(), output_data + i * channels * height * width);
    }
    return output_tensor;
}
}

VisionEncoder::VisionEncoder(const std::filesystem::path& model_dir, const VLMModelType model_type, const std::string& device, const ov::AnyMap device_config) :
    model_type(model_type) {
        m_vision_encoder = utils::singleton_core().compile_model(model_dir / "openvino_vision_embeddings_model.xml", device, device_config).create_infer_request();
        m_processor_config = utils::from_config_json_if_exists<ProcessorConfig>(
            model_dir, "preprocessor_config.json"
        );
}

VisionEncoder::VisionEncoder(
    const std::string& model,
    const ov::Tensor& weights,
    const std::filesystem::path& config_dir_path,
    const VLMModelType model_type,
    const std::string& device,
    const ov::AnyMap device_config
) :
    model_type(model_type) {
        m_vision_encoder = utils::singleton_core().compile_model(model, weights, device, device_config).create_infer_request();
        m_processor_config = utils::from_config_json_if_exists<ProcessorConfig>(
            config_dir_path, "preprocessor_config.json"
        );
}

EncodedImage VisionEncoder::encode(const ov::Tensor& image, const ProcessorConfig& config) {
    if (model_type == VLMModelType::MINICPM) {
        return encode_minicpm(image, config);
    } else if (model_type == VLMModelType::LLAVA) {
        return encode_llava(image, config);
    } else if (model_type == VLMModelType::LLAVA_NEXT) {
        return encode_llava_next(image, config);
    }  else if (model_type == VLMModelType::INTERNVL_CHAT) {
        return encode_internvl(image, config);
    } else {
        OPENVINO_THROW("Unsupported type of VisionEncoder");
    }
}

EncodedImage VisionEncoder::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    return encode(image, from_any_map(
        config_map, m_processor_config
    ));
}

EncodedImage VisionEncoder::encode_minicpm(const ov::Tensor& image, const ProcessorConfig& config) {
    clip_ctx ctx_clip;
    ctx_clip.image_size = m_processor_config.image_size;
    std::copy(config.norm_mean.begin(), config.norm_mean.end(), ctx_clip.image_mean);
    std::copy(config.norm_std.begin(), config.norm_std.end(), ctx_clip.image_std);
    return llava_image_embed_make_with_bytes_slice(ctx_clip, image, m_vision_encoder, config.max_slice_nums, config.scale_resolution, config.patch_size, 0 == config.max_slice_nums);
}

EncodedImage VisionEncoder::encode_llava(const ov::Tensor& image, const ProcessorConfig& config) {
    ov::Tensor pixel_values = get_pixel_values_llava(image, config);

    m_vision_encoder.set_tensor("pixel_values", pixel_values);
    m_vision_encoder.infer();

    const ov::Tensor& infer_output = m_vision_encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    ImageSize resized_source_size{config.crop_size_height / config.patch_size, config.crop_size_width / config.patch_size};

    return {std::move(image_features), resized_source_size};
}

EncodedImage VisionEncoder::encode_llava_next(const ov::Tensor& image, const ProcessorConfig& config) {
    ov::Tensor pixel_values = get_pixel_values_llava_next(image, config);

    m_vision_encoder.set_tensor("pixel_values", pixel_values);
    m_vision_encoder.infer();

    const ov::Tensor& infer_output = m_vision_encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    ImageSize resized_source_size{config.crop_size_height / config.patch_size, config.crop_size_width / config.patch_size};

    // Gen number of patches
    ImageSize original_image_size{image.get_shape().at(1), image.get_shape().at(2)};
    auto best_resolution = select_best_resolution({original_image_size.width, original_image_size.height}, config.image_grid_pinpoints);
    int num_patches_w = best_resolution.first / config.size_shortest_edge;
    int num_patches_h = best_resolution.second / config.size_shortest_edge;

    EncodedImage encoded_image;
    encoded_image.resized_source = std::move(image_features);
    encoded_image.resized_source_size = resized_source_size;
    encoded_image.patches_grid = {num_patches_h, num_patches_w};
    return encoded_image;
}

EncodedImage VisionEncoder::encode_internvl(const ov::Tensor& image, const ProcessorConfig& config) {
    ov::Tensor pixel_values = get_pixel_values_internvl(image, config);

    m_vision_encoder.set_tensor("pixel_values", pixel_values);
    m_vision_encoder.infer();

    const ov::Tensor& infer_output = m_vision_encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    ImageSize resized_source_size{config.crop_size_height / config.patch_size, config.crop_size_width / config.patch_size};

    return {std::move(image_features), resized_source_size};
}
