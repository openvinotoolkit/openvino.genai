// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Based on clip.cpp

#include "clip.hpp"
#include <cmath>

clip_image_u8 tensor_to_clip_image_u8(const ov::Tensor& image_tensor) {
    clip_image_u8 image{
        int(image_tensor.get_shape().at(2)),
        int(image_tensor.get_shape().at(1)),
        {image_tensor.data<uint8_t>(), image_tensor.data<uint8_t>() + image_tensor.get_size()}
    };
    return image;
}

ov::Tensor clip_image_f32_to_tensor(const clip_image_f32& image) {
    ov::Tensor image_tensor{
        ov::element::f32,
        {1, 3, static_cast<size_t>(image.ny), static_cast<size_t>(image.nx)}
    };
    std::memcpy(image_tensor.data<float>(), image.buf.data(), image.buf.size() * sizeof(float));
    return image_tensor;
}


// Linear interpolation between two points
static float clip_lerp(float s, float e, float t) {
    return s + (e - s) * t;
}

// Bilinear resize function
void bilinear_resize(const clip_image_u8& src, clip_image_u8& dst, int target_width, int target_height) {
    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    float x_ratio = static_cast<float>(src.nx - 1) / target_width;
    float y_ratio = static_cast<float>(src.ny - 1) / target_height;

    for (int y = 0; y < target_height; y++) {
        for (int x = 0; x < target_width; x++) {
            float px = x_ratio * x;
            float py = y_ratio * y;
            int x_floor = static_cast<int>(px);
            int y_floor = static_cast<int>(py);
            float x_lerp = px - x_floor;
            float y_lerp = py - y_floor;

            for (int c = 0; c < 3; c++) {
                float top = clip_lerp(
                    static_cast<float>(src.buf[3 * (y_floor * src.nx + x_floor) + c]),
                    static_cast<float>(src.buf[3 * (y_floor * src.nx + (x_floor + 1)) + c]),
                    x_lerp
                );
                float bottom = clip_lerp(
                    static_cast<float>(src.buf[3 * ((y_floor + 1) * src.nx + x_floor) + c]),
                    static_cast<float>(src.buf[3 * ((y_floor + 1) * src.nx + (x_floor + 1)) + c]),
                    x_lerp
                );
                dst.buf[3 * (y * target_width + x) + c] = static_cast<uint8_t>(clip_lerp(top, bottom, y_lerp));
            }
        }
    }
}

template<typename NUM>
NUM clip(NUM x, NUM lower, NUM upper) {
    return std::max(lower, std::min(x, upper));
}

void bicubic_resize(const clip_image_u8 &img, clip_image_u8 &dst, int target_width, int target_height) {
    const int nx = img.nx;
    const int ny = img.ny;

    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    float Cc;
    float C[5];
    float d0, d2, d3, a0, a1, a2, a3;
    int i, j, k, jj;
    int x, y;
    float dx, dy;
    float tx, ty;

    tx = (float)nx / (float)target_width;
    ty = (float)ny / (float)target_height;

    // Bicubic interpolation; adapted from ViT.cpp, inspired from :
    //    -> https://github.com/yglukhov/bicubic-interpolation-image-processing/blob/master/libimage.c#L36
    //    -> https://en.wikipedia.org/wiki/Bicubic_interpolation

    for (i = 0; i < target_height; i++) {
        for (j = 0; j < target_width; j++) {
            x = (int)(tx * j);
            y = (int)(ty * i);

            dx = tx * j - x;
            dy = ty * i - y;

            for (k = 0; k < 3; k++) {
                for (jj = 0; jj <= 3; jj++) {
                    d0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x - 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d2 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 1, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    d3 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x + 2, 0, nx - 1)) * 3 + k] - img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];
                    a0 = img.buf[(clip(y - 1 + jj, 0, ny - 1) * nx + clip(x, 0, nx - 1)) * 3 + k];

                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;

                    C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

                    d0 = C[0] - C[1];
                    d2 = C[2] - C[1];
                    d3 = C[3] - C[1];
                    a0 = C[1];
                    a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
                    a2 =  1.0 / 2 * d0 +      1.0 / 2 * d2;
                    a3 = -1.0 / 6 * d0 -      1.0 / 2 * d2 + 1.0 / 6 * d3;
                    Cc = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;

                    const uint8_t Cc2 = std::min(std::max(std::round(Cc), 0.0f), 255.0f);
                    dst.buf[(i * target_width + j) * 3 + k] = float(Cc2);
                }
            }
        }
    }
}

// llava-1.6 type of resize_and_pad (black by default)
clip_image_u8 resize_and_pad_image(const clip_image_u8& image, const std::pair<int, int>& target_resolution, uint8_t pad_value) {
    int target_width = target_resolution.first;
    int target_height = target_resolution.second;

    float scale_w = static_cast<float>(target_width) / image.nx;
    float scale_h = static_cast<float>(target_height) / image.ny;

    int new_width, new_height;

    if (scale_w < scale_h) {
        new_width = target_width;
        new_height = std::min(static_cast<int>(std::ceil(image.ny * scale_w)), target_height);
    } else {
        new_height = target_height;
        new_width = std::min(static_cast<int>(std::ceil(image.nx * scale_h)), target_width);
    }

    clip_image_u8 resized_image;
    bicubic_resize(image, resized_image, new_width, new_height);

    clip_image_u8 padded_image;
    padded_image.nx = target_width;
    padded_image.ny = target_height;
    padded_image.buf.resize(3 * target_width * target_height, pad_value); // Initialize with pad value

    // Calculate padding offsets
    int pad_x = (target_width - new_width) / 2;
    int pad_y = (target_height - new_height) / 2;

    // Copy the resized image into the center of the padded buffer
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                padded_image.buf[3 * ((y + pad_y) * target_width + (x + pad_x)) + c] = resized_image.buf[3 * (y * new_width + x) + c];
            }
        }
    }
    return padded_image;
}

/**
 * Select the best resolution from a list of possible resolutions based on the original size.
 *
 * @param original_size The original size of the image in the format (width, height).
 * @param possible_resolutions A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].
 * @return The best fit resolution in the format (width, height).
 */
std::pair<int, int> select_best_resolution(const std::pair<int, int> & original_size, const std::vector<std::pair<int, int>> & possible_resolutions) {
    // TODO Consider changing original_size and return value to (height, width) format
    int original_width = original_size.first;
    int original_height = original_size.second;
    std::pair<int, int> best_fit;
    int max_effective_resolution = 0;
    int min_wasted_resolution = std::numeric_limits<int>::max();

    for (const auto& resolution : possible_resolutions) {
        int width = resolution.first;
        int height = resolution.second;
        float scale = std::min(static_cast<float>(width) / original_width, static_cast<float>(height) / original_height);
        int downscaled_width = static_cast<int>(original_width * scale);
        int downscaled_height = static_cast<int>(original_height * scale);
        int effective_resolution = std::min(downscaled_width * downscaled_height, original_width * original_height);
        int wasted_resolution = (width * height) - effective_resolution;
        if (effective_resolution > max_effective_resolution || (effective_resolution == max_effective_resolution && wasted_resolution < min_wasted_resolution)) {
            max_effective_resolution = effective_resolution;
            min_wasted_resolution = wasted_resolution;
            best_fit = resolution;
        }
    }

    return best_fit;
}

// returns the normalized float tensor for llava-1.5, for spatial_unpad with anyres processing for llava-1.6 it returns the normalized image patch tensors as a vector
clip_image_f32 clip_image_preprocess(clip_ctx& ctx, const clip_image_u8& img) {
    bool pad_to_square = true;

    clip_image_u8 temp;  // we will keep the input image data here temporarily
    temp.nx = img.nx;
    temp.ny = img.ny;
    temp.buf.resize(img.buf.size());
    memcpy(temp.buf.data(), img.buf.data(), temp.buf.size());


    const int nx = temp.nx;
    const int ny = temp.ny;

    const int nx2 = temp.nx;
    const int ny2 = temp.ny;

    clip_image_f32 res;
    res.nx = nx2;
    res.ny = ny2;
    res.buf.resize(3 * nx2 * ny2);

    const int nx3 = nx;
    const int ny3 = ny;

    const auto& m3 = ctx.image_mean; // {0.48145466f, 0.4578275f, 0.40821073f};
    const auto& s3 = ctx.image_std;  // {0.26862954f, 0.26130258f, 0.27577711f};

    for (int y = 0; y < ny3; y++) {
        for (int x = 0; x < nx3; x++) {
            for (int c = 0; c < 3; c++) {
                // linear interpolation
                const float sx = x;
                const float sy = y;

                const int x0 = std::max(0, (int)std::floor(sx));
                const int y0 = std::max(0, (int)std::floor(sy));

                const int x1 = std::min(x0 + 1, nx - 1);
                const int y1 = std::min(y0 + 1, ny - 1);

                const float dx = sx - x0;
                const float dy = sy - y0;

                const int j00 = 3 * (y0 * nx + x0) + c;
                const int j01 = 3 * (y0 * nx + x1) + c;
                const int j10 = 3 * (y1 * nx + x0) + c;
                const int j11 = 3 * (y1 * nx + x1) + c;

                const float v00 = temp.buf[j00];
                const float v01 = temp.buf[j01];
                const float v10 = temp.buf[j10];
                const float v11 = temp.buf[j11];

                const float v0 = v00 * (1.0f - dx) + v01 * dx;
                const float v1 = v10 * (1.0f - dx) + v11 * dx;

                const float v = v0 * (1.0f - dy) + v1 * dy;

                const uint8_t v2 = std::min(std::max(std::round(v), 0.0f), 255.0f);

                //rgb hwc ->chw
                const int i = (y * nx3 + x) + c * nx3 * ny3;

                res.buf[i] = ((float(v2) / 255.0f) - m3[c]) / s3[c];
            }
        }
    }
    return res;
}

clip_image_u8 center_crop(const clip_image_u8& image, size_t crop_height, size_t crop_width) {
    clip_image_u8 cropped_image;
    size_t start_x = (image.nx - crop_width) / 2;
    size_t start_y = (image.ny - crop_height) / 2;

    cropped_image.nx = crop_width;
    cropped_image.ny = crop_height;
    cropped_image.buf.resize(3 * crop_width * crop_height);

    for (size_t y = 0; y < crop_height; ++y) {
        for (size_t x = 0; x < crop_width; ++x) {
            for (size_t c = 0; c < 3; ++c) {
                cropped_image.buf[(y * crop_width + x) * 3 + c] =
                    image.buf[((start_y + y) * image.nx + (start_x + x)) * 3 + c];
            }
        }
    }

    return cropped_image;
}

clip_image_f32 normalize_and_convert_to_chw(const clip_image_u8& img, const clip_ctx_double& image_mean_std) {
    const size_t nx = img.nx;
    const size_t ny = img.ny;
    const auto& image_mean = image_mean_std.image_mean;
    const auto& image_std = image_mean_std.image_std; 

    clip_image_f32 res;
    res.nx = nx;
    res.ny = ny;
    res.buf.resize(3 * nx * ny);

    for (size_t y = 0; y < ny; y++) {
        for (size_t x = 0; x < nx; x++) {
            for (size_t c = 0; c < 3; c++) {
                const uint8_t val = img.buf[3 * (y * nx + x) + c];
                const size_t i = (y * nx + x) + c * nx * ny;

                // perform division in double values, to align with python,
                // as some models are sensitive to small values deviations, like llava-next-video
                res.buf[i] = (double(val) - image_mean[c]) / image_std[c];
            }
        }
    }
    return res;
}

std::vector<clip_image_u8> get_image_patches(
    const clip_image_u8& image, 
    const std::vector<std::pair<int, int>>& image_grid_pinpoints,
    const std::pair<int, int>& size,
    int patch_size
) {
    std::vector<clip_image_u8> patches;

    // Get image dimensions
    int orig_width = image.nx;
    int orig_height = image.ny;

    // Resize base patch
    int base_patch_width = size.first;
    int base_patch_height = size.second;
    clip_image_u8 base_patch;
    bicubic_resize(image, base_patch, base_patch_width, base_patch_height);
    
    patches.push_back(base_patch);

    // Select best resolution for patching
    auto best_resolution = select_best_resolution({orig_width, orig_height}, image_grid_pinpoints);
    int width = best_resolution.first;
    int height = best_resolution.second;

    // Resize and pad image for patching
    clip_image_u8 resized_image = resize_and_pad_image(image, best_resolution);

    // Calculate patch dimensions
    int patches_w = width / patch_size;
    int patches_h = height / patch_size;

    // Extract patches
    for (int h = 0; h < patches_h; ++h) {
        for (int w = 0; w < patches_w; ++w) {
            clip_image_u8 patch;
            patch.nx = patch_size;
            patch.ny = patch_size;
            patch.buf.resize(3 * patch_size * patch_size);

            for (int y = 0; y < patch_size; ++y) {
                for (int x = 0; x < patch_size; ++x) {
                    for (int c = 0; c < 3; ++c) {
                        int src_y = h * patch_size + y;
                        int src_x = w * patch_size + x;
                        int src_idx = (src_y * width + src_x) * 3 + c;
                        int dst_idx = (y * patch_size + x) * 3 + c;
                        patch.buf[dst_idx] = resized_image.buf[src_idx];
                    }
                }
            }
            patches.push_back(patch);
        }
    }

    return patches;
}
