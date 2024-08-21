// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "stb_image.hpp"
#include "log.hpp"
#include "openvino/genai/clip.hpp"

#include <openvino/genai/vlm_minicpmv.hpp>


typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

static double get_duration_ms_until_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<ns>(Time::now() - startTime).count() * 0.000001;
}

int ensure_divide(int length, int patch_size) {
    return std::max(static_cast<int>(std::round(static_cast<float>(length) / patch_size) * patch_size), patch_size);
}

std::pair<int, int> find_best_resize(std::pair<int, int> original_size, int scale_resolution, int patch_size, bool allow_upscale = false) {
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

std::pair<int, int> get_refine_size(std::pair<int, int> original_size, std::pair<int, int> grid, int scale_resolution, int patch_size, bool allow_upscale = false) {
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



std::vector<std::vector<clip_image_u8*>> slice_image(const clip_image_u8* img, const int max_slice_nums, const int scale_resolution, const int patch_size, const bool never_split) {
    const std::pair<int, int> original_size = { img->nx,img->ny };
    const int original_width = img->nx;
    const int original_height = img->ny;
    const float log_ratio = log(1.0 * original_width / original_height); //
    const float ratio = 1.0 * original_width * original_height / (scale_resolution * scale_resolution);
    const int multiple = fmin(ceil(ratio), max_slice_nums);

    std::vector<std::vector<clip_image_u8*>> images;
    images.push_back(std::vector<clip_image_u8*>());

    if (multiple <= 1) {
        auto best_size = find_best_resize(original_size, scale_resolution, patch_size, true);
        clip_image_u8* source_image = clip_image_u8_init();
        bicubic_resize(*img, *source_image, best_size.first, best_size.second);
        images[images.size() - 1].push_back(source_image);
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
        clip_image_u8* source_image = clip_image_u8_init();
        bicubic_resize(*img, *source_image, best_size.first, best_size.second);
        images[images.size() - 1].push_back(source_image);

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
            float error = std::abs(log_ratio - std::log(1.0 * grid.first / grid.second));
            if (error < min_error) {
                best_grid = grid;
                min_error = error;
            }
        }
        //LOG_TEE("%s: image_size: %d %d; best_grid: %d %d\n", __func__, img->nx, img->ny, best_grid.first, best_grid.second);

        auto refine_size = get_refine_size(original_size, best_grid, scale_resolution, patch_size, true);
        clip_image_u8* refine_image = clip_image_u8_init();
        bicubic_resize(*img, *refine_image, refine_size.first, refine_size.second);

        //LOG_TEE("%s: refine_image_size: %d %d; best_grid: %d %d\n", __func__, refine_image->nx, refine_image->ny, best_grid.first, best_grid.second);

        // split_to_patches
        int width = refine_image->nx;
        int height = refine_image->ny;
        int grid_x = int(width / best_grid.first);
        int grid_y = int(height / best_grid.second);
        for (int patches_i = 0, ic = 0; patches_i < height && ic < best_grid.second; patches_i += grid_y, ic += 1) {
            images.push_back(std::vector<clip_image_u8*>());
            for (int patches_j = 0, jc = 0; patches_j < width && jc < best_grid.first; patches_j += grid_x, jc += 1) {
                clip_image_u8* patch = clip_image_u8_init();
                patch->nx = grid_x;
                patch->ny = grid_y;
                patch->buf.resize(3 * patch->nx * patch->ny);
                for (int y = patches_i; y < patches_i + grid_y; ++y) {
                    for (int x = patches_j; x < patches_j + grid_x; ++x) {
                        const int i = 3 * (y * refine_image->nx + x);
                        const int j = 3 * ((y - patches_i) * patch->nx + (x - patches_j));
                        patch->buf[j] = refine_image->buf[i];
                        patch->buf[j + 1] = refine_image->buf[i + 1];
                        patch->buf[j + 2] = refine_image->buf[i + 2];
                    }
                }
                images[images.size() - 1].push_back(patch);
            }
        }
    }

    return images;
}

ov::Tensor encode_image_with_clip(clip_ctx* ctx_clip, const clip_image_u8* img) {
    // std::vector<clip_image_f32*> img_res_v; // format VectN x H x W x RGB (N x 336 x 336 x 3), so interleaved RGB - different to the python implementation which is N x 3 x 336 x 336
    clip_image_f32_batch img_res_v;
    img_res_v.size = 0;
    img_res_v.data = nullptr;
    std::pair<int, int> load_image_size;
    load_image_size.first = img->nx;
    load_image_size.second = img->ny;
    if (!clip_image_preprocess(ctx_clip, img, &img_res_v)) {
        LOG_TEE("%s: unable to preprocess image\n", __func__);
        delete[] img_res_v.data;
        return ov::Tensor{};
    }

    return clip_image_encode(ctx_clip, &img_res_v.data[0], load_image_size); // image_embd shape is 576 x 4096
}

std::pair<std::vector<std::vector<ov::Tensor>>, std::pair<size_t, size_t>> llava_image_embed_make_with_bytes_slice(struct clip_ctx* ctx_clip, const ov::Tensor& img, int max_slice_nums, int scale_resolution, int patch_size, bool never_split) {
    clip_image_u8 source{int(img.get_shape()[2]), int(img.get_shape()[1]), {img.data<uint8_t>(), img.data<uint8_t>() + img.get_size()}};
    // clip_image_u8 resized;
    // bicubic_resize(source, resized, 800, 800);

    std::vector<std::vector<clip_image_u8*>> imgs = slice_image(&source, max_slice_nums, scale_resolution, patch_size, never_split);
    std::vector<std::vector<ov::Tensor>> results;

    for (size_t i = 0; i < imgs.size(); ++i) {
        results.push_back(std::vector<ov::Tensor>());
        for (size_t j = 0; j < imgs[i].size(); ++j) {
            results[i].push_back(encode_image_with_clip(ctx_clip, imgs[i][j]));
        }
    }
    return {results, {imgs.at(0).at(0)->nx / patch_size, imgs.at(0).at(0)->ny / patch_size}};
}

void llava_image_embed_free_slice(std::vector<std::vector<struct llava_image_embed*>> embed) {
    for (size_t i = 0; i < embed.size(); ++i) {
        for (size_t j = 0; j < embed[i].size(); ++j) {
            free(embed[i][j]->embed);
            free(embed[i][j]);
        }
        embed[i] = std::vector<struct llava_image_embed*>();
    }
    embed = std::vector<std::vector<struct llava_image_embed*>>();
}
