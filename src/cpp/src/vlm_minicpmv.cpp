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



bool load_file_to_bytes(const char* path, unsigned char** bytesOut, long* sizeOut) {
    auto file = fopen(path, "rb");
    if (file == NULL) {
        LOG_TEE("%s: can't read file %s\n", __func__, path);
        return false;
    }

    fseek(file, 0, SEEK_END);
    auto fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    auto buffer = (unsigned char*)malloc(fileSize); // Allocate memory to hold the file data
    if (buffer == NULL) {
        LOG_TEE("%s: failed to alloc %ld bytes for file %s\n", __func__, fileSize, path);
        perror("Memory allocation error");
        fclose(file);
        return false;
    }
    errno = 0;
    size_t ret = fread(buffer, 1, fileSize, file); // Read the file into the buffer
    if (ferror(file)) {
        LOG_TEE("read error: %s", strerror(errno));
    }
    if (ret != (size_t)fileSize) {
        LOG_TEE("unexpectedly reached end of file");
    }
    fclose(file); // Close the file

    *bytesOut = buffer;
    *sizeOut = fileSize;
    return true;
}

static bool encode_image_with_clip(clip_ctx* ctx_clip, int n_threads, const clip_image_u8* img, float* image_embd, int* n_img_pos) {
    // std::vector<clip_image_f32*> img_res_v; // format VectN x H x W x RGB (N x 336 x 336 x 3), so interleaved RGB - different to the python implementation which is N x 3 x 336 x 336
    clip_image_f32_batch img_res_v;
    img_res_v.size = 0;
    img_res_v.data = nullptr;
    std::pair<int, int> load_image_size;
    load_image_size.first = img->nx;
    load_image_size.second = img->ny;
    auto startTime = Time::now();
    if (!clip_image_preprocess(ctx_clip, img, &img_res_v)) {
        LOG_TEE("%s: unable to preprocess image\n", __func__);
        delete[] img_res_v.data;
        return false;
    }

    auto duration_ms = get_duration_ms_until_now(startTime);
    //LOG_TEE("\n%s: image encoded in %8.2f ms by clip_image_preprocess.\n", __func__, duration_ms);

    startTime = Time::now();

    //RESAMPLER query_num minicpmv-2 64, minicpmv-2.5 96
    *n_img_pos = clip_n_patches(ctx_clip);

    bool encoded = clip_image_encode(ctx_clip, n_threads, &img_res_v.data[0], image_embd, load_image_size); // image_embd shape is 576 x 4096
    delete[] img_res_v.data;
    if (!encoded) {
        LOG_TEE("Unable to encode image\n");

        return false;
    }

    //LOG_TEE("%s: image embedding created: %d tokens\n", __func__, *n_img_pos);

    duration_ms = get_duration_ms_until_now(startTime);

    //LOG_TEE("\n%s: image encoded in %8.2f ms by CLIP (%8.2f ms per image patch)\n", __func__, duration_ms, duration_ms / *n_img_pos);

    return true;
}


bool llava_image_embed_make_with_clip_img(clip_ctx* ctx_clip, int n_threads, const clip_image_u8* img, float** image_embd_out, int* n_img_pos_out) {
    float* image_embd = (float*)malloc(clip_embd_nbytes(ctx_clip) * 6); // TODO: base on gridsize/llava model
    if (!image_embd) {
        LOG_TEE("Unable to allocate memory for image embeddings\n");
        return false;
    }

    int n_img_pos;
    if (!encode_image_with_clip(ctx_clip, n_threads, img, image_embd, &n_img_pos)) {
        LOG_TEE("%s: cannot encode image, aborting\n", __func__);
        free(image_embd);
        return false;
    }
    *image_embd_out = image_embd;
    *n_img_pos_out = n_img_pos;

    return true;
}


std::vector<std::vector<struct llava_image_embed*>> llava_image_embed_make_with_bytes_slice(struct clip_ctx* ctx_clip, int n_threads, const unsigned char* image_bytes, int image_bytes_length) {
    clip_image_u8* img = clip_image_u8_init();
    if (!clip_image_load_from_bytes(image_bytes, image_bytes_length, img)) {
        clip_image_u8_free(img);
        LOG_TEE("%s: can't load image from bytes, is it a valid image?", __func__);
        return std::vector<std::vector<struct llava_image_embed*>>();
    }

    clip_image_u8* reshaped_image = clip_image_u8_init();

    //resize to 800x800
    bicubic_resize(*img, *reshaped_image, 800, 800);
    clip_image_u8_free(img);

    std::vector<std::vector<clip_image_u8*>> imgs = slice_image(reshaped_image);
    //for (size_t i = 0; i < imgs.size(); ++i) {
    //    for (size_t j = 0; j < imgs[i].size(); ++j) {
    //        LOG_TEE("%s: %d %d\n", __func__, imgs[i][j]->nx, imgs[i][j]->ny);
    //    }
    //}
    std::vector<std::vector<llava_image_embed*>> results;

    for (size_t i = 0; i < imgs.size(); ++i) {
        results.push_back(std::vector<llava_image_embed*>());
        for (size_t j = 0; j < imgs[i].size(); ++j) {
            float* image_embed = NULL;
            int n_image_pos = 0;
            bool image_embed_result = llava_image_embed_make_with_clip_img(ctx_clip, n_threads, imgs[i][j], &image_embed, &n_image_pos);
            if (!image_embed_result) {
                clip_image_u8_free(reshaped_image);
                LOG_TEE("%s: coulnd't embed the image\n", __func__);
                return std::vector<std::vector<struct llava_image_embed*>>();
            }

            auto result = (llava_image_embed*)malloc(sizeof(llava_image_embed));
            result->embed = image_embed;
            result->n_image_pos = n_image_pos;
            results[i].push_back(result);
        }
    }
    clip_image_u8_free(reshaped_image);
    return results;
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




