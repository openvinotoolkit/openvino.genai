// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/molmo2/classes.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>

#include <nlohmann/json.hpp>

#include "visual_language/clip.hpp"

#include "utils.hpp"

namespace ov::genai {
namespace {

// Faithful port of Molmo2ImageProcessor.select_tiling (image_processing_molmo2.py).
// Divides an image of size [h, w] into up to max_num_crops tiles of size patch_size.
std::pair<int, int> select_tiling(int h, int w, int patch_size, int max_num_crops) {
    std::vector<std::pair<int, int>> tilings;
    for (int i = 1; i <= max_num_crops; ++i) {
        for (int j = 1; j <= max_num_crops; ++j) {
            if (i * j <= max_num_crops) {
                tilings.emplace_back(i, j);
            }
        }
    }
    // Sort so argmin/argmax favour smaller tilings on ties: key = (area, rows).
    std::stable_sort(tilings.begin(), tilings.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        int aa = a.first * a.second, ba = b.first * b.second;
        if (aa != ba) return aa < ba;
        return a.first < b.first;
    });

    const double orig_h = static_cast<double>(h);
    const double orig_w = static_cast<double>(w);
    std::vector<double> required_scale(tilings.size());
    for (size_t k = 0; k < tilings.size(); ++k) {
        double res_h = static_cast<double>(tilings[k].first) * patch_size;
        double res_w = static_cast<double>(tilings[k].second) * patch_size;
        // min over the two dims of candidate_resolution / original_size
        required_scale[k] = std::min(res_h / orig_h, res_w / orig_w);
    }
    bool all_lt_1 = std::all_of(required_scale.begin(), required_scale.end(), [](double s) { return s < 1.0; });
    size_t ix = 0;
    if (all_lt_1) {
        // Forced to downscale: minimize downscaling -> argmax.
        double best = -1.0;
        for (size_t k = 0; k < required_scale.size(); ++k) {
            if (required_scale[k] > best) { best = required_scale[k]; ix = k; }
        }
    } else {
        // Pick the resolution requiring the least upscaling -> argmin over scales >= 1.
        double best = std::numeric_limits<double>::max();
        for (size_t k = 0; k < required_scale.size(); ++k) {
            double s = required_scale[k] < 1.0 ? 1e10 : required_scale[k];
            if (s < best) { best = s; ix = k; }
        }
    }
    return tilings[ix];
}

// Bilinear resize (align_corners=False) of an RGB uint8 image, mirroring torchvision.Resize on a
// uint8 tensor: interpolate in float, round back to uint8, scale to [0, 1], then normalize.
// Output is a normalized float HWC buffer of size out_h * out_w * 3.
std::vector<float> resize_bilinear_norm(const std::vector<uint8_t>& src, int sh, int sw,
                                        int out_h, int out_w,
                                        const std::array<float, 3>& mean,
                                        const std::array<float, 3>& stdev) {
    std::vector<float> out(static_cast<size_t>(out_h) * out_w * 3);
    const double scale_y = static_cast<double>(sh) / out_h;
    const double scale_x = static_cast<double>(sw) / out_w;
    for (int oy = 0; oy < out_h; ++oy) {
        double fy = (oy + 0.5) * scale_y - 0.5;
        int y0 = static_cast<int>(std::floor(fy));
        double wy = fy - y0;
        int y0c = std::min(std::max(y0, 0), sh - 1);
        int y1c = std::min(std::max(y0 + 1, 0), sh - 1);
        for (int ox = 0; ox < out_w; ++ox) {
            double fx = (ox + 0.5) * scale_x - 0.5;
            int x0 = static_cast<int>(std::floor(fx));
            double wx = fx - x0;
            int x0c = std::min(std::max(x0, 0), sw - 1);
            int x1c = std::min(std::max(x0 + 1, 0), sw - 1);
            for (int c = 0; c < 3; ++c) {
                double v00 = src[(static_cast<size_t>(y0c) * sw + x0c) * 3 + c];
                double v01 = src[(static_cast<size_t>(y0c) * sw + x1c) * 3 + c];
                double v10 = src[(static_cast<size_t>(y1c) * sw + x0c) * 3 + c];
                double v11 = src[(static_cast<size_t>(y1c) * sw + x1c) * 3 + c];
                double top = v00 * (1.0 - wx) + v01 * wx;
                double bot = v10 * (1.0 - wx) + v11 * wx;
                double v = top * (1.0 - wy) + bot * wy;
                double u = std::round(v);
                u = std::min(std::max(u, 0.0), 255.0);
                out[(static_cast<size_t>(oy) * out_w + ox) * 3 + c] =
                    static_cast<float>((u / 255.0 - mean[c]) / stdev[c]);
            }
        }
    }
    return out;
}

// Convert a normalized float HWC image [H, W, 3] into patches [n_patches, patch*patch*3], mirroring
// Molmo2 batch_pixels_to_patches. Patch p = (ph * w_patches + pw); within a patch the order is
// (dy, dx, c).
void append_patches(const std::vector<float>& img, int H, int W, int patch,
                    std::vector<float>& out /* appended */) {
    int h_patches = H / patch;
    int w_patches = W / patch;
    const int ppp = patch * patch * 3;
    size_t base = out.size();
    out.resize(base + static_cast<size_t>(h_patches) * w_patches * ppp);
    float* dst = out.data() + base;
    for (int ph = 0; ph < h_patches; ++ph) {
        for (int pw = 0; pw < w_patches; ++pw) {
            float* patch_dst = dst + (static_cast<size_t>(ph) * w_patches + pw) * ppp;
            int idx = 0;
            for (int dy = 0; dy < patch; ++dy) {
                for (int dx = 0; dx < patch; ++dx) {
                    int y = ph * patch + dy;
                    int x = pw * patch + dx;
                    const float* px = img.data() + (static_cast<size_t>(y) * W + x) * 3;
                    patch_dst[idx++] = px[0];
                    patch_dst[idx++] = px[1];
                    patch_dst[idx++] = px[2];
                }
            }
        }
    }
}

// Port of Molmo2 arange_for_pooling: pad idx_arr [H, W] (centered, -1 fill) so H and W are divisible
// by (pool_h, pool_w), then group into [out_h, out_w, pool_h * pool_w].
std::vector<int64_t> arange_for_pooling(const std::vector<int64_t>& idx, int H, int W,
                                        int pool_h, int pool_w, int& out_h, int& out_w) {
    out_h = (H + pool_h - 1) / pool_h;
    out_w = (W + pool_w - 1) / pool_w;
    int h_pad = out_h * pool_h - H;
    int w_pad = out_w * pool_w - W;
    int top = h_pad / 2;
    int left = w_pad / 2;
    int padded_h = H + h_pad;
    int padded_w = W + w_pad;
    std::vector<int64_t> padded(static_cast<size_t>(padded_h) * padded_w, -1);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            padded[static_cast<size_t>(top + y) * padded_w + (left + x)] = idx[static_cast<size_t>(y) * W + x];
        }
    }
    std::vector<int64_t> out(static_cast<size_t>(out_h) * out_w * pool_h * pool_w);
    for (int oy = 0; oy < out_h; ++oy) {
        for (int ox = 0; ox < out_w; ++ox) {
            for (int dy = 0; dy < pool_h; ++dy) {
                for (int dx = 0; dx < pool_w; ++dx) {
                    int py = oy * pool_h + dy;
                    int px = ox * pool_w + dx;
                    out[((static_cast<size_t>(oy) * out_w + ox) * (pool_h * pool_w)) + (dy * pool_w + dx)] =
                        padded[static_cast<size_t>(py) * padded_w + px];
                }
            }
        }
    }
    return out;
}

struct Molmo2Preprocessed {
    std::vector<float> pixel_values;   // [n_crops_total, n_patches, pixels_per_patch]
    int n_crops_total = 0;
    std::vector<int64_t> pooled_idx;   // [num_pooled, pool_area]
    int num_pooled = 0;
    int pool_area = 0;
    int resized_h = 0, resized_w = 0;  // low-resolution (global) pooled grid
    int high_h = 0, high_w = 0;        // high-resolution pooled grid
    int n_patches = 0, pixels_per_patch = 0;
};

// Port of Molmo2 image_to_patches_and_grids (image branch). Produces the patchified crops, the
// pooling indices, and the (low-res, high-res) pooled grids for a single image.
Molmo2Preprocessed preprocess_molmo2_image(const clip_image_u8& img, const ProcessorConfig& cfg) {
    const int base_h = static_cast<int>(cfg.size_height);
    const int base_w = static_cast<int>(cfg.size_width);
    const int patch = static_cast<int>(cfg.patch_size);
    OPENVINO_ASSERT(base_h == base_w, "Molmo2 expects a square base image input size");
    const int max_crops = static_cast<int>(cfg.molmo2_max_crops);
    const int left_margin = static_cast<int>(cfg.molmo2_overlap_margin_left);
    const int right_margin = static_cast<int>(cfg.molmo2_overlap_margin_right);
    const int pool_h = static_cast<int>(cfg.molmo2_pooling_h);
    const int pool_w = static_cast<int>(cfg.molmo2_pooling_w);
    const std::array<float, 3> mean = cfg.image_mean;
    const std::array<float, 3> stdev = cfg.image_std;

    const int crop_size = base_h;
    const int crop_patches = base_h / patch;          // e.g. 27
    const int crop_window_patches = crop_patches - (right_margin + left_margin);
    const int crop_window_size = crop_window_patches * patch;
    const int total_margin_pixels = patch * (right_margin + left_margin);

    Molmo2Preprocessed out;
    out.n_patches = crop_patches * crop_patches;
    out.pixels_per_patch = patch * patch * 3;
    out.pool_area = pool_h * pool_w;

    const int orig_h = img.ny;
    const int orig_w = img.nx;

    // ---- overlapping high-resolution crops ----
    std::pair<int, int> tiling = select_tiling(std::max(orig_h - total_margin_pixels, 1),
                                               std::max(orig_w - total_margin_pixels, 1),
                                               crop_window_size, max_crops);
    int tiling_h = tiling.first, tiling_w = tiling.second;
    int src_h = tiling_h * crop_window_size + total_margin_pixels;
    int src_w = tiling_w * crop_window_size + total_margin_pixels;
    std::vector<float> src = resize_bilinear_norm(img.buf, orig_h, orig_w, src_h, src_w, mean, stdev);
    int src_h_p = src_h / patch;
    int src_w_p = src_w / patch;

    int n_crops = tiling_h * tiling_w;
    // patch_idx_arr accumulated crop-by-crop as [n_crops, crop_patches, crop_patches]
    std::vector<int64_t> patch_idx(static_cast<size_t>(n_crops) * crop_patches * crop_patches);
    // crop pixels appended after the global image below; store crops separately first
    std::vector<float> crop_patches_buf;
    crop_patches_buf.reserve(static_cast<size_t>(n_crops) * out.n_patches * out.pixels_per_patch);
    int on_crop = 0;
    for (int i = 0; i < tiling_h; ++i) {
        int y0 = i * crop_window_size;
        for (int j = 0; j < tiling_w; ++j) {
            int x0 = j * crop_window_size;
            // Extract the crop [crop_size, crop_size, 3] from src.
            std::vector<float> crop(static_cast<size_t>(crop_size) * crop_size * 3);
            for (int y = 0; y < crop_size; ++y) {
                const float* srow = src.data() + (static_cast<size_t>(y0 + y) * src_w + x0) * 3;
                std::copy(srow, srow + static_cast<size_t>(crop_size) * 3,
                          crop.data() + static_cast<size_t>(y) * crop_size * 3);
            }
            append_patches(crop, crop_size, crop_size, patch, crop_patches_buf);

            // Build the patch index grid for this crop and mask overlap regions.
            int64_t* pidx = patch_idx.data() + static_cast<size_t>(on_crop) * crop_patches * crop_patches;
            for (int p = 0; p < crop_patches * crop_patches; ++p) {
                pidx[p] = static_cast<int64_t>(p) + static_cast<int64_t>(on_crop) * crop_patches * crop_patches;
            }
            auto mask_rows = [&](int r0, int r1) {
                for (int r = r0; r < r1; ++r)
                    for (int cc = 0; cc < crop_patches; ++cc)
                        pidx[static_cast<size_t>(r) * crop_patches + cc] = -1;
            };
            auto mask_cols = [&](int c0, int c1) {
                for (int r = 0; r < crop_patches; ++r)
                    for (int cc = c0; cc < c1; ++cc)
                        pidx[static_cast<size_t>(r) * crop_patches + cc] = -1;
            };
            if (i != 0) mask_rows(0, left_margin);
            if (j != 0) mask_cols(0, left_margin);
            if (i != tiling_h - 1) mask_rows(crop_patches - right_margin, crop_patches);
            if (j != tiling_w - 1) mask_cols(crop_patches - right_margin, crop_patches);
            ++on_crop;
        }
    }

    // Reorder patch_idx to left-to-right: [tiling_h, tiling_w, cph, cpw] -> [tiling_h, cph, tiling_w, cpw],
    // flatten, drop the masked (-1) entries, giving a [src_h_p, src_w_p] grid.
    std::vector<int64_t> patch_idx_grid;
    patch_idx_grid.reserve(static_cast<size_t>(src_h_p) * src_w_p);
    for (int ti = 0; ti < tiling_h; ++ti) {
        for (int cy = 0; cy < crop_patches; ++cy) {
            for (int tj = 0; tj < tiling_w; ++tj) {
                for (int cx = 0; cx < crop_patches; ++cx) {
                    int crop_index = ti * tiling_w + tj;
                    int64_t v = patch_idx[(static_cast<size_t>(crop_index) * crop_patches + cy) * crop_patches + cx];
                    if (v >= 0) patch_idx_grid.push_back(v);
                }
            }
        }
    }
    OPENVINO_ASSERT(static_cast<int>(patch_idx_grid.size()) == src_h_p * src_w_p,
                    "Molmo2 preprocessing: unexpected number of high-resolution patches");

    int high_h = 0, high_w = 0;
    std::vector<int64_t> high_pool = arange_for_pooling(patch_idx_grid, src_h_p, src_w_p, pool_h, pool_w, high_h, high_w);

    // ---- global (low-resolution) resized image ----
    std::vector<float> resized = resize_bilinear_norm(img.buf, orig_h, orig_w, base_h, base_w, mean, stdev);
    std::vector<float> global_patches;
    append_patches(resized, base_h, base_w, patch, global_patches);
    std::vector<int64_t> resize_idx_grid(static_cast<size_t>(crop_patches) * crop_patches);
    for (int p = 0; p < crop_patches * crop_patches; ++p) resize_idx_grid[p] = p;
    int resized_h = 0, resized_w = 0;
    std::vector<int64_t> resize_pool = arange_for_pooling(resize_idx_grid, crop_patches, crop_patches,
                                                          pool_h, pool_w, resized_h, resized_w);

    // Global image goes first: shift high-resolution patch indices by one crop of patches.
    const int64_t global_offset = static_cast<int64_t>(crop_patches) * crop_patches;
    for (int64_t& v : high_pool) {
        if (v >= 0) v += global_offset;
    }

    // ---- assemble outputs ----
    out.n_crops_total = 1 + n_crops;
    out.pixel_values.reserve(static_cast<size_t>(out.n_crops_total) * out.n_patches * out.pixels_per_patch);
    out.pixel_values.insert(out.pixel_values.end(), global_patches.begin(), global_patches.end());
    out.pixel_values.insert(out.pixel_values.end(), crop_patches_buf.begin(), crop_patches_buf.end());

    out.pooled_idx.reserve(resize_pool.size() + high_pool.size());
    out.pooled_idx.insert(out.pooled_idx.end(), resize_pool.begin(), resize_pool.end());
    out.pooled_idx.insert(out.pooled_idx.end(), high_pool.begin(), high_pool.end());
    out.num_pooled = static_cast<int>(out.pooled_idx.size() / out.pool_area);
    out.resized_h = resized_h;
    out.resized_w = resized_w;
    out.high_h = high_h;
    out.high_w = high_w;
    return out;
}

// Port of Molmo2VideoProcessor.sample_frames: indices of the frames to keep from a video of total_frames frames
// recorded at fps. NumPy semantics are kept: linspace + astype(int) truncates, np.round rounds half to even.
std::vector<size_t> sample_frames(size_t total_frames, float fps, const std::string& mode, size_t num_frames, float max_fps) {
    auto linspace_int = [](double stop, size_t num) {
        std::vector<size_t> out(num);
        const double step = num > 1 ? stop / static_cast<double>(num - 1) : 0.0;
        for (size_t i = 0; i < num; ++i) {
            out[i] = static_cast<size_t>(i + 1 == num && num > 1 ? stop : static_cast<double>(i) * step);
        }
        return out;
    };
    OPENVINO_ASSERT(mode == "uniform_last_frame", "Molmo2: unsupported frame_sample_mode '", mode, "'");
    if (max_fps > 0.0f) {
        const double duration = static_cast<double>(total_frames) / fps;
        if (total_frames <= 2) {
            std::vector<size_t> all(total_frames);
            std::iota(all.begin(), all.end(), 0);
            return all;
        }
        if (duration > static_cast<double>(num_frames - 1) / max_fps) {
            return linspace_int(static_cast<double>(total_frames - 1), std::min(num_frames, total_frames));
        }
        const double step = static_cast<double>(fps) / max_fps;
        const double stop = static_cast<double>(total_frames - 1);
        std::vector<double> float_indices;
        for (size_t i = 0; static_cast<double>(i) * step < stop; ++i) {
            float_indices.push_back(static_cast<double>(i) * step);
        }
        if (float_indices.empty() || std::nearbyint(float_indices.back()) != stop) {
            float_indices.push_back(stop);
        }
        std::vector<size_t> indices;
        for (double v : float_indices) {
            indices.push_back(static_cast<size_t>(std::nearbyint(v)));
        }
        return indices;
    }
    return linspace_int(static_cast<double>(total_frames - 1), std::min(num_frames, total_frames));
}

} // namespace

EncodedImage VisionEncoderMolmo2::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();

    ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);

    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    Molmo2Preprocessed pre = preprocess_molmo2_image(input_image, config);

    // build_batched_images for a single image (N == 1): add the batch dimension.
    ov::Tensor images_t(ov::element::f32, {1, static_cast<size_t>(pre.n_crops_total),
                                           static_cast<size_t>(pre.n_patches),
                                           static_cast<size_t>(pre.pixels_per_patch)});
    std::copy(pre.pixel_values.begin(), pre.pixel_values.end(), images_t.data<float>());

    ov::Tensor pooled_t(ov::element::i64, {1, static_cast<size_t>(pre.num_pooled),
                                           static_cast<size_t>(pre.pool_area)});
    std::copy(pre.pooled_idx.begin(), pre.pooled_idx.end(), pooled_t.data<int64_t>());

    encoder.set_tensor("images", images_t);
    encoder.set_tensor("pooled_patches_idx", pooled_t);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_tensor("last_hidden_state");
    const ov::Shape& out_shape = infer_output.get_shape();  // [num_pooled, hidden]
    size_t num_tokens = out_shape[0];
    size_t hidden = out_shape[1];
    OPENVINO_ASSERT(static_cast<int>(num_tokens) == pre.num_pooled,
                    "Molmo2 vision backbone returned ", num_tokens, " tokens, expected ", pre.num_pooled);

    ov::Tensor image_features(infer_output.get_element_type(), ov::Shape{1, num_tokens, hidden});
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    EncodedImage encoded;
    encoded.resized_source = std::move(image_features);
    encoded.resized_source_size = ImageSize{static_cast<size_t>(pre.resized_h), static_cast<size_t>(pre.resized_w)};
    encoded.patches_grid = {pre.high_h, pre.high_w};
    encoded.num_image_tokens = num_tokens;
    return encoded;
}

EncodedVideo VisionEncoderMolmo2::encode_frames(const std::vector<ov::Tensor>& frames) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    const VideoProcessorConfig& config = m_video_processor_config;

    const int base_h = static_cast<int>(config.size_height);
    const int base_w = static_cast<int>(config.size_width);
    const int patch = static_cast<int>(config.patch_size);
    const int pool_h = static_cast<int>(config.molmo2_pooling_h);
    const int pool_w = static_cast<int>(config.molmo2_pooling_w);
    const int crop_patches_h = base_h / patch;
    const int crop_patches_w = base_w / patch;
    const size_t n_patches = static_cast<size_t>(crop_patches_h) * crop_patches_w;
    const size_t pixels_per_patch = static_cast<size_t>(patch) * patch * 3;

    std::vector<int64_t> patch_idx(n_patches);
    std::iota(patch_idx.begin(), patch_idx.end(), 0);
    int grid_h = 0, grid_w = 0;
    const std::vector<int64_t> frame_pool = arange_for_pooling(patch_idx, crop_patches_h, crop_patches_w, pool_h, pool_w, grid_h, grid_w);
    const size_t pool_area = static_cast<size_t>(pool_h) * pool_w;
    const size_t tokens_per_frame = frame_pool.size() / pool_area;

    std::vector<float> pixel_values;
    pixel_values.reserve(frames.size() * n_patches * pixels_per_patch);
    std::vector<int64_t> pooled_idx;
    pooled_idx.reserve(frames.size() * frame_pool.size());
    for (size_t f = 0; f < frames.size(); ++f) {
        clip_image_u8 frame = tensor_to_clip_image_u8(frames[f]);
        std::vector<float> resized = resize_bilinear_norm(frame.buf, frame.ny, frame.nx, base_h, base_w, config.image_mean, config.image_std);
        append_patches(resized, base_h, base_w, patch, pixel_values);
        const int64_t offset = static_cast<int64_t>(f * n_patches);
        for (int64_t v : frame_pool) {
            pooled_idx.push_back(v >= 0 ? v + offset : v);
        }
    }

    ov::Tensor images_t(ov::element::f32, {1, frames.size(), n_patches, pixels_per_patch});
    std::copy(pixel_values.begin(), pixel_values.end(), images_t.data<float>());
    ov::Tensor pooled_t(ov::element::i64, {1, frames.size() * tokens_per_frame, pool_area});
    std::copy(pooled_idx.begin(), pooled_idx.end(), pooled_t.data<int64_t>());
    encoder.set_tensor("images", images_t);
    encoder.set_tensor("pooled_patches_idx", pooled_t);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_tensor("last_hidden_state");  // [frames * tokens_per_frame, hidden]
    const size_t num_tokens = infer_output.get_shape().at(0);
    OPENVINO_ASSERT(num_tokens == frames.size() * tokens_per_frame,
                    "Molmo2 vision backbone returned ", num_tokens, " video tokens, expected ", frames.size() * tokens_per_frame);
    ov::Tensor video_features(infer_output.get_element_type(), ov::Shape{1, num_tokens, infer_output.get_shape().at(1)});
    std::memcpy(video_features.data(), infer_output.data(), infer_output.get_byte_size());

    EncodedVideo encoded;
    encoded.video_features = std::move(video_features);
    encoded.num_video_tokens = num_tokens;
    encoded.resized_source_size = ImageSize{static_cast<size_t>(grid_h), static_cast<size_t>(grid_w)};
    encoded.frame_num = frames.size();
    return encoded;
}

InputsEmbedderMolmo2::InputsEmbedderMolmo2(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) {
        read_molmo2_configs(model_dir);
        patch_chat_template();
    }

InputsEmbedderMolmo2::InputsEmbedderMolmo2(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
        read_molmo2_configs(config_dir_path);
        patch_chat_template();
    }

void InputsEmbedderMolmo2::read_molmo2_configs(const std::filesystem::path& config_dir) {
    auto read_json = [&](const char* name) {
        std::ifstream stream(config_dir / name);
        return stream.is_open() ? nlohmann::json::parse(stream) : nlohmann::json::object();
    };
    auto read = [](const nlohmann::json& json, const char* key, auto& value) {
        if (json.contains(key) && !json.at(key).is_null()) {
            value = json.at(key).get<std::decay_t<decltype(value)>>();
        }
    };
    const nlohmann::json processor = read_json("processor_config.json");
    read(processor, "video_use_col_tokens", m_video_use_col_tokens);
    read(processor, "use_frame_special_tokens", m_use_frame_special_tokens);
    const nlohmann::json video = read_json("video_preprocessor_config.json");
    read(video, "frame_sample_mode", m_frame_sample_mode);
    read(video, "num_frames", m_num_frames);
    read(video, "max_fps", m_max_fps);
    read(video, "sampling_fps", m_sampling_fps);
}

bool InputsEmbedderMolmo2::has_token_type_ids() const {
    return true;
}

void InputsEmbedderMolmo2::patch_chat_template() {
    // Molmo2 places all image special tokens at the very front of the sequence, before the "User:"
    // role prefix, and separates turns without additional markers. The normalized prompt produced by
    // normalize_prompt already contains the expanded image tokens and the "User:" prefix, so the
    // chat template only needs to concatenate message contents and append the assistant cue.
    const std::string molmo2_template =
        "{% for message in messages %}"
        "{% if message['role'] == 'assistant' %} {{ message['content'] }}"
        "{% else %}{{ message['content'] }}{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %} Assistant:{% endif %}";
    m_tokenizer.set_chat_template(molmo2_template);
}

std::string InputsEmbedderMolmo2::build_image_tokens(size_t resized_h, size_t resized_w,
                                                     size_t high_h, size_t high_w) const {
    const std::string& im_start = m_vlm_config.molmo2_im_start;
    const std::string& im_end = m_vlm_config.molmo2_im_end;
    const std::string& im_patch = m_vlm_config.molmo2_im_patch;
    const std::string& im_col = m_vlm_config.molmo2_im_col;
    const std::string& low_res_im_start = m_vlm_config.molmo2_low_res_im_start;
    const ProcessorConfig m_processor_config = m_vision_encoder->get_processor_config();

    std::string result;
    // Low-resolution (global) section.
    {
        bool use_col = m_processor_config.molmo2_use_single_crop_col_tokens < 0
                           ? m_processor_config.molmo2_image_use_col_tokens
                           : (m_processor_config.molmo2_use_single_crop_col_tokens != 0);
        const std::string& start_tok = m_processor_config.molmo2_use_single_crop_start_token ? low_res_im_start : im_start;
        result += start_tok;
        for (size_t r = 0; r < resized_h; ++r) {
            for (size_t c = 0; c < resized_w; ++c) result += im_patch;
            if (use_col) result += im_col;
        }
        result += im_end;
    }
    // High-resolution section.
    {
        bool use_col = m_processor_config.molmo2_image_use_col_tokens;
        result += im_start;
        for (size_t r = 0; r < high_h; ++r) {
            for (size_t c = 0; c < high_w; ++c) result += im_patch;
            if (use_col) result += im_col;
        }
        result += im_end;
    }
    return result;
}

std::vector<ov::genai::EncodedImage> InputsEmbedderMolmo2::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;
    ov::AnyMap vision_config = {{"patch_size", m_vlm_config.vision_config_patch_size}};
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image, vision_config));
    }
    return embeds;
}

NormalizedPrompt InputsEmbedderMolmo2::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    return normalize_prompt(prompt, base_id, 0, images, {});
}

std::vector<ov::genai::EncodedVideo> InputsEmbedderMolmo2::encode_videos(const std::vector<ov::Tensor>& videos,
                                                                         const std::vector<VideoMetadata>& videos_metadata) {
    OPENVINO_ASSERT(videos_metadata.empty() || videos_metadata.size() == videos.size(),
                    "Number of videos and videos metadata must match if metadata provided.");
    std::vector<EncodedVideo> encoded_videos;
    for (size_t i = 0; i < videos.size(); ++i) {
        const size_t total_frames = videos[i].get_shape().at(0);
        VideoMetadata metadata = i < videos_metadata.size() ? videos_metadata[i] : VideoMetadata{};
        if (metadata.fps <= 0.0f) {
            // Without metadata the frames are treated as sampled at the model's sampling rate.
            metadata.fps = m_sampling_fps;
        }
        if (metadata.frames_indices.empty()) {
            metadata.frames_indices = sample_frames(total_frames, metadata.fps, m_frame_sample_mode, m_num_frames, m_max_fps);
        }
        const ov::Tensor sampled = sample_video_if_needed(videos[i], metadata);
        EncodedVideo encoded = m_vision_encoder->encode_frames(to_single_image_tensors({sampled}));
        encoded.metadata = std::move(metadata);
        encoded_videos.emplace_back(std::move(encoded));
    }
    return encoded_videos;
}

std::string InputsEmbedderMolmo2::build_video_tokens(const EncodedVideo& video) const {
    const std::string& start = m_use_frame_special_tokens ? std::string("<frame_start>") : m_vlm_config.molmo2_im_start;
    const std::string& end = m_use_frame_special_tokens ? std::string("<frame_end>") : m_vlm_config.molmo2_im_end;
    const size_t grid_h = video.resized_source_size.height;
    const size_t grid_w = video.resized_source_size.width;
    std::string result;
    for (size_t f = 0; f < video.frame_num; ++f) {
        char timestamp[32];
        std::snprintf(timestamp, sizeof(timestamp), "%.1f",
                      static_cast<double>(video.metadata.frames_indices.at(f)) / static_cast<double>(video.metadata.fps));
        result += (f > 0 ? " " : "") + std::string(timestamp) + " " + start;
        for (size_t r = 0; r < grid_h; ++r) {
            for (size_t c = 0; c < grid_w; ++c) result += m_vlm_config.molmo2_im_patch;
            if (m_video_use_col_tokens) result += m_vlm_config.molmo2_im_col;
        }
        result += end;
    }
    return result;
}

NormalizedPrompt InputsEmbedderMolmo2::normalize_prompt(const std::string& prompt,
                                                        size_t base_image_id,
                                                        size_t base_video_id,
                                                        const std::vector<EncodedImage>& images,
                                                        const std::vector<EncodedVideo>& videos) const {
    const std::string& image_tag = m_vlm_config.image_token;  // "<|image|>"
    const std::string& video_tag = m_vlm_config.video_token;  // "<|video|>"
    auto [with_images, images_sequence] = normalize(prompt, image_tag, image_tag, base_image_id, images.size());
    auto [normalized, videos_sequence] = normalize(with_images, video_tag, video_tag, base_video_id, videos.size(), VisionType::VIDEO);

    // Molmo2's chat template moves all visual inputs in front of the "User:" role prefix: drop the tags from the text.
    std::string text_part;
    for (size_t pos = 0; pos < normalized.size();) {
        const size_t image_pos = normalized.find(image_tag, pos);
        const size_t video_pos = normalized.find(video_tag, pos);
        const size_t found = std::min(image_pos, video_pos);
        if (found == std::string::npos) {
            text_part.append(normalized, pos, std::string::npos);
            break;
        }
        text_part.append(normalized, pos, found - pos);
        pos = found + (found == image_pos ? image_tag.size() : video_tag.size());
    }

    std::string unified_prompt;
    for (size_t new_image_id : images_sequence) {
        const EncodedImage& enc = images.at(new_image_id - base_image_id);
        unified_prompt += build_image_tokens(enc.resized_source_size.height, enc.resized_source_size.width,
                                             static_cast<size_t>(enc.patches_grid.first),
                                             static_cast<size_t>(enc.patches_grid.second));
    }
    for (size_t new_video_id : videos_sequence) {
        unified_prompt += build_video_tokens(videos.at(new_video_id - base_video_id));
    }
    unified_prompt += "User: ";
    unified_prompt += text_part;
    return {std::move(unified_prompt), std::move(images_sequence), std::move(videos_sequence)};
}

ov::Tensor InputsEmbedderMolmo2::get_inputs_embeds(const std::string& prompt, const std::vector<EncodedImage>& images, VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& image_sequence) {
    OPENVINO_THROW(
        "[InputsEmbedderMolmo2] get_inputs_embeds is not supported for Molmo2 models because "
        "token_type_ids are required to build the prefill bidirectional image attention mask. "
        "Use get_inputs_embeds_with_token_type_ids instead.");
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderMolmo2::get_inputs_embeds_with_token_type_ids(const std::string& unified_prompt, const std::vector<EncodedImage>& images, VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& image_sequence) {
    return get_inputs_embeds_with_token_type_ids(unified_prompt, images, {}, metrics, recalculate_merged_embeddings, image_sequence, {}, {});
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderMolmo2::get_inputs_embeds_with_token_type_ids(
    const std::string& unified_prompt,
    const std::vector<EncodedImage>& images,
    const std::vector<EncodedVideo>& videos,
    VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& image_sequence,
    const std::vector<size_t>& videos_sequence,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    // Vision features in prompt order: images first, then videos (see normalize_prompt).
    std::vector<const ov::Tensor*> vision_features;
    for (size_t id : image_sequence) vision_features.push_back(&images.at(id).resized_source);
    for (size_t id : videos_sequence) vision_features.push_back(&videos.at(id).video_features);
    return merge_embeddings(unified_prompt, vision_features, metrics);
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderMolmo2::merge_embeddings(const std::string& unified_prompt,
                                                                         const std::vector<const ov::Tensor*>& vision_features,
                                                                         VLMPerfMetrics& metrics) {
    // Tokenize the normalized prompt (no BOS is added by the tokenizer for the templated path).
    ov::Tensor encoded_input_ids = get_encoded_input_ids(unified_prompt, metrics);
    const size_t base_len = encoded_input_ids.get_size();

    // Molmo2 prepends a single BOS token at the very start of the sequence, mirroring the reference
    // Molmo2Processor.insert_bos. Every prompt embedded outside of an incremental chat (including the continuous
    // batching pipeline, which re-embeds the whole templated history on each turn) is a complete sequence and gets the
    // BOS. In an incremental chat (VLMPipeline start_chat) later turns extend the existing KV cache, so the BOS belongs
    // only to the first prefill (m_prev_hist_length == 0, set by get_encoded_input_ids); adding it on every turn would
    // leave untracked tokens in the KV cache that the chat rollback cannot trim.
    const int64_t molmo_bos = 151645;
    const bool prepend_bos = !m_is_chat_conversation || m_prev_hist_length == 0;
    const size_t seq_len = prepend_bos ? base_len + 1 : base_len;
    ov::Tensor input_ids(ov::element::i64, ov::Shape{1, seq_len});
    int64_t* ids = input_ids.data<int64_t>();
    if (prepend_bos) {
        ids[0] = molmo_bos;
        std::memcpy(ids + 1, encoded_input_ids.data<int64_t>(), base_len * sizeof(int64_t));
    } else {
        std::memcpy(ids, encoded_input_ids.data<int64_t>(), base_len * sizeof(int64_t));
    }

    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = get_text_embedding(req, input_ids, metrics);

    const size_t hidden = text_embeds.get_shape().at(2);
    ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
    std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());

    // token_type_ids: mark every image special token so the LM applies bidirectional image attention.
    const int64_t patch_id = m_vlm_config.molmo2_image_patch_id;
    const std::array<int64_t, 8> image_token_ids = {
        m_vlm_config.molmo2_image_patch_id,
        m_vlm_config.molmo2_image_col_id,
        m_vlm_config.molmo2_image_start_token_id,
        m_vlm_config.molmo2_low_res_image_start_token_id,
        m_vlm_config.molmo2_frame_start_token_id,
        m_vlm_config.molmo2_image_end_token_id,
        m_vlm_config.molmo2_frame_end_token_id,
        m_vlm_config.molmo2_image_low_res_id,
    };
    ov::Tensor token_type_ids(ov::element::i64, ov::Shape{1, seq_len});
    int64_t* tt = token_type_ids.data<int64_t>();
    for (size_t i = 0; i < seq_len; ++i) {
        int64_t id = ids[i];
        tt[i] = std::find(image_token_ids.begin(), image_token_ids.end(), id) != image_token_ids.end() ? 1 : 0;
    }

    if (vision_features.empty()) {
        return {inputs_embeds, token_type_ids};
    }

    // Additive merge: Molmo2 adds pooled image / frame features to the "<im_patch>" placeholder embeddings.
    // Concatenate the vision embeddings in prompt order and add each row at the next "<im_patch>" position.
    float* embeds_data = inputs_embeds.data<float>();
    size_t cur_image = 0;                  // index into vision_features
    const float* cur_vision = vision_features.at(0)->data<const float>();
    size_t cur_vision_rows = vision_features.at(0)->get_shape().at(1);
    size_t cur_row = 0;

    for (size_t i = 0; i < seq_len; ++i) {
        if (ids[i] != patch_id) {
            continue;
        }
        // Advance to the next image that still has rows.
        while (cur_vision != nullptr && cur_row >= cur_vision_rows) {
            ++cur_image;
            if (cur_image >= vision_features.size()) {
                cur_vision = nullptr;
                break;
            }
            cur_vision = vision_features.at(cur_image)->data<const float>();
            cur_vision_rows = vision_features.at(cur_image)->get_shape().at(1);
            cur_row = 0;
        }
        OPENVINO_ASSERT(cur_vision != nullptr,
                        "Molmo2: number of image placeholder tokens exceeds available vision embeddings");
        float* dst = embeds_data + i * hidden;
        const float* srow = cur_vision + cur_row * hidden;
        for (size_t h = 0; h < hidden; ++h) {
            dst[h] += srow[h];
        }
        ++cur_row;
    }

    return {inputs_embeds, token_type_ids};
}

} // namespace ov::genai
