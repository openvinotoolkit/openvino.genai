// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/jvlm/classes.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

#include "visual_language/clip.hpp"
#include "json_utils.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {

// PyTorch/torchvision bilinear resize with align_corners=False, antialias=False.
// Input is RGB uint8 (HWC). Output is float32 in [0, 1] (HWC), rounded to the
// nearest uint8 value first to match torchvision's uint8 resize semantics.
std::vector<float> bilinear_resize_u8_to_unit(const clip_image_u8& img,
                                              int out_h,
                                              int out_w) {
    const int in_h = img.ny;
    const int in_w = img.nx;
    std::vector<float> out(static_cast<size_t>(out_h) * out_w * 3);
    const double scale_h = static_cast<double>(in_h) / out_h;
    const double scale_w = static_cast<double>(in_w) / out_w;
    for (int oy = 0; oy < out_h; ++oy) {
        double src_y = (oy + 0.5) * scale_h - 0.5;
        int y0 = static_cast<int>(std::floor(src_y));
        double wy = src_y - y0;
        int y0c = std::min(std::max(y0, 0), in_h - 1);
        int y1c = std::min(std::max(y0 + 1, 0), in_h - 1);
        for (int ox = 0; ox < out_w; ++ox) {
            double src_x = (ox + 0.5) * scale_w - 0.5;
            int x0 = static_cast<int>(std::floor(src_x));
            double wx = src_x - x0;
            int x0c = std::min(std::max(x0, 0), in_w - 1);
            int x1c = std::min(std::max(x0 + 1, 0), in_w - 1);
            for (int c = 0; c < 3; ++c) {
                double Ia = img.buf[(static_cast<size_t>(y0c) * in_w + x0c) * 3 + c];
                double Ib = img.buf[(static_cast<size_t>(y0c) * in_w + x1c) * 3 + c];
                double Ic = img.buf[(static_cast<size_t>(y1c) * in_w + x0c) * 3 + c];
                double Id = img.buf[(static_cast<size_t>(y1c) * in_w + x1c) * 3 + c];
                double top = Ia * (1.0 - wx) + Ib * wx;
                double bot = Ic * (1.0 - wx) + Id * wx;
                double val = top * (1.0 - wy) + bot * wy;
                // Match torchvision uint8 resize: round-half-to-even, clip, then /255.
                double r = std::nearbyint(val);
                if (r < 0.0)
                    r = 0.0;
                if (r > 255.0)
                    r = 255.0;
                out[(static_cast<size_t>(oy) * out_w + ox) * 3 + c] = static_cast<float>(r / 255.0);
            }
        }
    }
    return out;
}

void normalize_inplace(std::vector<float>& x, const JinaVLMProcParams& p) {
    if (p.normalization_method == "gaussian") {
        for (size_t i = 0; i < x.size(); i += 3) {
            for (int c = 0; c < 3; ++c) {
                x[i + c] = (x[i + c] - p.image_mean[c]) / p.image_std[c];
            }
        }
    } else {  // minmax: image_min + x * (image_max - image_min)
        const float span = p.image_max - p.image_min;
        for (float& v : x) {
            v = p.image_min + v * span;
        }
    }
}

// Patchify an HWC float image [h, w, 3] into [n_patches, patch*patch*3].
// Matches numpy reshape/transpose in image_processing_jvlm.patchify.
void patchify(const std::vector<float>& src,
              int h,
              int w,
              int patch,
              std::vector<float>& dst,
              int& n_patches) {
    const int hp = h / patch;
    const int wp = w / patch;
    n_patches = hp * wp;
    const int ppp = patch * patch * 3;
    dst.assign(static_cast<size_t>(n_patches) * ppp, 0.0f);
    for (int ph = 0; ph < hp; ++ph) {
        for (int pw = 0; pw < wp; ++pw) {
            const int patch_idx = ph * wp + pw;
            float* out = dst.data() + static_cast<size_t>(patch_idx) * ppp;
            int k = 0;
            for (int iy = 0; iy < patch; ++iy) {
                for (int ix = 0; ix < patch; ++ix) {
                    const int y = ph * patch + iy;
                    const int x = pw * patch + ix;
                    const float* pix = src.data() + (static_cast<size_t>(y) * w + x) * 3;
                    out[k++] = pix[0];
                    out[k++] = pix[1];
                    out[k++] = pix[2];
                }
            }
        }
    }
}

// Per-patch mask mean for an HW mask [h, w] -> [n_patches].
void patchify_mask_mean(const std::vector<float>& mask,
                        int h,
                        int w,
                        int patch,
                        std::vector<float>& dst) {
    const int hp = h / patch;
    const int wp = w / patch;
    dst.assign(static_cast<size_t>(hp) * wp, 0.0f);
    for (int ph = 0; ph < hp; ++ph) {
        for (int pw = 0; pw < wp; ++pw) {
            double acc = 0.0;
            for (int iy = 0; iy < patch; ++iy) {
                for (int ix = 0; ix < patch; ++ix) {
                    acc += mask[static_cast<size_t>(ph * patch + iy) * w + (pw * patch + ix)];
                }
            }
            dst[static_cast<size_t>(ph) * wp + pw] = static_cast<float>(acc / (patch * patch));
        }
    }
}

// Molmo tiling selection (image_processing_jvlm._molmo_select_tiling).
std::pair<int, int> select_tiling(int h, int w, int crop_window_size, int max_crops) {
    std::vector<std::pair<int, int>> tilings;
    for (int i = 1; i <= max_crops; ++i) {
        for (int j = 1; j <= max_crops; ++j) {
            if (i * j <= max_crops) {
                tilings.emplace_back(i, j);
            }
        }
    }
    std::sort(tilings.begin(), tilings.end(), [](const auto& a, const auto& b) {
        int pa = a.first * a.second;
        int pb = b.first * b.second;
        if (pa != pb)
            return pa < pb;
        return a.first < b.first;
    });
    const double orig_h = static_cast<double>(h);
    const double orig_w = static_cast<double>(w);
    // required_scale per tiling = min over dims of (tiling*cws / original).
    std::vector<double> required_scale(tilings.size());
    for (size_t k = 0; k < tilings.size(); ++k) {
        double res_h = static_cast<double>(tilings[k].first) * crop_window_size;
        double res_w = static_cast<double>(tilings[k].second) * crop_window_size;
        double sh = (orig_h != 0.0) ? res_h / orig_h : std::numeric_limits<double>::infinity();
        double sw = (orig_w != 0.0) ? res_w / orig_w : std::numeric_limits<double>::infinity();
        required_scale[k] = std::min(sh, sw);
    }
    bool all_lt_1 = std::all_of(required_scale.begin(), required_scale.end(), [](double v) {
        return v < 1.0;
    });
    size_t ix = 0;
    if (all_lt_1) {
        // argmax
        double best = -std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < required_scale.size(); ++k) {
            if (required_scale[k] > best) {
                best = required_scale[k];
                ix = k;
            }
        }
    } else {
        double best = std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < required_scale.size(); ++k) {
            double v = required_scale[k] < 1.0 ? 10e9 : required_scale[k];
            if (v < best) {
                best = v;
                ix = k;
            }
        }
    }
    return tilings[ix];
}

int get_patches_from_tiling(int num_tiles,
                            int pooling_size,
                            int crop_patches,
                            int crop_window_patches,
                            int left_margin,
                            int right_margin) {
    auto ceil_to = [](int v, int p) {
        return (v + p - 1) / p * p;
    };
    if (num_tiles > 1) {
        int left = ceil_to(crop_window_patches + left_margin, pooling_size);
        int mid = ceil_to(crop_window_patches, pooling_size);
        int right = ceil_to(crop_window_patches + right_margin, pooling_size);
        return left + (num_tiles - 2) * mid + right;
    }
    return ceil_to(crop_patches, pooling_size);
}

}  // namespace

VisionEncoderJVLM::VisionEncoderJVLM(const std::filesystem::path& model_dir,
                                     const std::string& device,
                                     const ov::AnyMap properties)
    : VisionEncoder(model_dir, device, properties) {
    load_params(model_dir);
}

VisionEncoderJVLM::VisionEncoderJVLM(const ModelsMap& models_map,
                                     const std::filesystem::path& config_dir_path,
                                     const std::string& device,
                                     const ov::AnyMap device_config)
    : VisionEncoder(models_map, config_dir_path, device, device_config) {
    load_params(config_dir_path);
}

void VisionEncoderJVLM::load_params(const std::filesystem::path& config_dir_path) {
    JinaVLMProcParams p;  // defaults
    std::ifstream stream(config_dir_path / "preprocessor_config.json");
    if (stream.is_open()) {
        nlohmann::json parsed = nlohmann::json::parse(stream);
        using ov::genai::utils::read_json_param;
        read_json_param(parsed, "patch_size", p.patch_size);
        if (parsed.contains("base_input_size")) {
            const auto& bis = parsed.at("base_input_size");
            if (bis.is_array() && !bis.empty()) {
                p.base_input_size = bis.at(0).get<size_t>();
            } else if (bis.is_number()) {
                p.base_input_size = bis.get<size_t>();
            }
        }
        read_json_param(parsed, "max_crops", p.max_crops);
        if (parsed.contains("overlap_margins") && parsed.at("overlap_margins").is_array() &&
            parsed.at("overlap_margins").size() == 2) {
            p.overlap_left = parsed.at("overlap_margins").at(0).get<size_t>();
            p.overlap_right = parsed.at("overlap_margins").at(1).get<size_t>();
        }
        read_json_param(parsed, "pooling_h", p.pooling_h);
        read_json_param(parsed, "pooling_w", p.pooling_w);
        read_json_param(parsed, "token_length_h", p.token_length_h);
        read_json_param(parsed, "token_length_w", p.token_length_w);
        read_json_param(parsed, "tokens_per_image", p.tokens_per_image);
        read_json_param(parsed, "use_column_tokens", p.use_column_tokens);
        read_json_param(parsed, "image_min", p.image_min);
        read_json_param(parsed, "image_max", p.image_max);
        read_json_param(parsed, "normalization_method", p.normalization_method);
        if (parsed.contains("image_mean") && parsed.at("image_mean").is_array() &&
            parsed.at("image_mean").size() == 3) {
            for (int c = 0; c < 3; ++c)
                p.image_mean[c] = parsed.at("image_mean").at(c).get<float>();
        }
        if (parsed.contains("image_std") && parsed.at("image_std").is_array() &&
            parsed.at("image_std").size() == 3) {
            for (int c = 0; c < 3; ++c)
                p.image_std[c] = parsed.at("image_std").at(c).get<float>();
        }
    }
    m_params = p;
}

EncodedImage VisionEncoderJVLM::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    const JinaVLMProcParams& p = m_params;
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    const int H = input_image.ny;
    const int W = input_image.nx;

    const int patch = static_cast<int>(p.patch_size);
    const int base = static_cast<int>(p.base_input_size);
    const int left_margin = static_cast<int>(p.overlap_left);
    const int right_margin = static_cast<int>(p.overlap_right);
    const int pooling_h = static_cast<int>(p.pooling_h);
    const int pooling_w = static_cast<int>(p.pooling_w);
    const int TL_h = static_cast<int>(p.token_length_h);
    const int TL_w = static_cast<int>(p.token_length_w);
    const int total_margin_pixels = patch * (right_margin + left_margin);
    const int crop_patches = base / patch;
    const int crop_window_patches = crop_patches - (right_margin + left_margin);
    const int crop_window_size = crop_window_patches * patch;

    std::pair<int, int> tiling =
        select_tiling(H - total_margin_pixels, W - total_margin_pixels, crop_window_size, static_cast<int>(p.max_crops));
    const int tiling_rows = tiling.first;
    const int tiling_cols = tiling.second;
    const int rh = tiling_rows * crop_window_size + total_margin_pixels;
    const int rw = tiling_cols * crop_window_size + total_margin_pixels;

    // Resize source and normalize.
    std::vector<float> src = bilinear_resize_u8_to_unit(input_image, rh, rw);
    normalize_inplace(src, p);
    // Mask is all-ones (preserve_aspect_ratio == false, no padding).
    std::vector<float> src_mask(static_cast<size_t>(rh) * rw, 1.0f);

    const int image_base_patch = base / patch;  // both dims
    const int crop_size = base;
    const int ppp = patch * patch * 3;
    const int n_patches_per_crop = image_base_patch * image_base_patch;

    const int n_tiled_crops = tiling_rows * tiling_cols;
    const int n_crops = n_tiled_crops + 1;  // + global thumbnail

    // patches[0] is the global thumbnail; patches[1..] are the tiled crops.
    ov::Tensor image_patches(ov::element::f32,
                             ov::Shape{1, static_cast<size_t>(n_crops), static_cast<size_t>(n_patches_per_crop), static_cast<size_t>(ppp)});
    ov::Tensor image_masks(ov::element::i64,
                           ov::Shape{1, static_cast<size_t>(n_crops), static_cast<size_t>(n_patches_per_crop)});
    float* patches_data = image_patches.data<float>();
    int64_t* masks_data = image_masks.data<int64_t>();

    // Global thumbnail crop (crop index 0 in the tensor layout expected by the IR).
    {
        std::vector<float> resized = bilinear_resize_u8_to_unit(input_image, base, base);
        normalize_inplace(resized, p);
        std::vector<float> gpatch;
        int np = 0;
        patchify(resized, base, base, patch, gpatch, np);
        std::copy(gpatch.begin(), gpatch.end(), patches_data);
        // image_processing_jvlm builds the mask array from the tiled crops only and then
        // pads a single -1 row at the END (np.pad(img_mask, [[0,1],[0,0]], -1)), while the
        // thumbnail patches are prepended at the FRONT of the patch array. Reproduce that
        // asymmetry: thumbnail mask row goes to the LAST crop slot, tiled masks to slots
        // 0..n_tiled_crops-1. The vision IR is exported to consume this exact layout.
        int64_t* thumb_mask = masks_data + static_cast<size_t>(n_crops - 1) * n_patches_per_crop;
        for (int i = 0; i < n_patches_per_crop; ++i) {
            thumb_mask[i] = -1;
        }
    }

    // Tiled crops.
    // Also compute patch_ordering exactly as image_processing_jvlm.molmo_overlap_and_resize_cropping:
    // it numbers pooled patches crop-by-crop into a per-crop token grid (padded with -1 to the
    // token_length grid), then transposes to left-to-right order across the whole tiled region.
    const int tl_h = TL_h;
    const int tl_w = TL_w;
    // po_grid holds, for each (tile_row, tile_col, ty, tx), the crop-major running index or -1.
    std::vector<int64_t> po_grid(static_cast<size_t>(tiling_rows) * tiling_cols * tl_h * tl_w, -1);
    int on = 0;
    int crop_out = 1;
    for (int i = 0; i < tiling_rows; ++i) {
        const int y0 = i * crop_window_size;
        int crop_y0 = (i == 0) ? 0 : (left_margin / pooling_h);
        int crop_h = image_base_patch - (right_margin + left_margin);
        if (i == 0)
            crop_h += left_margin;
        if (i == tiling_rows - 1)
            crop_h += right_margin;
        for (int j = 0; j < tiling_cols; ++j) {
            const int x0 = j * crop_window_size;
            int crop_x0 = (j == 0) ? 0 : (left_margin / pooling_w);
            int crop_w = image_base_patch - (right_margin + left_margin);
            if (j == 0)
                crop_w += left_margin;
            if (j == tiling_cols - 1)
                crop_w += right_margin;
            int pooled_w = (crop_w + pooling_w - 1) / pooling_w;
            int pooled_h = (crop_h + pooling_h - 1) / pooling_h;
            // Fill po_grid[i, j, crop_y0:crop_y0+pooled_h, crop_x0:crop_x0+pooled_w] with arange(on, ...)
            int running = on;
            for (int py = 0; py < pooled_h; ++py) {
                for (int px = 0; px < pooled_w; ++px) {
                    const int ty = crop_y0 + py;
                    const int tx = crop_x0 + px;
                    const size_t idx =
                        (((static_cast<size_t>(i) * tiling_cols + j) * tl_h) + ty) * tl_w + tx;
                    po_grid[idx] = running++;
                }
            }
            on += pooled_h * pooled_w;

            // Extract crop [crop_size x crop_size] from src / src_mask.
            std::vector<float> crop_img(static_cast<size_t>(crop_size) * crop_size * 3);
            std::vector<float> crop_mask(static_cast<size_t>(crop_size) * crop_size);
            for (int cy = 0; cy < crop_size; ++cy) {
                for (int cx = 0; cx < crop_size; ++cx) {
                    const int sy = y0 + cy;
                    const int sx = x0 + cx;
                    for (int c = 0; c < 3; ++c) {
                        crop_img[(static_cast<size_t>(cy) * crop_size + cx) * 3 + c] =
                            src[(static_cast<size_t>(sy) * rw + sx) * 3 + c];
                    }
                    crop_mask[static_cast<size_t>(cy) * crop_size + cx] =
                        src_mask[static_cast<size_t>(sy) * rw + sx];
                }
            }
            std::vector<float> cpatch;
            int np = 0;
            patchify(crop_img, crop_size, crop_size, patch, cpatch, np);
            std::copy(cpatch.begin(),
                      cpatch.end(),
                      patches_data + static_cast<size_t>(crop_out) * n_patches_per_crop * ppp);
            std::vector<float> mmean;
            patchify_mask_mean(crop_mask, crop_size, crop_size, patch, mmean);
            // Tiled crop masks occupy slots 0..n_tiled_crops-1 (crop_out is the patch slot,
            // which is offset by +1 for the leading thumbnail patch; the mask array is not).
            int64_t* crop_mask_out = masks_data + static_cast<size_t>(crop_out - 1) * n_patches_per_crop;
            for (int k = 0; k < n_patches_per_crop; ++k) {
                crop_mask_out[k] = static_cast<int64_t>(std::llround(mmean[k]));
            }
            ++crop_out;
        }
    }

    // Reproduce the numpy reshape/transpose that reorders patch_ordering into the flat
    // crop-major layout used by the vision-output rows, so that valid entries, taken in
    // flat order, give left-to-right token order.
    // patch_ordering (flat, crop-major) has n_tiled_crops * tl_h * tl_w entries.
    const size_t tiled_slots = static_cast<size_t>(tiling_rows) * tiling_cols * tl_h * tl_w;
    std::vector<int64_t> patch_ordering(po_grid);  // flat crop-major, values are running indices or -1
    {
        // valid positions in the crop-major flat order
        std::vector<int64_t> valid_values;
        valid_values.reserve(tiled_slots);
        for (size_t s = 0; s < tiled_slots; ++s) {
            if (patch_ordering[s] >= 0)
                valid_values.push_back(patch_ordering[s]);
        }
        // Build the transposed (left-to-right) order: reshape [tr, tc, tl_h, tl_w] -> transpose to
        // [tr, tl_h, tc, tl_w] -> flatten, collect valid values in that order.
        std::vector<int64_t> porh_valid;
        porh_valid.reserve(valid_values.size());
        for (int i = 0; i < tiling_rows; ++i) {
            for (int ty = 0; ty < tl_h; ++ty) {
                for (int j = 0; j < tiling_cols; ++j) {
                    for (int tx = 0; tx < tl_w; ++tx) {
                        const size_t idx =
                            (((static_cast<size_t>(i) * tiling_cols + j) * tl_h) + ty) * tl_w + tx;
                        if (po_grid[idx] >= 0)
                            porh_valid.push_back(po_grid[idx]);
                    }
                }
            }
        }
        // patch_ordering[valid] = porh_valid  (project transposed order into sparse structure)
        size_t vp = 0;
        for (size_t s = 0; s < tiled_slots; ++s) {
            if (patch_ordering[s] >= 0) {
                patch_ordering[s] = porh_valid[vp++];
            }
        }
    }

    // Prepend the global thumbnail: thumbnail patches map to rows 0..(tokens_per_image-1);
    // tiled rows are offset by tokens_per_image.
    const int tokens_per_image = static_cast<int>(p.tokens_per_image);
    // image_input_idx_flat has length n_crops * tokens_per_image, matching the vision output rows.
    // Entry v (>=0) means: this vision row is the v-th <im_patch> slot in joint order.
    // Build joint-slot -> vision-row (inverse) then invert to vision-row -> joint-slot rank.
    // Thumbnail occupies the first tokens_per_image slots (0..tpi-1) in vision-row order.
    // For the tiled region, po_grid running index == tiled vision-row index; joint slot order is the
    // flat crop-major slot order (with -1 removed) after the transpose remap.
    // We produce image_input_idx per vision row = joint slot rank, or -1 for dropped rows.
    ov::Tensor image_input_idx(ov::element::i64, ov::Shape{static_cast<size_t>(n_crops) * tokens_per_image});
    int64_t* iii_data = image_input_idx.data<int64_t>();
    std::fill(iii_data, iii_data + image_input_idx.get_size(), -1);
    // Thumbnail: vision rows 0..tpi-1 -> joint slots 0..tpi-1.
    for (int r = 0; r < tokens_per_image; ++r) {
        iii_data[r] = r;
    }
    // Tiled region: iterate joint slots in crop-major flat order; each valid slot's value is the
    // tiled vision-row index; the joint-slot rank (its order among valid slots) + tokens_per_image
    // is the token position rank. We store, for each tiled vision row, its slot rank.
    {
        int slot_rank = 0;
        for (size_t s = 0; s < tiled_slots; ++s) {
            if (patch_ordering[s] >= 0) {
                const int64_t tiled_row = patch_ordering[s];  // tiled vision-row index
                const size_t vision_row = static_cast<size_t>(tokens_per_image) + tiled_row;
                iii_data[vision_row] = tokens_per_image + slot_rank;
                ++slot_rank;
            }
        }
    }


    // Run the vision embeddings model.
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    encoder.set_tensor("image_patches", image_patches);
    encoder.set_tensor("image_masks", image_masks);
    encoder.infer();
    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    // Token layout for the tiled crops region (image_processing_jvlm output tokens).
    const int h_tok =
        get_patches_from_tiling(tiling_rows, pooling_h, crop_patches, crop_window_patches, left_margin, right_margin);
    const int w_tok =
        get_patches_from_tiling(tiling_cols, pooling_w, crop_patches, crop_window_patches, left_margin, right_margin);

    EncodedImage encoded;
    encoded.resized_source = std::move(image_features);              // [1, n_crops*196, hidden]
    encoded.resized_source_size = {static_cast<size_t>(h_tok / pooling_h), static_cast<size_t>(w_tok / pooling_w)};
    encoded.patches_grid = {tiling_rows, tiling_cols};
    encoded.original_image_size = {static_cast<size_t>(TL_h), static_cast<size_t>(TL_w)};
    encoded.num_image_tokens = static_cast<size_t>(infer_output.get_shape().at(1));
    // Carry image_input_idx (per vision-row -> joint slot rank, or -1) for the scatter merge.
    encoded.images_features_projection = std::move(image_input_idx);
    return encoded;
}

InputsEmbedderJVLM::InputsEmbedderJVLM(const VLMConfig& vlm_config,
                                       const std::filesystem::path& model_dir,
                                       const Tokenizer& tokenizer,
                                       const std::string& device,
                                       const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) {}

InputsEmbedderJVLM::InputsEmbedderJVLM(const VLMConfig& vlm_config,
                                       const ModelsMap& models_map,
                                       const Tokenizer& tokenizer,
                                       const std::filesystem::path& config_dir_path,
                                       const std::string& device,
                                       const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

namespace {

// Build the joint image-token string for a single crop grid (rows x cols), matching
// image_processing_jvlm: <im_start> + rows * (cols * <im_patch> + <im_col>) + <im_end>.
std::string build_crop_tokens(size_t rows,
                              size_t cols,
                              bool use_column_tokens,
                              const std::string& im_start,
                              const std::string& im_patch,
                              const std::string& im_col,
                              const std::string& im_end) {
    std::string s = im_start;
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            s += im_patch;
        }
        if (use_column_tokens) {
            s += im_col;
        }
    }
    s += im_end;
    return s;
}

}  // namespace

NormalizedPrompt InputsEmbedderJVLM::normalize_prompt(const std::string& prompt,
                                                      size_t base_id,
                                                      const std::vector<EncodedImage>& images) const {
    const std::string image_prompt_token = m_vlm_config.jvlm_image_prompt_token;  // "<|image|>"
    const std::string im_start = m_vlm_config.jvlm_image_start_token;              // "<im_start>"
    const std::string im_end = m_vlm_config.jvlm_image_end_token;                  // "<im_end>"
    const std::string im_patch = m_vlm_config.jvlm_image_patch_token;              // "<im_patch>"
    const std::string im_col = m_vlm_config.jvlm_image_column_token;               // "<im_col>"

    auto [unified_prompt, images_sequence] = normalize(prompt, image_prompt_token, image_prompt_token, base_id, images.size());

    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        const EncodedImage& enc = images.at(new_image_id - base_id);
        const size_t tl_h = enc.original_image_size.height;
        const size_t tl_w = enc.original_image_size.width;
        const size_t crop_rows = enc.resized_source_size.height;
        const size_t crop_cols = enc.resized_source_size.width;

        // Global thumbnail first (token_length_h x token_length_w), then the tiled crops.
        std::string expanded =
            build_crop_tokens(tl_h, tl_w, /*use_column_tokens=*/true, im_start, im_patch, im_col, im_end);
        expanded += build_crop_tokens(crop_rows, crop_cols, /*use_column_tokens=*/true, im_start, im_patch, im_col, im_end);

        searched_pos = unified_prompt.find(image_prompt_token, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos,
                        "JinaVLM: image placeholder token not found in prompt during normalization");
        unified_prompt.replace(searched_pos, image_prompt_token.length(), expanded);
        searched_pos += expanded.length();
    }
    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderJVLM::build_jvlm_input_ids(const std::string& unified_prompt,
                                                    ov::genai::VLMPerfMetrics& metrics) {
    // In chat conversation mode the pipeline maintains templated history itself, so
    // fall back to the generic path (which encodes the already-templated prompt).
    if (m_is_chat_conversation) {
        return get_encoded_input_ids(unified_prompt, metrics);
    }

    auto encode_start = std::chrono::steady_clock::now();

    // Reproduce the JinaVLM chat template (chat_template.jinja):
    //   {{ ' ' }}{{ role.capitalize() + ': ' }}{{ content_text + ' ' }} ... {{ 'Assistant:' }}
    // The leading space and the space before "Assistant:" are significant and are
    // dropped by the exported OV tokenizer's apply_chat_template, so build it here.
    std::string templated_prompt;
    bool apply_template = m_apply_chat_template;
    if (apply_template) {
        templated_prompt = " User: " + unified_prompt + " Assistant:";
    } else {
        templated_prompt = unified_prompt;
    }
    auto template_end_time = std::chrono::steady_clock::now();

    // The OV tokenizer does not add the BOS token via add_special_tokens; add it
    // explicitly to match the reference tokenization used by optimum-intel.
    ov::Tensor encoded =
        m_tokenizer.encode(templated_prompt, ov::genai::add_special_tokens(false)).input_ids;
    auto encode_end = std::chrono::steady_clock::now();

    if (apply_template) {
        metrics.raw_metrics.chat_template_durations.emplace_back(
            PerfMetrics::get_microsec(template_end_time - encode_start));
        metrics.raw_metrics.tokenization_durations.emplace_back(
            PerfMetrics::get_microsec(encode_end - template_end_time));
    } else {
        metrics.raw_metrics.tokenization_durations.emplace_back(
            PerfMetrics::get_microsec(encode_end - encode_start));
    }

    // Prepend the BOS token (JinaVLM: <|endoftext|>, config bos_token_id).
    const int64_t bos_id = m_tokenizer.get_bos_token_id();
    ov::Tensor new_chat_tokens;
    if (bos_id >= 0) {
        const auto enc_shape = encoded.get_shape();
        OPENVINO_ASSERT(enc_shape.size() == 2 && enc_shape[0] == 1,
                        "JinaVLM: unexpected encoded input_ids shape");
        const size_t n = enc_shape[1];
        new_chat_tokens = ov::Tensor(encoded.get_element_type(), ov::Shape{1, n + 1});
        int64_t* dst = new_chat_tokens.data<int64_t>();
        const int64_t* src = encoded.data<int64_t>();
        dst[0] = bos_id;
        std::copy_n(src, n, dst + 1);
    } else {
        new_chat_tokens = encoded;
    }

    // Mirror get_encoded_input_ids() history/cache bookkeeping.
    ov::Tensor new_input_ids = update_history(new_chat_tokens);
    m_prev_hist_length = m_cache_state.get_state().size();
    m_cache_state.add_inputs(new_input_ids);
    return new_input_ids;
}

ov::Tensor InputsEmbedderJVLM::get_inputs_embeds(const std::string& unified_prompt,
                                                 const std::vector<ov::genai::EncodedImage>& images,
                                                 ov::genai::VLMPerfMetrics& metrics,
                                                 bool recalculate_merged_embeddings,
                                                 const std::vector<size_t>& images_sequence) {
    ov::Tensor input_ids = build_jvlm_input_ids(unified_prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    if (images.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    // Determine the <im_patch> token id.
    auto start_tok = std::chrono::steady_clock::now();
    ov::Tensor patch_tok = m_tokenizer.encode(m_vlm_config.jvlm_image_patch_token, ov::genai::add_special_tokens(false)).input_ids;
    auto end_tok = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] +=
        ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tok - start_tok));
    const int64_t image_patch_token_id = patch_tok.data<int64_t>()[patch_tok.get_size() - 1];

    const auto text_shape = text_embeds.get_shape();
    const size_t seq_len = text_shape[1];
    const size_t hidden = text_shape[2];
    const int64_t* ids = input_ids.data<const int64_t>();

    ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
    std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
    float* out = inputs_embeds.data<float>();

    // For each image (in prompt order), scatter its vision embedding rows onto the <im_patch>
    // token positions. image_input_idx[row] gives the joint-slot rank of that vision row (or -1
    // if the row is dropped due to overlap-margin padding). Sorting valid rows by their slot rank
    // and placing them at the ordered <im_patch> positions reproduces the reference
    // image_input_idx scatter exactly (validated in .model_enabler/jvlm/proto_multicrop_merge.py).
    size_t token_cursor = 0;
    for (size_t new_image_id : images_sequence) {
        const EncodedImage& enc = images.at(new_image_id);
        const ov::Tensor& feats = enc.resized_source;  // [1, n_rows, hidden]
        const size_t n_rows = feats.get_shape().at(1);
        const float* feat_data = feats.data<const float>();
        const ov::Tensor& iii = enc.images_features_projection;
        OPENVINO_ASSERT(iii && iii.get_size() == n_rows,
                        "JinaVLM: image_input_idx size does not match vision embedding rows");
        const int64_t* iii_data = iii.data<const int64_t>();

        // Collect valid (slot_rank, vision_row) pairs and sort by slot_rank.
        std::vector<std::pair<int64_t, size_t>> valid;
        valid.reserve(n_rows);
        for (size_t r = 0; r < n_rows; ++r) {
            if (iii_data[r] >= 0) {
                valid.emplace_back(iii_data[r], r);
            }
        }
        std::sort(valid.begin(), valid.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        // Place each valid vision row at the next <im_patch> position.
        size_t placed = 0;
        for (; token_cursor < seq_len && placed < valid.size(); ++token_cursor) {
            if (ids[token_cursor] == image_patch_token_id) {
                const size_t vision_row = valid[placed].second;
                std::copy_n(feat_data + vision_row * hidden, hidden, out + token_cursor * hidden);
                ++placed;
            }
        }
        OPENVINO_ASSERT(placed == valid.size(),
                        "JinaVLM: number of <im_patch> tokens does not match valid vision embedding rows");
    }
    return inputs_embeds;
}

}  // namespace ov::genai
