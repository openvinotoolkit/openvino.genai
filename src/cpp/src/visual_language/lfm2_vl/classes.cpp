// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/lfm2_vl/classes.hpp"

#include "visual_language/clip.hpp"

#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <set>

namespace ov::genai {

namespace {

// ---------------------------------------------------------------------------
// Small float RGB image container (HWC, RGBRGB... layout, value range 0..255).
// ---------------------------------------------------------------------------
struct FloatImage {
    int h = 0;
    int w = 0;
    std::vector<float> buf;  // h * w * 3
};

FloatImage to_float_image(const clip_image_u8& src) {
    FloatImage img;
    img.h = src.ny;
    img.w = src.nx;
    img.buf.resize(static_cast<size_t>(img.h) * img.w * 3);
    for (size_t i = 0; i < img.buf.size(); ++i) {
        img.buf[i] = static_cast<float>(src.buf[i]);
    }
    return img;
}

int round_by_factor(double number, int factor) {
    return static_cast<int>(std::llround(number / factor)) * factor;
}

// ---------------------------------------------------------------------------
// Separable resampling weights matching torchvision / PyTorch antialias resize.
// For each output index: {first_input_index, weights...}.
// ---------------------------------------------------------------------------
double triangle_filter(double x) {
    x = std::abs(x);
    return x < 1.0 ? 1.0 - x : 0.0;
}

double cubic_filter(double x) {
    // Keys cubic convolution, a = -0.5 (Pillow / torchvision bicubic).
    constexpr double a = -0.5;
    x = std::abs(x);
    if (x < 1.0) {
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    }
    if (x < 2.0) {
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    }
    return 0.0;
}

struct WeightRow {
    int first = 0;
    std::vector<double> w;
};

// filter_support: 1.0 for bilinear (triangle), 2.0 for bicubic.
std::vector<WeightRow> compute_weights(int in_size, int out_size, bool bicubic, bool antialias) {
    const double base_support = bicubic ? 2.0 : 1.0;
    const double scale = static_cast<double>(in_size) / static_cast<double>(out_size);
    const double filterscale = (antialias && scale > 1.0) ? scale : 1.0;
    const double support = base_support * filterscale;
    const double inv_filterscale = 1.0 / filterscale;

    std::vector<WeightRow> rows(out_size);
    for (int o = 0; o < out_size; ++o) {
        const double center = (o + 0.5) * scale;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) {
            xmin = 0;
        }
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size) {
            xmax = in_size;
        }
        WeightRow row;
        row.first = xmin;
        double total = 0.0;
        for (int x = xmin; x < xmax; ++x) {
            const double arg = (x + 0.5 - center) * inv_filterscale;
            const double weight = bicubic ? cubic_filter(arg) : triangle_filter(arg);
            row.w.push_back(weight);
            total += weight;
        }
        if (total != 0.0) {
            for (double& value : row.w) {
                value /= total;
            }
        }
        rows[o] = std::move(row);
    }
    return rows;
}

// Clamp an integer channel value to the uint8 range.
inline int clamp_u8(int value) {
    return value < 0 ? 0 : (value > 255 ? 255 : value);
}

// int16 fixed-point resampling weights matching torchvision's uint8 antialias
// path. torchvision converts the normalized double weights to int16 with a
// per-axis precision and runs the whole convolution in integer arithmetic.
struct IntWeightRow {
    int first = 0;
    std::vector<int> w;
};

// Convert double weight rows to int16 fixed-point, returning {rows, precision}.
// Reproduces aten `_compute_index_ranges_int16_weights`: the precision is the
// largest value keeping every scaled weight inside int16, and weights are cast
// with round-half-away-from-zero via a truncating cast.
std::pair<std::vector<IntWeightRow>, int> quantize_weights(const std::vector<WeightRow>& rows) {
    double wt_max = 0.0;
    for (const WeightRow& row : rows) {
        for (double value : row.w) {
            wt_max = std::max(wt_max, value);
        }
    }
    int precision = 0;
    while (precision < 22) {
        const int next_value = static_cast<int>(0.5 + wt_max * (1 << (precision + 1)));
        if (next_value >= (1 << 15)) {
            break;
        }
        ++precision;
    }
    std::vector<IntWeightRow> out(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        out[i].first = rows[i].first;
        out[i].w.reserve(rows[i].w.size());
        for (double weight : rows[i].w) {
            const double scaled = weight * (1 << precision);
            const int quantized =
                scaled < 0.0 ? static_cast<int>(-0.5 + scaled) : static_cast<int>(0.5 + scaled);
            out[i].w.push_back(quantized);
        }
    }
    return {std::move(out), precision};
}

// Separable antialias resize reproducing torchvision's uint8 BICUBIC path
// bit-exactly (`Lfm2VlImageProcessor` uses `resample=3`, antialias=True with the
// TorchvisionBackend). The vision tower must receive the exact same pixel_values
// as optimum-intel / HF; a float pipeline diverges by ~1/255 which shifts greedy
// decoding on this dtype-sensitive model. Weights are quantized to int16 and each
// separable pass is clamped to uint8 before the next, matching aten's kernel.
FloatImage resize_image(const FloatImage& src, int out_h, int out_w) {
    if (src.h == out_h && src.w == out_w) {
        return src;
    }
    const auto [wx, precx] = quantize_weights(compute_weights(src.w, out_w, /*bicubic=*/true, /*antialias=*/true));
    const auto [wy, precy] = quantize_weights(compute_weights(src.h, out_h, /*bicubic=*/true, /*antialias=*/true));
    const int halfx = 1 << (precx - 1);
    const int halfy = 1 << (precy - 1);

    // Horizontal pass: src.h x out_w, integer arithmetic clamped to uint8.
    std::vector<int> tmp(static_cast<size_t>(src.h) * out_w * 3);
    for (int y = 0; y < src.h; ++y) {
        const float* row = src.buf.data() + static_cast<size_t>(y) * src.w * 3;
        for (int ox = 0; ox < out_w; ++ox) {
            const IntWeightRow& wr = wx[ox];
            int acc[3] = {halfx, halfx, halfx};
            for (size_t k = 0; k < wr.w.size(); ++k) {
                const float* px = row + static_cast<size_t>(wr.first + static_cast<int>(k)) * 3;
                acc[0] += wr.w[k] * static_cast<int>(px[0]);
                acc[1] += wr.w[k] * static_cast<int>(px[1]);
                acc[2] += wr.w[k] * static_cast<int>(px[2]);
            }
            int* dst = tmp.data() + (static_cast<size_t>(y) * out_w + ox) * 3;
            dst[0] = clamp_u8(acc[0] >> precx);
            dst[1] = clamp_u8(acc[1] >> precx);
            dst[2] = clamp_u8(acc[2] >> precx);
        }
    }

    // Vertical pass: out_h x out_w.
    FloatImage out;
    out.h = out_h;
    out.w = out_w;
    out.buf.resize(static_cast<size_t>(out_h) * out_w * 3);
    for (int oy = 0; oy < out_h; ++oy) {
        const IntWeightRow& wr = wy[oy];
        for (int x = 0; x < out_w; ++x) {
            int acc[3] = {halfy, halfy, halfy};
            for (size_t k = 0; k < wr.w.size(); ++k) {
                const int* px = tmp.data() + (static_cast<size_t>(wr.first + static_cast<int>(k)) * out_w + x) * 3;
                acc[0] += wr.w[k] * px[0];
                acc[1] += wr.w[k] * px[1];
                acc[2] += wr.w[k] * px[2];
            }
            float* dst = out.buf.data() + (static_cast<size_t>(oy) * out_w + x) * 3;
            dst[0] = static_cast<float>(clamp_u8(acc[0] >> precy));
            dst[1] = static_cast<float>(clamp_u8(acc[1] >> precy));
            dst[2] = static_cast<float>(clamp_u8(acc[2] >> precy));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Naflex geometry helpers (ports of Lfm2VlImageProcessor).
// ---------------------------------------------------------------------------
struct Geometry {
    const VLMConfig& cfg;
    int patch() const { return static_cast<int>(cfg.lfm2_encoder_patch_size); }
    int downsample() const { return static_cast<int>(cfg.lfm2_downsample_factor); }
    int total_factor() const { return patch() * downsample(); }
};

// Returns {new_width, new_height} (python order).
std::pair<int, int> smart_resize(const VLMConfig& cfg, int height, int width) {
    Geometry g{cfg};
    const int tf = g.total_factor();
    const double min_pixels = static_cast<double>(cfg.lfm2_min_image_tokens) * g.patch() * g.patch() *
                              g.downsample() * g.downsample();
    const double max_pixels = static_cast<double>(cfg.lfm2_max_image_tokens) * g.patch() * g.patch() *
                              g.downsample() * g.downsample();

    int h_bar = std::max(tf, round_by_factor(height, tf));
    int w_bar = std::max(tf, round_by_factor(width, tf));

    if (static_cast<double>(h_bar) * w_bar > max_pixels) {
        const double beta = std::sqrt(static_cast<double>(height) * width / max_pixels);
        h_bar = std::max(tf, static_cast<int>(std::floor(height / beta / tf)) * tf);
        w_bar = std::max(tf, static_cast<int>(std::floor(width / beta / tf)) * tf);
    } else if (static_cast<double>(h_bar) * w_bar < min_pixels) {
        const double beta = std::sqrt(min_pixels / (static_cast<double>(height) * width));
        h_bar = static_cast<int>(std::ceil(height * beta / tf)) * tf;
        w_bar = static_cast<int>(std::ceil(width * beta / tf)) * tf;
    }
    return {w_bar, h_bar};
}

bool is_image_too_large(const VLMConfig& cfg, int height, int width) {
    Geometry g{cfg};
    const int tf = g.total_factor();
    const int h_bar = std::max(g.patch(), round_by_factor(height, tf));
    const int w_bar = std::max(g.patch(), round_by_factor(width, tf));
    const double limit = static_cast<double>(cfg.lfm2_max_image_tokens) * g.patch() * g.patch() *
                         g.downsample() * g.downsample() * cfg.lfm2_max_pixels_tolerance;
    return static_cast<double>(h_bar) * w_bar > limit;
}

std::vector<std::pair<int, int>> target_ratios(int min_tiles, int max_tiles) {
    std::set<std::pair<int, int>> ratios;
    for (int n = min_tiles; n <= max_tiles; ++n) {
        for (int w = 1; w <= n; ++w) {
            for (int h = 1; h <= n; ++h) {
                if (w * h >= min_tiles && w * h <= max_tiles) {
                    ratios.insert({w, h});
                }
            }
        }
    }
    std::vector<std::pair<int, int>> result(ratios.begin(), ratios.end());
    std::stable_sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.first * a.second < b.first * b.second;
    });
    return result;
}

// Returns {grid_width, grid_height}.
std::pair<int, int> find_closest_aspect_ratio(double aspect_ratio,
                                               const std::vector<std::pair<int, int>>& ratios,
                                               int width,
                                               int height,
                                               int image_size) {
    double best_diff = std::numeric_limits<double>::infinity();
    std::pair<int, int> best{1, 1};
    const double area = static_cast<double>(width) * height;
    for (const auto& ratio : ratios) {
        const double target_aspect = static_cast<double>(ratio.first) / ratio.second;
        const double diff = std::abs(aspect_ratio - target_aspect);
        if (diff < best_diff) {
            best_diff = diff;
            best = ratio;
        } else if (diff == best_diff) {
            const double target_area =
                static_cast<double>(image_size) * image_size * ratio.first * ratio.second;
            if (area > 0.5 * target_area) {
                best = ratio;
            }
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Normalize a (already resized) float RGB tile into packed patch pixel values
// [1, num_patches, patch*patch*3] matching convert_image_to_patches ordering.
// ---------------------------------------------------------------------------
ov::Tensor build_pixel_values(const FloatImage& img, const VLMConfig& cfg) {
    const int patch = static_cast<int>(cfg.lfm2_encoder_patch_size);
    const int nph = img.h / patch;
    const int npw = img.w / patch;
    const size_t num_patches = static_cast<size_t>(nph) * npw;
    const size_t patch_dim = static_cast<size_t>(patch) * patch * 3;

    ov::Tensor pixel_values(ov::element::f32, ov::Shape{1, num_patches, patch_dim});
    float* dst = pixel_values.data<float>();

    // Normalization: rescale (1/255) then (x - mean) / std, mean = std = 0.5.
    // -> (px / 255 - 0.5) / 0.5 = px / 127.5 - 1.
    for (int ph = 0; ph < nph; ++ph) {
        for (int pw = 0; pw < npw; ++pw) {
            const size_t patch_idx = static_cast<size_t>(ph) * npw + pw;
            float* patch_dst = dst + patch_idx * patch_dim;
            size_t o = 0;
            for (int i = 0; i < patch; ++i) {
                const int y = ph * patch + i;
                for (int j = 0; j < patch; ++j) {
                    const int x = pw * patch + j;
                    const float* px = img.buf.data() + (static_cast<size_t>(y) * img.w + x) * 3;
                    patch_dst[o++] = px[0] / 127.5f - 1.0f;
                    patch_dst[o++] = px[1] / 127.5f - 1.0f;
                    patch_dst[o++] = px[2] / 127.5f - 1.0f;
                }
            }
        }
    }
    return pixel_values;
}

// Positional resample kernel [num_out_patches, num_positions] equal to the
// optimum-intel `_build_pos_resample_kernel`: bilinear (align_corners=False,
// antialias=True) interpolation of the identity position grid to (nph, npw).
ov::Tensor build_pos_resample_kernel(int nph, int npw, const VLMConfig& cfg) {
    const int num_positions = static_cast<int>(cfg.lfm2_vision_num_patches);
    const int side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(num_positions))));

    const std::vector<WeightRow> wh = compute_weights(side, nph, /*bicubic=*/false, /*antialias=*/true);
    const std::vector<WeightRow> ww = compute_weights(side, npw, /*bicubic=*/false, /*antialias=*/true);

    const size_t out = static_cast<size_t>(nph) * npw;
    ov::Tensor kernel(ov::element::f32, ov::Shape{out, static_cast<size_t>(num_positions)});
    float* data = kernel.data<float>();
    std::fill(data, data + out * num_positions, 0.0f);

    for (int oh = 0; oh < nph; ++oh) {
        const WeightRow& rh = wh[oh];
        for (int ow = 0; ow < npw; ++ow) {
            const WeightRow& rw = ww[ow];
            float* row = data + (static_cast<size_t>(oh) * npw + ow) * num_positions;
            for (size_t a = 0; a < rh.w.size(); ++a) {
                const int in_h = rh.first + static_cast<int>(a);
                for (size_t b = 0; b < rw.w.size(); ++b) {
                    const int in_w = rw.first + static_cast<int>(b);
                    row[static_cast<size_t>(in_h) * side + in_w] =
                        static_cast<float>(rh.w[a] * rw.w[b]);
                }
            }
        }
    }
    return kernel;
}

} // namespace

EncodedImage VisionEncoderLFM2VL::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    // config_map carries the loaded VLMConfig fields via "vlm_config".
    OPENVINO_ASSERT(config_map.count("vlm_config"), "LFM2-VL vision encoder requires vlm_config");
    const VLMConfig& cfg = config_map.at("vlm_config").as<VLMConfig>();

    clip_image_u8 u8 = tensor_to_clip_image_u8(image);
    FloatImage src = to_float_image(u8);
    const int patch = static_cast<int>(cfg.lfm2_encoder_patch_size);
    const int tile_size = static_cast<int>(cfg.lfm2_tile_size);

    // Determine tiling.
    const bool do_split = cfg.lfm2_do_image_splitting && !(cfg.lfm2_min_tiles == 1 && cfg.lfm2_max_tiles == 1);
    const bool is_large = is_image_too_large(cfg, src.h, src.w);
    const auto [new_w, new_h] = smart_resize(cfg, src.h, src.w);

    std::vector<FloatImage> tiles;
    int rows = 1;
    int cols = 1;

    if (is_large && do_split) {
        const std::vector<std::pair<int, int>> ratios =
            target_ratios(static_cast<int>(cfg.lfm2_min_tiles), static_cast<int>(cfg.lfm2_max_tiles));
        const double aspect = static_cast<double>(src.w) / src.h;
        const auto [grid_w, grid_h] = find_closest_aspect_ratio(aspect, ratios, src.w, src.h, tile_size);
        rows = grid_h;
        cols = grid_w;

        const int target_h = tile_size * grid_h;
        const int target_w = tile_size * grid_w;
        FloatImage resized = resize_image(src, target_h, target_w);

        // Split into grid_h x grid_w tiles (row-major).
        for (int r = 0; r < grid_h; ++r) {
            for (int c = 0; c < grid_w; ++c) {
                FloatImage tile;
                tile.h = tile_size;
                tile.w = tile_size;
                tile.buf.resize(static_cast<size_t>(tile_size) * tile_size * 3);
                for (int y = 0; y < tile_size; ++y) {
                    const int sy = r * tile_size + y;
                    const float* src_row = resized.buf.data() + (static_cast<size_t>(sy) * target_w + c * tile_size) * 3;
                    float* dst_row = tile.buf.data() + static_cast<size_t>(y) * tile_size * 3;
                    std::copy_n(src_row, static_cast<size_t>(tile_size) * 3, dst_row);
                }
                tiles.push_back(std::move(tile));
            }
        }
        if (cfg.lfm2_use_thumbnail && grid_h * grid_w != 1) {
            tiles.push_back(resize_image(src, new_h, new_w));
        }
    } else {
        tiles.push_back(resize_image(src, new_h, new_w));
        rows = 1;
        cols = 1;
    }

    // Run the vision IR per tile and concatenate the projected token features.
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();

    std::vector<ov::Tensor> tile_features;
    size_t total_tokens = 0;
    size_t hidden = 0;
    for (const FloatImage& tile : tiles) {
        const int nph = tile.h / patch;
        const int npw = tile.w / patch;

        ov::Tensor pixel_values = build_pixel_values(tile, cfg);
        ov::Tensor pos_kernel = build_pos_resample_kernel(nph, npw, cfg);
        ov::Tensor spatial_shapes(ov::element::i64, ov::Shape{1, 2});
        spatial_shapes.data<int64_t>()[0] = nph;
        spatial_shapes.data<int64_t>()[1] = npw;

        encoder.set_tensor("pixel_values", pixel_values);
        encoder.set_tensor("pos_resample_kernel", pos_kernel);
        encoder.set_tensor("spatial_shapes", spatial_shapes);
        encoder.infer();

        const ov::Tensor& out = encoder.get_output_tensor();
        const ov::Shape out_shape = out.get_shape();  // [num_tokens, hidden]
        hidden = out_shape.back();
        const size_t num_tokens = out.get_size() / hidden;

        ov::Tensor feat(out.get_element_type(), ov::Shape{1, num_tokens, hidden});
        std::memcpy(feat.data(), out.data(), out.get_byte_size());
        total_tokens += num_tokens;
        tile_features.push_back(std::move(feat));
    }

    // Concatenate along the token dimension -> [1, total_tokens, hidden].
    ov::Tensor image_features(ov::element::f32, ov::Shape{1, total_tokens, hidden});
    float* dst = image_features.data<float>();
    for (const ov::Tensor& feat : tile_features) {
        std::memcpy(dst, feat.data(), feat.get_byte_size());
        dst += feat.get_size();
    }

    EncodedImage encoded;
    encoded.resized_source = std::move(image_features);
    encoded.patches_grid = {rows, cols};
    encoded.num_image_tokens = total_tokens;
    return encoded;
}

InputsEmbedderLFM2VL::InputsEmbedderLFM2VL(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) { }

InputsEmbedderLFM2VL::InputsEmbedderLFM2VL(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) { }

namespace {

// Number of image tokens per single tile (fixed by tile geometry).
size_t tokens_per_tile(const VLMConfig& cfg) {
    const int tile_patches = static_cast<int>(cfg.lfm2_tile_size) / static_cast<int>(cfg.lfm2_encoder_patch_size);
    const int downsampled = static_cast<int>(std::ceil(static_cast<double>(tile_patches) / cfg.lfm2_downsample_factor));
    return static_cast<size_t>(downsampled) * downsampled;
}

// Per-run image-token counts: one entry per contiguous run of image tokens the
// processor emits (each tile, then the thumbnail, or a single run).
std::vector<size_t> run_sizes_for(const VLMConfig& cfg, const EncodedImage& image) {
    const size_t rows = static_cast<size_t>(image.patches_grid.first);
    const size_t cols = static_cast<size_t>(image.patches_grid.second);
    std::vector<size_t> runs;
    if (rows <= 1 && cols <= 1) {
        runs.push_back(image.num_image_tokens);
        return runs;
    }
    const size_t tpt = tokens_per_tile(cfg);
    for (size_t t = 0; t < rows * cols; ++t) {
        runs.push_back(tpt);
    }
    const size_t consumed = rows * cols * tpt;
    if (image.num_image_tokens > consumed) {
        runs.push_back(image.num_image_tokens - consumed);  // thumbnail
    }
    return runs;
}

} // namespace

std::string InputsEmbedderLFM2VL::build_image_placeholders(const EncodedImage& image, std::vector<size_t>& run_sizes) const {
    const std::string& image_token = m_vlm_config.im_start;  // "<image>"
    const bool special = m_vlm_config.lfm2_use_image_special_tokens;
    const size_t rows = static_cast<size_t>(image.patches_grid.first);
    const size_t cols = static_cast<size_t>(image.patches_grid.second);
    const std::vector<size_t> runs = run_sizes_for(m_vlm_config, image);

    std::string result;
    if (special) {
        result += m_vlm_config.lfm2_image_start_token;
    }

    const bool multi_tile = rows > 1 || cols > 1;
    if (multi_tile) {
        size_t run_idx = 0;
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                if (special) {
                    result += "<|img_row_" + std::to_string(r + 1) + "_col_" + std::to_string(c + 1) + "|>";
                }
                const size_t n = runs[run_idx++];
                for (size_t i = 0; i < n; ++i) {
                    result += image_token;
                }
                run_sizes.push_back(n);
            }
        }
        if (m_vlm_config.lfm2_use_thumbnail && run_idx < runs.size()) {
            if (special) {
                result += m_vlm_config.lfm2_image_thumbnail_token;
            }
            const size_t n = runs[run_idx++];
            for (size_t i = 0; i < n; ++i) {
                result += image_token;
            }
            run_sizes.push_back(n);
        }
    } else {
        const size_t n = runs[0];
        for (size_t i = 0; i < n; ++i) {
            result += image_token;
        }
        run_sizes.push_back(n);
    }

    if (special) {
        result += m_vlm_config.lfm2_image_end_token;
    }
    return result;
}

std::vector<ov::genai::EncodedImage> InputsEmbedderLFM2VL::encode_images(const std::vector<ov::Tensor>& images) {
    // The naflex vision encoder needs the model geometry (patch/tile/downsample
    // and tiling parameters) that live in VLMConfig, so pass it through.
    ov::AnyMap vision_config = {{"vlm_config", m_vlm_config}};
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    std::vector<EncodedImage> embeds;
    embeds.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image, vision_config));
    }
    return embeds;
}

NormalizedPrompt InputsEmbedderLFM2VL::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    const std::string& image_token = m_vlm_config.im_start;
    auto [unified_prompt, images_sequence] = normalize(prompt, image_token, image_token, base_id, images.size());

    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        std::vector<size_t> unused;
        std::string expanded = build_image_placeholders(images.at(new_image_id - base_id), unused);
        searched_pos = unified_prompt.find(image_token, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos);
        unified_prompt.replace(searched_pos, image_token.length(), expanded);
        searched_pos += expanded.length();
    }
    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderLFM2VL::get_inputs_embeds(const std::string& unified_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
    // Split each image's concatenated features into per-run chunks so that the
    // llava-style merge scatters each contiguous run of <image> tokens
    // (separated by boundary/row-col/thumbnail special tokens) correctly.
    std::vector<ov::Tensor> image_embeds;
    for (size_t new_image_id : images_sequence) {
        const EncodedImage& enc = images.at(new_image_id);
        const std::vector<size_t> runs = run_sizes_for(m_vlm_config, enc);
        const ov::Shape shape = enc.resized_source.get_shape();  // [1, total, hidden]
        const size_t hidden = shape.back();
        const float* src = enc.resized_source.data<const float>();
        size_t offset = 0;
        for (size_t run : runs) {
            ov::Tensor chunk(ov::element::f32, ov::Shape{1, run, hidden});
            std::memcpy(chunk.data(), src + offset * hidden, run * hidden * sizeof(float));
            image_embeds.push_back(std::move(chunk));
            offset += run;
        }
    }

    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = get_text_embedding(req, input_ids, metrics);

    if (images.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    return utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, m_vlm_config.image_token_id);
}

} // namespace ov::genai
