// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/lfm2_vl/classes.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <set>

#include "visual_language/clip.hpp"
#include "visual_language/qwen2vl/classes.hpp"

namespace ov::genai {

namespace {

// Round `number` to the closest multiple of `factor`, matching Python's round-half-to-even.
int round_by_factor(double number, int factor) {
    return static_cast<int>(std::nearbyint(number / factor)) * factor;
}

// Siglip2 NaFlex positional-embedding interpolation.
// Replicates torch F.interpolate(mode="bilinear", align_corners=False, antialias=True)
// one-dimensional weights for resizing `in_size` samples to `out_size` samples.
std::vector<std::vector<float>> antialias_linear_weights(int in_size, int out_size) {
    const double scale = static_cast<double>(in_size) / out_size;
    const bool downscale = scale >= 1.0;
    const double support = downscale ? scale : 1.0;
    const double invscale = downscale ? 1.0 / scale : 1.0;
    std::vector<std::vector<float>> weights(out_size, std::vector<float>(in_size, 0.0f));
    for (int oi = 0; oi < out_size; ++oi) {
        const double center = scale * (oi + 0.5);
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) {
            xmin = 0;
        }
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size) {
            xmax = in_size;
        }
        double total = 0.0;
        for (int j = xmin; j < xmax; ++j) {
            const double arg = std::abs((j - center + 0.5) * invscale);
            const double w = arg < 1.0 ? 1.0 - arg : 0.0;
            weights[oi][j] = static_cast<float>(w);
            total += w;
        }
        if (total != 0.0) {
            for (int j = xmin; j < xmax; ++j) {
                weights[oi][j] = static_cast<float>(weights[oi][j] / total);
            }
        }
    }
    return weights;
}

// Keys cubic convolution kernel with a = -0.5 (PIL/Pillow bicubic, which the
// torchvision uint8 antialias backend used by Lfm2VlImageProcessor emulates).
double cubic_kernel(double x) {
    const double a = -0.5;
    x = std::abs(x);
    if (x < 1.0) {
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    }
    if (x < 2.0) {
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    }
    return 0.0;
}

// One-dimensional antialiased cubic resampling weights, replicating torchvision's
// bicubic antialias resize (interpolation support = 2). Each output row holds the
// first contributing input index and its normalized weights.
std::vector<std::pair<int, std::vector<double>>> antialias_cubic_weights(int in_size, int out_size) {
    const double scale = static_cast<double>(in_size) / out_size;
    const bool downscale = scale >= 1.0;
    const double support = downscale ? 2.0 * scale : 2.0;
    const double invscale = downscale ? 1.0 / scale : 1.0;
    std::vector<std::pair<int, std::vector<double>>> rows(out_size);
    for (int oi = 0; oi < out_size; ++oi) {
        const double center = scale * (oi + 0.5);
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) {
            xmin = 0;
        }
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size) {
            xmax = in_size;
        }
        std::vector<double> weights(static_cast<size_t>(xmax - xmin), 0.0);
        double total = 0.0;
        for (int j = xmin; j < xmax; ++j) {
            const double w = cubic_kernel((j - center + 0.5) * invscale);
            weights[j - xmin] = w;
            total += w;
        }
        if (total != 0.0) {
            for (double& w : weights) {
                w /= total;
            }
        }
        rows[oi] = {xmin, std::move(weights)};
    }
    return rows;
}

// Per-axis quantized cubic weights plus the fixed-point precision used for them.
struct QuantizedAxisWeights {
    std::vector<std::pair<int, std::vector<int32_t>>> rows;
    int precision = 0;
};

// Quantize normalized float weights to integers exactly as PyTorch's uint8
// antialias backend does: the precision is chosen per axis as the largest value
// (<= 22) for which the largest-magnitude weight still fits in a signed 16-bit
// integer, and each weight is rounded toward zero after adding 0.5 (matching
// (int)(0.5 + w * (1 << precision)) for positive and its negation for negative
// weights). This reproduces torchvision's resize bit-for-bit.
QuantizedAxisWeights quantize_axis_weights(const std::vector<std::pair<int, std::vector<double>>>& weights) {
    double max_weight = 0.0;
    for (const auto& row : weights) {
        for (double w : row.second) {
            max_weight = std::max(max_weight, std::abs(w));
        }
    }
    int precision = 0;
    while (precision < 22) {
        const int next_value = static_cast<int>(0.5 + max_weight * static_cast<double>(int64_t(1) << (precision + 1)));
        if (next_value >= (1 << 15)) {
            break;
        }
        ++precision;
    }

    QuantizedAxisWeights out;
    out.precision = precision;
    out.rows.resize(weights.size());
    const double scale = static_cast<double>(int64_t(1) << precision);
    for (size_t i = 0; i < weights.size(); ++i) {
        out.rows[i].first = weights[i].first;
        out.rows[i].second.resize(weights[i].second.size());
        for (size_t k = 0; k < weights[i].second.size(); ++k) {
            const double w = weights[i].second[k];
            out.rows[i].second[k] = w >= 0.0 ? static_cast<int32_t>(0.5 + w * scale)
                                             : -static_cast<int32_t>(0.5 - w * scale);
        }
    }
    return out;
}

// Antialiased bicubic resize of an RGB uint8 image, matching the torchvision
// uint8 backend used by Lfm2VlImageProcessor (resample=BICUBIC, antialias=True).
// That backend is derived from Pillow's separable fixed-point resampler but
// selects the fixed-point precision per axis from the weight magnitudes and
// stores the weights as signed 16-bit integers. The weights are accumulated in
// integer arithmetic with a rounded right shift and uint8 clamp after each pass.
// Reproducing this scheme (rather than a fixed PRECISION_BITS with symmetric
// rounding) matches torchvision's output bit-for-bit, which the SigLIP2 tower
// requires so greedy VLM decoding stays aligned with optimum-intel.
void bicubic_antialias_resize(const clip_image_u8& src, clip_image_u8& dst, int out_width, int out_height) {
    const int in_w = src.nx;
    const int in_h = src.ny;

    const QuantizedAxisWeights row_weights = quantize_axis_weights(antialias_cubic_weights(in_h, out_height));
    const QuantizedAxisWeights col_weights = quantize_axis_weights(antialias_cubic_weights(in_w, out_width));

    auto clip8 = [](int64_t acc, int precision) -> uint8_t {
        const int64_t v = (acc + (int64_t(1) << (precision - 1))) >> precision;
        return static_cast<uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
    };

    // Horizontal pass: (in_h, in_w) -> (in_h, out_width), clamped to uint8.
    const int col_precision = col_weights.precision;
    std::vector<uint8_t> horizontal(static_cast<size_t>(in_h) * out_width * 3);
    for (int y = 0; y < in_h; ++y) {
        for (int ox = 0; ox < out_width; ++ox) {
            const int xmin = col_weights.rows[ox].first;
            const std::vector<int32_t>& ws = col_weights.rows[ox].second;
            int64_t acc[3] = {0, 0, 0};
            for (size_t k = 0; k < ws.size(); ++k) {
                const size_t src_idx = (static_cast<size_t>(y) * in_w + (xmin + k)) * 3;
                acc[0] += static_cast<int64_t>(ws[k]) * src.buf[src_idx + 0];
                acc[1] += static_cast<int64_t>(ws[k]) * src.buf[src_idx + 1];
                acc[2] += static_cast<int64_t>(ws[k]) * src.buf[src_idx + 2];
            }
            const size_t dst_idx = (static_cast<size_t>(y) * out_width + ox) * 3;
            horizontal[dst_idx + 0] = clip8(acc[0], col_precision);
            horizontal[dst_idx + 1] = clip8(acc[1], col_precision);
            horizontal[dst_idx + 2] = clip8(acc[2], col_precision);
        }
    }

    // Vertical pass: (in_h, out_width) -> (out_height, out_width).
    const int row_precision = row_weights.precision;
    dst.nx = out_width;
    dst.ny = out_height;
    dst.buf.resize(static_cast<size_t>(out_width) * out_height * 3);
    for (int oy = 0; oy < out_height; ++oy) {
        const int ymin = row_weights.rows[oy].first;
        const std::vector<int32_t>& ws = row_weights.rows[oy].second;
        for (int ox = 0; ox < out_width; ++ox) {
            int64_t acc[3] = {0, 0, 0};
            for (size_t k = 0; k < ws.size(); ++k) {
                const size_t src_idx = (static_cast<size_t>(ymin + k) * out_width + ox) * 3;
                acc[0] += static_cast<int64_t>(ws[k]) * horizontal[src_idx + 0];
                acc[1] += static_cast<int64_t>(ws[k]) * horizontal[src_idx + 1];
                acc[2] += static_cast<int64_t>(ws[k]) * horizontal[src_idx + 2];
            }
            const size_t dst_idx = (static_cast<size_t>(oy) * out_width + ox) * 3;
            dst.buf[dst_idx + 0] = clip8(acc[0], row_precision);
            dst.buf[dst_idx + 1] = clip8(acc[1], row_precision);
            dst.buf[dst_idx + 2] = clip8(acc[2], row_precision);
        }
    }
}

struct Lfm2VlParams {
    int patch_size;
    int downsample_factor;
    int tile_size;
    int min_tiles;
    int max_tiles;
    int min_image_tokens;
    int max_image_tokens;
    float max_pixels_tolerance;
    bool use_thumbnail;
    bool do_image_splitting;
    std::array<float, 3> image_mean;
    std::array<float, 3> image_std;
    int num_pos;   // base positional-embedding count of the vision tower (e.g. 256)
    int pos_size;  // sqrt(num_pos) (e.g. 16)
};

Lfm2VlParams make_params(const ProcessorConfig& cfg, int num_pos) {
    Lfm2VlParams p;
    p.patch_size = static_cast<int>(cfg.encoder_patch_size);
    p.downsample_factor = static_cast<int>(cfg.downsample_factor);
    p.tile_size = static_cast<int>(cfg.tile_size);
    p.min_tiles = static_cast<int>(cfg.min_tiles);
    p.max_tiles = static_cast<int>(cfg.max_tiles);
    p.min_image_tokens = static_cast<int>(cfg.min_image_tokens);
    p.max_image_tokens = static_cast<int>(cfg.max_image_tokens);
    p.max_pixels_tolerance = cfg.max_pixels_tolerance;
    p.use_thumbnail = cfg.use_thumbnail;
    p.do_image_splitting = cfg.do_image_splitting;
    p.image_mean = cfg.image_mean;
    p.image_std = cfg.image_std;
    p.num_pos = num_pos;
    p.pos_size = static_cast<int>(std::lround(std::sqrt(static_cast<double>(num_pos))));
    return p;
}

// Returns (new_width, new_height) keeping aspect ratio and bounding the patch count.
std::pair<int, int> smart_resize(int height, int width, const Lfm2VlParams& p) {
    const int total_factor = p.patch_size * p.downsample_factor;
    const double min_pixels =
        static_cast<double>(p.min_image_tokens) * p.patch_size * p.patch_size * p.downsample_factor * p.downsample_factor;
    const double max_pixels =
        static_cast<double>(p.max_image_tokens) * p.patch_size * p.patch_size * p.downsample_factor * p.downsample_factor;

    int h_bar = std::max(total_factor, round_by_factor(height, total_factor));
    int w_bar = std::max(total_factor, round_by_factor(width, total_factor));

    if (static_cast<double>(h_bar) * w_bar > max_pixels) {
        const double beta = std::sqrt((static_cast<double>(height) * width) / max_pixels);
        h_bar = std::max(total_factor, static_cast<int>(std::floor(height / beta / total_factor)) * total_factor);
        w_bar = std::max(total_factor, static_cast<int>(std::floor(width / beta / total_factor)) * total_factor);
    } else if (static_cast<double>(h_bar) * w_bar < min_pixels) {
        const double beta = std::sqrt(min_pixels / (static_cast<double>(height) * width));
        h_bar = static_cast<int>(std::ceil(height * beta / total_factor)) * total_factor;
        w_bar = static_cast<int>(std::ceil(width * beta / total_factor)) * total_factor;
    }
    return {w_bar, h_bar};
}

bool is_image_too_large(int height, int width, const Lfm2VlParams& p) {
    const int total_factor = p.patch_size * p.downsample_factor;
    const int h_bar = std::max(p.patch_size, round_by_factor(height, total_factor));
    const int w_bar = std::max(p.patch_size, round_by_factor(width, total_factor));
    const double limit = static_cast<double>(p.max_image_tokens) * p.patch_size * p.patch_size *
                         p.downsample_factor * p.downsample_factor * p.max_pixels_tolerance;
    return static_cast<double>(h_bar) * w_bar > limit;
}

std::vector<std::pair<int, int>> target_ratios(int min_tiles, int max_tiles) {
    std::set<std::pair<int, int>> ratios;
    for (int n = min_tiles; n <= max_tiles; ++n) {
        for (int w = 1; w <= n; ++w) {
            for (int h = 1; h <= n; ++h) {
                if (min_tiles <= w * h && w * h <= max_tiles) {
                    ratios.emplace(w, h);
                }
            }
        }
    }
    std::vector<std::pair<int, int>> sorted(ratios.begin(), ratios.end());
    std::stable_sort(sorted.begin(), sorted.end(),
                     [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                         return a.first * a.second < b.first * b.second;
                     });
    return sorted;
}

// Returns grid (width, height).
std::pair<int, int> closest_aspect_ratio(double aspect_ratio,
                                         const std::vector<std::pair<int, int>>& ratios,
                                         int width,
                                         int height,
                                         int image_size) {
    double best_diff = std::numeric_limits<double>::infinity();
    std::pair<int, int> best{1, 1};
    const double area = static_cast<double>(width) * height;
    for (const auto& ratio : ratios) {
        const double target = static_cast<double>(ratio.first) / ratio.second;
        const double diff = std::abs(aspect_ratio - target);
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

clip_image_u8 crop_tile(const clip_image_u8& src, int x0, int y0, int tile_w, int tile_h) {
    clip_image_u8 tile;
    tile.nx = tile_w;
    tile.ny = tile_h;
    tile.buf.resize(static_cast<size_t>(tile_w) * tile_h * 3);
    for (int y = 0; y < tile_h; ++y) {
        const size_t src_row = (static_cast<size_t>(y0 + y) * src.nx + x0) * 3;
        const size_t dst_row = static_cast<size_t>(y) * tile_w * 3;
        std::copy_n(src.buf.begin() + src_row, static_cast<size_t>(tile_w) * 3, tile.buf.begin() + dst_row);
    }
    return tile;
}

}  // namespace

EncodedImage VisionEncoderLFM2VL::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    const ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);

    // The base positional-embedding count is the static last dimension of the
    // vision model's `interp_matrix` input (e.g. 256 for a 16x16 Siglip2 grid).
    int num_pos = 256;
    {
        CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue_vision_encoder.get());
        const auto interp_shape = guard.get().get_compiled_model().input("interp_matrix").get_partial_shape();
        OPENVINO_ASSERT(interp_shape.rank().get_length() == 3 && interp_shape[2].is_static(),
                        "LFM2-VL vision model must expose a static interp_matrix position dimension.");
        num_pos = static_cast<int>(interp_shape[2].get_length());
    }
    const Lfm2VlParams p = make_params(config, num_pos);

    const ov::Shape& shape = image.get_shape();
    OPENVINO_ASSERT(shape.size() == 4 && shape[0] == 1 && shape[1] > 0 && shape[2] > 0 && shape[3] == 3,
                    "LFM2-VL expects a non-empty RGB image with shape [1, height, width, 3].");
    const clip_image_u8 source = tensor_to_clip_image_u8(image);
    const int height = static_cast<int>(shape[1]);
    const int width = static_cast<int>(shape[2]);

    const auto [new_width, new_height] = smart_resize(height, width, p);
    const bool large = is_image_too_large(height, width, p);

    // Collect sub-images (tiles first row-major, then an optional thumbnail).
    std::vector<clip_image_u8> sub_images;
    int grid_rows = 1;  // grid_height
    int grid_cols = 1;  // grid_width
    if (large && p.do_image_splitting) {
        const double aspect_ratio = static_cast<double>(width) / height;
        const auto ratios = target_ratios(p.min_tiles, p.max_tiles);
        const auto [grid_w, grid_h] = closest_aspect_ratio(aspect_ratio, ratios, width, height, p.tile_size);
        grid_cols = grid_w;
        grid_rows = grid_h;
        clip_image_u8 resized;
        bicubic_antialias_resize(source, resized, p.tile_size * grid_w, p.tile_size * grid_h);
        for (int r = 0; r < grid_h; ++r) {
            for (int c = 0; c < grid_w; ++c) {
                sub_images.push_back(crop_tile(resized, c * p.tile_size, r * p.tile_size, p.tile_size, p.tile_size));
            }
        }
        if (p.use_thumbnail && grid_w * grid_h != 1) {
            clip_image_u8 thumbnail;
            bicubic_antialias_resize(source, thumbnail, new_width, new_height);
            sub_images.push_back(std::move(thumbnail));
        }
    } else {
        clip_image_u8 resized;
        bicubic_antialias_resize(source, resized, new_width, new_height);
        sub_images.push_back(std::move(resized));
    }

    // Run the exported vision model per sub-image and concatenate the projected features.
    const int ps = p.patch_size;
    const int patch_dim = ps * ps * 3;
    std::vector<ov::Tensor> features;
    std::vector<ImageSize> token_grids;
    size_t total_tokens = 0;
    size_t hidden_size = 0;
    for (const clip_image_u8& img : sub_images) {
        const int nph = img.ny / ps;
        const int npw = img.nx / ps;
        const int length = nph * npw;

        // Patchify + fused rescale/normalize into [1, length, patch_dim].
        ov::Tensor pixel_values(ov::element::f32, {1, static_cast<size_t>(length), static_cast<size_t>(patch_dim)});
        float* pv = pixel_values.data<float>();
        for (int ph = 0; ph < nph; ++ph) {
            for (int pw = 0; pw < npw; ++pw) {
                float* dst = pv + (static_cast<size_t>(ph) * npw + pw) * patch_dim;
                for (int yy = 0; yy < ps; ++yy) {
                    for (int xx = 0; xx < ps; ++xx) {
                        const size_t src = ((static_cast<size_t>(ph) * ps + yy) * img.nx + (pw * ps + xx)) * 3;
                        for (int c = 0; c < 3; ++c) {
                            *dst++ = (static_cast<float>(img.buf[src + c]) / 255.0f - p.image_mean[c]) / p.image_std[c];
                        }
                    }
                }
            }
        }

        // Interpolation matrix [1, length, num_pos] resizing the (pos_size, pos_size)
        // positional-embedding grid to (nph, npw).
        const auto wh = antialias_linear_weights(p.pos_size, nph);
        const auto ww = antialias_linear_weights(p.pos_size, npw);
        ov::Tensor interp_matrix(ov::element::f32, {1, static_cast<size_t>(length), static_cast<size_t>(p.num_pos)});
        float* im = interp_matrix.data<float>();
        std::fill_n(im, interp_matrix.get_size(), 0.0f);
        for (int y = 0; y < nph; ++y) {
            for (int x = 0; x < npw; ++x) {
                float* row = im + (static_cast<size_t>(y) * npw + x) * p.num_pos;
                for (int sy = 0; sy < p.pos_size; ++sy) {
                    const float wy = wh[y][sy];
                    if (wy == 0.0f) {
                        continue;
                    }
                    for (int sx = 0; sx < p.pos_size; ++sx) {
                        row[sy * p.pos_size + sx] = wy * ww[x][sx];
                    }
                }
            }
        }

        ov::Tensor spatial_shapes(ov::element::i64, {1, 2});
        spatial_shapes.data<int64_t>()[0] = nph;
        spatial_shapes.data<int64_t>()[1] = npw;

        CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue_vision_encoder.get());
        ov::InferRequest& encoder = guard.get();
        encoder.set_tensor("pixel_values", pixel_values);
        encoder.set_tensor("interp_matrix", interp_matrix);
        encoder.set_tensor("spatial_shapes", spatial_shapes);
        encoder.infer();
        const ov::Tensor out = encoder.get_output_tensor();
        const ov::Shape& out_shape = out.get_shape();
        OPENVINO_ASSERT(out_shape.size() == 2, "LFM2-VL vision output must be a 2D [tokens, hidden] tensor.");
        hidden_size = out_shape[1];

        ov::Tensor feature(out.get_element_type(), out_shape);
        out.copy_to(feature);
        features.push_back(std::move(feature));
        token_grids.push_back({static_cast<size_t>(nph / p.downsample_factor),
                               static_cast<size_t>(npw / p.downsample_factor)});
        total_tokens += out_shape[0];
    }

    // Concatenate projected features into a single [1, total_tokens, hidden] tensor.
    ov::Tensor merged(features.front().get_element_type(), {1, total_tokens, hidden_size});
    float* merged_data = merged.data<float>();
    size_t offset = 0;
    for (const ov::Tensor& feature : features) {
        std::copy_n(feature.data<float>(), feature.get_size(), merged_data + offset);
        offset += feature.get_size();
    }

    EncodedImage result;
    result.resized_source = std::move(merged);
    result.resized_source_size = {static_cast<size_t>(new_height / ps), static_cast<size_t>(new_width / ps)};
    result.patches_grid = {grid_rows, grid_cols};
    result.crop_sizes = std::move(token_grids);
    result.num_image_tokens = total_tokens;
    return result;
}

InputsEmbedderLFM2VL::InputsEmbedderLFM2VL(const VLMConfig& vlm_config,
                                           const std::filesystem::path& model_dir,
                                           const Tokenizer& tokenizer,
                                           const std::string& device,
                                           const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) {}

InputsEmbedderLFM2VL::InputsEmbedderLFM2VL(const VLMConfig& vlm_config,
                                           const ModelsMap& models_map,
                                           const Tokenizer& tokenizer,
                                           const std::filesystem::path& config_dir_path,
                                           const std::string& device,
                                           const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

std::vector<EncodedImage> InputsEmbedderLFM2VL::encode_images(const std::vector<ov::Tensor>& images) {
    const auto single_images = to_single_image_tensors(images);
    std::vector<EncodedImage> encoded_images;
    encoded_images.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        encoded_images.emplace_back(m_vision_encoder->encode(image));
    }
    return encoded_images;
}

NormalizedPrompt InputsEmbedderLFM2VL::normalize_prompt(const std::string& prompt,
                                                        size_t base_id,
                                                        const std::vector<EncodedImage>& images) const {
    // LFM2-VL special tokens are fixed by the Lfm2VlProcessor implementation.
    const std::string image_token = "<image>";
    const std::string image_start = "<|image_start|>";
    const std::string image_end = "<|image_end|>";
    const std::string image_thumbnail = "<|img_thumbnail|>";

    auto [unified_prompt, images_sequence] = normalize(prompt, image_token, image_token, base_id, images.size());

    size_t offset = 0;
    for (size_t image_id : images_sequence) {
        const EncodedImage& image = images.at(image_id - base_id);
        const int rows = image.patches_grid.first;
        const int cols = image.patches_grid.second;
        const bool multi_tile = rows > 1 || cols > 1;

        // Reproduce Lfm2VlProcessor._build_image_tokens: the per-image expansion is
        // <|image_start|> [ (<|img_row_r_col_c|> + image_token*tokens_per_tile) ... ]
        // [ <|img_thumbnail|> + image_token*tokens_for_thumbnail ] <|image_end|>.
        std::string rebuilt = image_start;
        size_t crop_idx = 0;
        if (multi_tile) {
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    rebuilt += "<|img_row_" + std::to_string(r + 1) + "_col_" + std::to_string(c + 1) + "|>";
                    const ImageSize& grid = image.crop_sizes.at(crop_idx++);
                    for (size_t t = 0; t < grid.height * grid.width; ++t) {
                        rebuilt += image_token;
                    }
                }
            }
            if (crop_idx < image.crop_sizes.size()) {
                rebuilt += image_thumbnail;
                const ImageSize& grid = image.crop_sizes.at(crop_idx++);
                for (size_t t = 0; t < grid.height * grid.width; ++t) {
                    rebuilt += image_token;
                }
            }
        } else {
            const ImageSize& grid = image.crop_sizes.at(crop_idx++);
            for (size_t t = 0; t < grid.height * grid.width; ++t) {
                rebuilt += image_token;
            }
        }
        rebuilt += image_end;

        const size_t pos = unified_prompt.find(image_token, offset);
        OPENVINO_ASSERT(pos != std::string::npos, "LFM2-VL image tag is missing during prompt normalization.");
        unified_prompt.replace(pos, image_token.size(), rebuilt);
        offset = pos + rebuilt.size();
    }
    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderLFM2VL::get_inputs_embeds(const std::string& prompt,
                                                   const std::vector<EncodedImage>& images,
                                                   VLMPerfMetrics& metrics,
                                                   bool recalculate_merged_embeddings,
                                                   const std::vector<size_t>& image_sequence) {
    const ov::Tensor input_ids = get_encoded_input_ids(prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> guard(m_embedding->get_request_queue().get());
    ov::Tensor text_embeds = get_text_embedding(guard.get(), input_ids, metrics);

    std::vector<ov::Tensor> image_embeds;
    for (size_t image_id : image_sequence) {
        image_embeds.push_back(images.at(image_id).resized_source);
    }

    const size_t hidden_size = text_embeds.get_shape()[2];
    const int64_t* ids = input_ids.data<const int64_t>();
    size_t num_tokens = 0;
    for (auto& feature : image_embeds) {
        const auto& shape = feature.get_shape();
        OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1 && shape[2] == hidden_size,
                        "LFM2-VL vision embeddings must have shape [1, tokens, text hidden size].");
        num_tokens += shape[1];
        feature = ov::Tensor(feature.get_element_type(), {shape[1], hidden_size}, feature.data());
    }
    OPENVINO_ASSERT(
        static_cast<size_t>(std::count(ids, ids + input_ids.get_size(), m_vlm_config.image_token_id)) == num_tokens,
        "LFM2-VL image token count does not match the number of vision embeddings.");

    const ov::Tensor flat_images =
        image_embeds.empty() ? ov::Tensor(text_embeds.get_element_type(), {0, hidden_size})
                             : qwen2_vl_utils::concatenate_video_image_embeds({}, image_embeds);
    const ov::Tensor empty_videos(text_embeds.get_element_type(), {0, hidden_size});
    ov::Tensor merged = qwen2_vl_utils::merge_text_and_video_image_embeddings(input_ids, text_embeds, flat_images, empty_videos,
                                                                 m_vlm_config.image_token_id,
                                                                 m_vlm_config.video_token_id);
    return merged;
}

}  // namespace ov::genai
