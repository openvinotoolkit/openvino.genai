// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_5preprocessor.hpp"
#include "openvino/core/except.hpp"
#include "nlohmann/json.hpp"
#include <fstream>
#include <cmath>
#include <algorithm>

namespace ov::genai::module {

struct PreparedImage {
    std::vector<float> data;
    size_t frames = 0;
    size_t height = 0;
    size_t width = 0;
    int64_t grid_t = 0;
    int64_t grid_h = 0;
    int64_t grid_w = 0;
};

Qwen3_5Preprocessor::Qwen3_5Preprocessor(const std::filesystem::path &model_path)
    : m_preprocess_config(Qwen3_5VisionPreprocessConfig::from_json_file(model_path / "preprocessor_config.json")),
      m_vision_config(Qwen3_5VisionConfig::from_json_file(model_path / "config.json")) {
    load_pos_embed_weight(model_path);
}

Qwen3_5PreprocessorOutput Qwen3_5Preprocessor::preprocess(const ov::Tensor &images) {
    const auto img_shape = images.get_shape();
    if (img_shape.size() != 3 && img_shape.size() != 4) {
        OPENVINO_THROW("images must have shape [H, W, C] or [B, H, W, C]");
    }
    if (images.get_element_type() != ov::element::u8) {
        OPENVINO_THROW("images must be u8 for Qwen3_5 preprocessing");
    }

    const bool has_batch = img_shape.size() == 4;
    const size_t batch = has_batch ? img_shape[0] : 1;
    const size_t in_h = has_batch ? img_shape[1] : img_shape[0];
    const size_t in_w = has_batch ? img_shape[2] : img_shape[1];
    const size_t channels = has_batch ? img_shape[3] : img_shape[2];
    if (channels != 3) {
        OPENVINO_THROW("images must have 3 channels");
    }

    const size_t factor = static_cast<size_t>(m_preprocess_config.patch_size * m_preprocess_config.merge_size);
    const uint8_t* src = images.data<const uint8_t>();
    const bool nchw = false;

    std::vector<PreparedImage> prepared;
    prepared.reserve(batch);

    for (size_t b = 0; b < batch; ++b) {
        const uint8_t* src_img = src + b * in_h * in_w * channels;
        size_t out_h = in_h;
        size_t out_w = in_w;
        if (m_preprocess_config.do_resize) {
            auto resized = smart_resize(in_h,
                                        in_w,
                                        factor);
            out_h = resized.first;
            out_w = resized.second;
        }
        if (out_h % m_preprocess_config.patch_size != 0 || out_w % m_preprocess_config.patch_size != 0) {
            OPENVINO_THROW("Resized image must be divisible by patch_size");
        }

        std::vector<float> frame;
        resize_bilinear_to_chw(src_img,
                               in_h,
                               in_w,
                               channels,
                               nchw,
                               out_h,
                               out_w,
                               frame);

        const size_t frames = 1;
        size_t padded_frames = frames;
        if (frames % static_cast<size_t>(m_preprocess_config.temporal_patch_size) != 0) {
            padded_frames += static_cast<size_t>(m_preprocess_config.temporal_patch_size) -
                             (frames % static_cast<size_t>(m_preprocess_config.temporal_patch_size));
        }

        std::vector<float> stacked(padded_frames * frame.size());
        for (size_t t = 0; t < padded_frames; ++t) {
            const size_t dst_offset = t * frame.size();
            std::copy(frame.begin(), frame.end(), stacked.begin() + dst_offset);
        }

        PreparedImage item;
        item.data = std::move(stacked);
        item.frames = padded_frames;
        item.height = out_h;
        item.width = out_w;
        item.grid_t = static_cast<int64_t>(padded_frames / static_cast<size_t>(m_preprocess_config.temporal_patch_size));
        item.grid_h = static_cast<int64_t>(out_h / static_cast<size_t>(m_preprocess_config.patch_size));
        item.grid_w = static_cast<int64_t>(out_w / static_cast<size_t>(m_preprocess_config.patch_size));
        prepared.push_back(std::move(item));
    }

    int64_t total_patches = 0;
    for (const auto& item : prepared) {
        total_patches += item.grid_t * item.grid_h * item.grid_w;
    }

    ov::Tensor grid_thw(ov::element::i64, {batch, 3});
    auto* grid = grid_thw.data<int64_t>();
    for (size_t b = 0; b < prepared.size(); ++b) {
        grid[b * 3 + 0] = prepared[b].grid_t;
        grid[b * 3 + 1] = prepared[b].grid_h;
        grid[b * 3 + 2] = prepared[b].grid_w;
    }

    const size_t patch_size = static_cast<size_t>(m_preprocess_config.patch_size);
    const size_t temporal_patch = static_cast<size_t>(m_preprocess_config.temporal_patch_size);
    const size_t merge_size = static_cast<size_t>(m_preprocess_config.merge_size);
    const size_t patch_stride = channels * temporal_patch * patch_size * patch_size;

    ov::Tensor pixel_values(ov::element::f32,
                            {static_cast<size_t>(total_patches),
                             channels,
                             temporal_patch,
                             patch_size,
                             patch_size});
    float* out = pixel_values.data<float>();
    size_t patch_offset = 0;

    for (const auto& item : prepared) {
        const size_t height = item.height;
        const size_t width = item.width;
        const size_t frame_stride = channels * height * width;
        const float* data = item.data.data();
        const size_t grid_t = static_cast<size_t>(item.grid_t);
        const size_t grid_h = static_cast<size_t>(item.grid_h);
        const size_t grid_w = static_cast<size_t>(item.grid_w);
        if (grid_h % merge_size != 0 || grid_w % merge_size != 0) {
            OPENVINO_THROW("grid_h/grid_w must be divisible by merge_size");
        }
        const size_t merged_h = grid_h / merge_size;
        const size_t merged_w = grid_w / merge_size;

        for (size_t t = 0; t < grid_t; ++t) {
            for (size_t bh = 0; bh < merged_h; ++bh) {
                for (size_t bw = 0; bw < merged_w; ++bw) {
                    for (size_t mh = 0; mh < merge_size; ++mh) {
                        for (size_t mw = 0; mw < merge_size; ++mw) {
                            float* dst = out + patch_offset * patch_stride;
                            size_t dst_idx = 0;
                            const size_t h_idx = (bh * merge_size + mh) * patch_size;
                            const size_t w_idx = (bw * merge_size + mw) * patch_size;
                            for (size_t c = 0; c < channels; ++c) {
                                for (size_t tp = 0; tp < temporal_patch; ++tp) {
                                    const size_t t_idx = (t * temporal_patch + tp) * frame_stride;
                                    for (size_t ph = 0; ph < patch_size; ++ph) {
                                        for (size_t pw = 0; pw < patch_size; ++pw) {
                                            const size_t src_idx =
                                                t_idx + (c * height + h_idx + ph) * width + w_idx + pw;
                                            dst[dst_idx++] = data[src_idx];
                                        }
                                    }
                                }
                            }
                            patch_offset++;
                        }
                    }
                }
            }
        }
    }

    auto pos_embeds = build_pos_embeds(grid_thw);
    auto rotary = build_rotary_cos_sin(grid_thw);

    return {pixel_values, grid_thw, pos_embeds, rotary.first, rotary.second};
}

void Qwen3_5Preprocessor::load_pos_embed_weight(const std::filesystem::path &model_path) {
    std::ifstream meta_data_file(model_path / "pos_embed_weight.json");
    if (!meta_data_file) {
        OPENVINO_THROW("Cannot open: " + (model_path / "pos_embed_weight.json").string());
    }
    nlohmann::json meta = nlohmann::json::parse(meta_data_file);
    const ov::element::Type dtype = parse_ov_dtype(meta.at("dtype").get<std::string>());
    const std::vector<size_t> shape_vec = meta.at("shape").get<std::vector<size_t>>();
    const size_t expected_bytes = meta.at("byte_size").get<size_t>();
    const ov::Shape shape(shape_vec.begin(), shape_vec.end());
    m_pos_embed_weight = ov::Tensor(dtype, shape);
    if (m_pos_embed_weight.get_byte_size() != expected_bytes) {
        OPENVINO_THROW("byte_size mismatch: json=" +
                       std::to_string(expected_bytes) +
                       " tensor=" + std::to_string(m_pos_embed_weight.get_byte_size()));
    }
    std::ifstream bin_file(model_path / "pos_embed_weight.bin", std::ios::binary);
    if (!bin_file) {
        OPENVINO_THROW("Cannot open: " + (model_path / "pos_embed_weight.bin").string());
    }
    bin_file.read(reinterpret_cast<char*>(m_pos_embed_weight.data()), expected_bytes);
    if (static_cast<size_t>(bin_file.gcount()) != expected_bytes) {
        OPENVINO_THROW("Incomplete read from: " + (model_path / "pos_embed_weight.bin").string());
    }
}

ov::element::Type Qwen3_5Preprocessor::parse_ov_dtype(const std::string &s) {
    if (s == "f32")  return ov::element::f32;
    if (s == "f16")  return ov::element::f16;
    if (s == "bf16") return ov::element::bf16;
    if (s == "i32")  return ov::element::i32;
    if (s == "i16")  return ov::element::i16;
    if (s == "i8")   return ov::element::i8;
    if (s == "u8")   return ov::element::u8;
    OPENVINO_THROW("Unknown dtype in sidecar: " + s);
}

std::pair<size_t, size_t> Qwen3_5Preprocessor::smart_resize(size_t height,
                                                            size_t width,
                                                            size_t factor) {
    size_t min_pixels = static_cast<size_t>(m_preprocess_config.min_pixels);
    size_t max_pixels = static_cast<size_t>(m_preprocess_config.max_pixels);
    if (height < factor || width < factor) {
        OPENVINO_THROW("Height and width must be >= resize factor");
    }
    if (std::max(height, width) / std::min(height, width) > 200) {
        OPENVINO_THROW("Absolute aspect ratio must be smaller than 200");
    }

    auto round_to_factor = [factor](double value) {
        return static_cast<size_t>(std::round(value / static_cast<double>(factor)) * factor);
    };

    size_t h_bar = round_to_factor(static_cast<double>(height));
    size_t w_bar = round_to_factor(static_cast<double>(width));

    const double pixels = static_cast<double>(height) * static_cast<double>(width);
    if (static_cast<double>(h_bar) * static_cast<double>(w_bar) > static_cast<double>(max_pixels)) {
        double beta = std::sqrt(pixels / static_cast<double>(max_pixels));
        h_bar = std::max(factor, static_cast<size_t>(std::floor(height / beta / factor) * factor));
        w_bar = std::max(factor, static_cast<size_t>(std::floor(width / beta / factor) * factor));
    } else if (static_cast<double>(h_bar) * static_cast<double>(w_bar) < static_cast<double>(min_pixels)) {
        double beta = std::sqrt(static_cast<double>(min_pixels) / pixels);
        h_bar = static_cast<size_t>(std::ceil(height * beta / factor) * factor);
        w_bar = static_cast<size_t>(std::ceil(width * beta / factor) * factor);
    }

    return {h_bar, w_bar};
}

void Qwen3_5Preprocessor::resize_bilinear_to_chw(const uint8_t *src,
                                                 size_t src_h,
                                                 size_t src_w,
                                                 size_t channels,
                                                 bool nchw,
                                                 size_t dst_h,
                                                 size_t dst_w,
                                                 std::vector<float> &dst_chw) {
    dst_chw.assign(channels * dst_h * dst_w, 0.0f);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);
    const float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);

    auto fetch = [&](size_t y, size_t x, size_t c) -> float {
        size_t idx = 0;
        if (nchw) {
            idx = (c * src_h + y) * src_w + x;
        } else {
            idx = (y * src_w + x) * channels + c;
        }
        return static_cast<float>(src[idx]);
    };

    for (size_t y = 0; y < dst_h; ++y) {
        float in_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        in_y = std::max(0.0f, std::min(in_y, static_cast<float>(src_h - 1)));
        size_t y0 = static_cast<size_t>(std::floor(in_y));
        size_t y1 = std::min(y0 + 1, src_h - 1);
        float wy = in_y - static_cast<float>(y0);
        for (size_t x = 0; x < dst_w; ++x) {
            float in_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            in_x = std::max(0.0f, std::min(in_x, static_cast<float>(src_w - 1)));
            size_t x0 = static_cast<size_t>(std::floor(in_x));
            size_t x1 = std::min(x0 + 1, src_w - 1);
            float wx = in_x - static_cast<float>(x0);
            const float w00 = (1.0f - wy) * (1.0f - wx);
            const float w01 = (1.0f - wy) * wx;
            const float w10 = wy * (1.0f - wx);
            const float w11 = wy * wx;

            for (size_t c = 0; c < channels; ++c) {
                float v = 0.0f;
                v += w00 * fetch(y0, x0, c);
                v += w01 * fetch(y0, x1, c);
                v += w10 * fetch(y1, x0, c);
                v += w11 * fetch(y1, x1, c);
                const float norm = (v / 255.0f - m_preprocess_config.image_mean[c]) / m_preprocess_config.image_std[c];
                const size_t out_idx = (c * dst_h + y) * dst_w + x;
                dst_chw[out_idx] = norm;
            }
        }
    }
}

ov::Tensor Qwen3_5Preprocessor::build_pos_embeds(const ov::Tensor &grid_thw) {
    auto weight_f32 = to_f32(m_pos_embed_weight);
    const auto weight_shape = weight_f32.get_shape();
    if (weight_shape.size() != 2) {
        OPENVINO_THROW("pos_embed weight must be 2D");
    }
    const int64_t num_pos = static_cast<int64_t>(weight_shape[0]);
    const int64_t hidden = static_cast<int64_t>(weight_shape[1]);
    const int64_t num_grid = static_cast<int64_t>(std::llround(std::sqrt(static_cast<double>(num_pos))));
    if (num_grid * num_grid != num_pos) {
        OPENVINO_THROW("pos_embed num_position_embeddings must be a square");
    }

    if (grid_thw.get_element_type() != ov::element::i64) {
        OPENVINO_THROW("grid_thw must be i64");
    }
    const auto grid_shape = grid_thw.get_shape();
    if (grid_shape.size() != 2 || grid_shape[1] != 3) {
        OPENVINO_THROW("grid_thw must have shape [N, 3]");
    }

    const int64_t* grid = grid_thw.data<const int64_t>();
    const size_t num_images = grid_shape[0];
    int64_t total_tokens = 0;
    for (size_t i = 0; i < num_images; ++i) {
        total_tokens += grid[i * 3 + 0] * grid[i * 3 + 1] * grid[i * 3 + 2];
    }

    ov::Tensor pos_embeds(ov::element::f32, {static_cast<size_t>(total_tokens), static_cast<size_t>(hidden)});
    float* out = pos_embeds.data<float>();
    const float* weight = weight_f32.data<const float>();
    size_t offset = 0;

    for (size_t i = 0; i < num_images; ++i) {
        const int64_t t = grid[i * 3 + 0];
        const int64_t h = grid[i * 3 + 1];
        const int64_t w = grid[i * 3 + 2];
        if (t <= 0 || h <= 0 || w <= 0) {
            OPENVINO_THROW("Invalid grid_thw for pos_embed interpolation");
        }
        if (h % m_preprocess_config.merge_size != 0 || w % m_preprocess_config.merge_size != 0) {
            OPENVINO_THROW("grid_thw must be divisible by merge_size");
        }

        std::vector<float> h_pos(static_cast<size_t>(h));
        std::vector<float> w_pos(static_cast<size_t>(w));
        if (h == 1) {
            h_pos[0] = 0.0f;
        } else {
            const float step = static_cast<float>(num_grid - 1) / static_cast<float>(h - 1);
            for (int64_t y = 0; y < h; ++y) {
                h_pos[static_cast<size_t>(y)] = static_cast<float>(y) * step;
            }
        }
        if (w == 1) {
            w_pos[0] = 0.0f;
        } else {
            const float step = static_cast<float>(num_grid - 1) / static_cast<float>(w - 1);
            for (int64_t x = 0; x < w; ++x) {
                w_pos[static_cast<size_t>(x)] = static_cast<float>(x) * step;
            }
        }

        std::vector<int64_t> h_floor(static_cast<size_t>(h));
        std::vector<int64_t> h_ceil(static_cast<size_t>(h));
        std::vector<float> h_lerp(static_cast<size_t>(h));
        for (int64_t y = 0; y < h; ++y) {
            float val = h_pos[static_cast<size_t>(y)];
            int64_t f = static_cast<int64_t>(std::floor(val));
            int64_t c = std::min(f + 1, num_grid - 1);
            h_floor[static_cast<size_t>(y)] = f;
            h_ceil[static_cast<size_t>(y)] = c;
            h_lerp[static_cast<size_t>(y)] = val - static_cast<float>(f);
        }

        std::vector<int64_t> w_floor(static_cast<size_t>(w));
        std::vector<int64_t> w_ceil(static_cast<size_t>(w));
        std::vector<float> w_lerp(static_cast<size_t>(w));
        for (int64_t x = 0; x < w; ++x) {
            float val = w_pos[static_cast<size_t>(x)];
            int64_t f = static_cast<int64_t>(std::floor(val));
            int64_t c = std::min(f + 1, num_grid - 1);
            w_floor[static_cast<size_t>(x)] = f;
            w_ceil[static_cast<size_t>(x)] = c;
            w_lerp[static_cast<size_t>(x)] = val - static_cast<float>(f);
        }

        std::vector<float> pos_hw(static_cast<size_t>(h * w * hidden));
        for (int64_t yy = 0; yy < h; ++yy) {
            const float dh = h_lerp[static_cast<size_t>(yy)];
            const int64_t y0 = h_floor[static_cast<size_t>(yy)];
            const int64_t y1 = h_ceil[static_cast<size_t>(yy)];
            const int64_t base_y0 = y0 * num_grid;
            const int64_t base_y1 = y1 * num_grid;
            for (int64_t xx = 0; xx < w; ++xx) {
                const float dw = w_lerp[static_cast<size_t>(xx)];
                const int64_t x0 = w_floor[static_cast<size_t>(xx)];
                const int64_t x1 = w_ceil[static_cast<size_t>(xx)];
                const float w00 = (1.0f - dh) * (1.0f - dw);
                const float w01 = (1.0f - dh) * dw;
                const float w10 = dh * (1.0f - dw);
                const float w11 = dh * dw;

                const int64_t idx00 = base_y0 + x0;
                const int64_t idx01 = base_y0 + x1;
                const int64_t idx10 = base_y1 + x0;
                const int64_t idx11 = base_y1 + x1;

                const size_t out_base = static_cast<size_t>((yy * w + xx) * hidden);
                const size_t w00_base = static_cast<size_t>(idx00 * hidden);
                const size_t w01_base = static_cast<size_t>(idx01 * hidden);
                const size_t w10_base = static_cast<size_t>(idx10 * hidden);
                const size_t w11_base = static_cast<size_t>(idx11 * hidden);

                for (int64_t hidx = 0; hidx < hidden; ++hidx) {
                    float value = 0.0f;
                    value += w00 * weight[w00_base + static_cast<size_t>(hidx)];
                    value += w01 * weight[w01_base + static_cast<size_t>(hidx)];
                    value += w10 * weight[w10_base + static_cast<size_t>(hidx)];
                    value += w11 * weight[w11_base + static_cast<size_t>(hidx)];
                    pos_hw[out_base + static_cast<size_t>(hidx)] = value;
                }
            }
        }

        const int64_t merged_h = h / m_preprocess_config.merge_size;
        const int64_t merged_w = w / m_preprocess_config.merge_size;
        for (int64_t tt = 0; tt < t; ++tt) {
            (void)tt;
            for (int64_t bh = 0; bh < merged_h; ++bh) {
                for (int64_t bw = 0; bw < merged_w; ++bw) {
                    for (int64_t mh = 0; mh < m_preprocess_config.merge_size; ++mh) {
                        for (int64_t mw = 0; mw < m_preprocess_config.merge_size; ++mw) {
                            const int64_t h_idx = bh * m_preprocess_config.merge_size + mh;
                            const int64_t w_idx = bw * m_preprocess_config.merge_size + mw;
                            const size_t src_base = static_cast<size_t>((h_idx * w + w_idx) * hidden);
                            const size_t dst_base = offset * static_cast<size_t>(hidden);
                            std::copy_n(pos_hw.data() + src_base, static_cast<size_t>(hidden), out + dst_base);
                            offset++;
                        }
                    }
                }
            }
        }
    }

    return pos_embeds;
}

ov::Tensor Qwen3_5Preprocessor::to_f32(const ov::Tensor &src) {
    if (src.get_element_type() == ov::element::f32) {
        return src;
    }
    ov::Tensor dst(ov::element::f32, src.get_shape());
    float* out = dst.data<float>();
    const size_t total = src.get_size();

    if (src.get_element_type() == ov::element::f16) {
        const auto* in = src.data<const ov::float16>();
        for (size_t i = 0; i < total; ++i) {
            out[i] = static_cast<float>(in[i]);
        }
        return dst;
    }
    if (src.get_element_type() == ov::element::bf16) {
        const auto* in = src.data<const ov::bfloat16>();
        for (size_t i = 0; i < total; ++i) {
            out[i] = static_cast<float>(in[i]);
        }
        return dst;
    }
    if (src.get_element_type() == ov::element::f64) {
        const auto* in = src.data<const double>();
        for (size_t i = 0; i < total; ++i) {
            out[i] = static_cast<float>(in[i]);
        }
        return dst;
    }

    OPENVINO_THROW("Unsupported pos_embed dtype for Qwen3_5 preprocessing");
}

std::pair<ov::Tensor, ov::Tensor> Qwen3_5Preprocessor::build_rotary_cos_sin(const ov::Tensor& grid_thw) {
    if (grid_thw.get_element_type() != ov::element::i64) {
        OPENVINO_THROW("grid_thw must be i64");
    }
    const auto grid_shape = grid_thw.get_shape();
    if (grid_shape.size() != 2 || grid_shape[1] != 3) {
        OPENVINO_THROW("grid_thw must have shape [N, 3]");
    }
    const int64_t* grid = grid_thw.data<const int64_t>();
    const size_t num_images = grid_shape[0];

    int64_t total_tokens = 0;
    for (size_t i = 0; i < num_images; ++i) {
        const int64_t t = grid[i * 3 + 0];
        const int64_t h = grid[i * 3 + 1];
        const int64_t w = grid[i * 3 + 2];
        total_tokens += t * h * w;
    }

    const int32_t head_dim = m_vision_config.head_dim();
    if (head_dim <= 0 || head_dim % 2 != 0) {
        OPENVINO_THROW("Invalid head_dim for Qwen3_5 rotary embedding");
    }
    const int32_t rotary_dim = head_dim / 2;
    if (rotary_dim % 2 != 0) {
        OPENVINO_THROW("Vision rotary_dim must be even");
    }
    const int32_t inv_len = rotary_dim / 2;
    const float theta = 10000.0f;

    std::vector<float> inv_freq(static_cast<size_t>(inv_len));
    for (int32_t i = 0; i < inv_len; ++i) {
        float exponent = static_cast<float>(2 * i) / static_cast<float>(rotary_dim);
        inv_freq[static_cast<size_t>(i)] = 1.0f / std::pow(theta, exponent);
    }

    ov::Tensor rotary_cos(ov::element::f32, {static_cast<size_t>(total_tokens), static_cast<size_t>(head_dim)});
    ov::Tensor rotary_sin(ov::element::f32, {static_cast<size_t>(total_tokens), static_cast<size_t>(head_dim)});
    float* cos_out = rotary_cos.data<float>();
    float* sin_out = rotary_sin.data<float>();

    size_t offset = 0;
    for (size_t i = 0; i < num_images; ++i) {
        const int64_t t = grid[i * 3 + 0];
        const int64_t h = grid[i * 3 + 1];
        const int64_t w = grid[i * 3 + 2];
        if (t <= 0 || h <= 0 || w <= 0) {
            OPENVINO_THROW("Invalid grid_thw for rotary embedding");
        }
        if (h % m_preprocess_config.merge_size != 0 || w % m_preprocess_config.merge_size != 0) {
            OPENVINO_THROW("grid_thw must be divisible by merge_size");
        }
        const int64_t merged_h = h / m_preprocess_config.merge_size;
        const int64_t merged_w = w / m_preprocess_config.merge_size;

        for (int64_t tt = 0; tt < t; ++tt) {
            (void)tt;
            for (int64_t bh = 0; bh < merged_h; ++bh) {
                for (int64_t bw = 0; bw < merged_w; ++bw) {
                    for (int64_t mh = 0; mh < m_preprocess_config.merge_size; ++mh) {
                        for (int64_t mw = 0; mw < m_preprocess_config.merge_size; ++mw) {
                            const int64_t row = bh * m_preprocess_config.merge_size + mh;
                            const int64_t col = bw * m_preprocess_config.merge_size + mw;
                            float* cos_ptr = cos_out + offset * static_cast<size_t>(head_dim);
                            float* sin_ptr = sin_out + offset * static_cast<size_t>(head_dim);

                            for (int32_t j = 0; j < inv_len; ++j) {
                                const float inv = inv_freq[static_cast<size_t>(j)];
                                const float row_freq = static_cast<float>(row) * inv;
                                const float col_freq = static_cast<float>(col) * inv;
                                const float cos_row = std::cos(row_freq);
                                const float sin_row = std::sin(row_freq);
                                const float cos_col = std::cos(col_freq);
                                const float sin_col = std::sin(col_freq);

                                const int32_t row_idx = j;
                                const int32_t col_idx = inv_len + j;

                                cos_ptr[row_idx] = cos_row;
                                sin_ptr[row_idx] = sin_row;
                                cos_ptr[col_idx] = cos_col;
                                sin_ptr[col_idx] = sin_col;
                            }
                            for (int32_t j = 0; j < rotary_dim; ++j) {
                                cos_ptr[rotary_dim + j] = cos_ptr[j];
                                sin_ptr[rotary_dim + j] = sin_ptr[j];
                            }

                            offset++;
                        }
                    }
                }
            }
        }
    }

    return {rotary_cos, rotary_sin};
}

}