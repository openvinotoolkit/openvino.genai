// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>

#include "openvino/core/except.hpp"
#include "openvino/runtime/tensor.hpp"
#include "visual_language/clip.hpp"

// Host-side parts of LTX-2 image and video conditioning, following diffusers' LTX2ConditionPipeline
// (preprocess_conditions, trim_conditioning_sequence and _prepare_keyframe_coords).
namespace ov::genai::ltx2_conditioning {

/// A condition resolved against the output video: frames trimmed to fit, a non-negative latent index
struct ResolvedCondition {
    ov::Tensor frames;  // u8 [F, H, W, 3]
    size_t latent_index = 0;
    float strength = 1.0f;
};

/// Views an image [H, W, 3] as a one-frame video [1, H, W, 3]
inline ov::Tensor as_video(const ov::Tensor& frames) {
    const ov::Shape shape = frames.get_shape();
    OPENVINO_ASSERT(frames.get_element_type() == ov::element::u8, "Condition frames must be u8, got ",
                    frames.get_element_type());
    OPENVINO_ASSERT((shape.size() == 3 || shape.size() == 4) && shape.back() == 3,
                    "Condition frames must be [F, H, W, 3] or [H, W, 3], got ", shape);
    if (shape.size() == 4) {
        return frames;
    }
    return ov::Tensor(frames.get_element_type(), {1, shape[0], shape[1], shape[2]}, frames.data());
}

/// The number of frames of a condition that starts at 'start_frame' to keep: what fits into the video, trimmed to
/// 8 * k + 1 frames (diffusers' trim_conditioning_sequence)
inline size_t trim_num_frames(size_t start_frame, size_t sequence_num_frames, size_t target_num_frames,
                              size_t temporal_ratio) {
    const size_t num_frames = std::min(sequence_num_frames, target_num_frames - start_frame);
    return (num_frames - 1) / temporal_ratio * temporal_ratio + 1;
}

/// Resolves a latent index against 'latent_num_frames': negative ones count from the end. Returns false for an index
/// past the end, whose condition is skipped.
inline bool resolve_latent_index(int64_t index, size_t latent_num_frames, size_t& latent_index) {
    const auto frames = static_cast<int64_t>(latent_num_frames);
    if (index < 0) {
        index = ((index % frames) + frames) % frames;
    }
    if (index >= frames) {
        return false;
    }
    latent_index = static_cast<size_t>(index);
    return true;
}

/// u8 frames [F, H, W, 3] -> f32 [1, 3, F, height, width] in [-1, 1]: scaled so the longer side fills the target and
/// center-cropped. Bilinear like torch's F.interpolate(mode="bilinear", align_corners=False), without antialiasing.
inline ov::Tensor preprocess_frames(const ov::Tensor& frames, size_t height, size_t width) {
    const ov::Shape shape = frames.get_shape();
    const size_t num_frames = shape[0], src_h = shape[1], src_w = shape[2];
    const double scale = std::max(static_cast<double>(height) / src_h, static_cast<double>(width) / src_w);
    const auto new_h = static_cast<size_t>(std::ceil(src_h * scale));
    const auto new_w = static_cast<size_t>(std::ceil(src_w * scale));
    const size_t top = (new_h - height) / 2, left = (new_w - width) / 2;

    // Source taps along one axis for every kept output position: torch's area_pixel_compute_source_index
    struct Tap {
        size_t i0, i1;
        float l0, l1;
    };
    auto taps = [](size_t in_size, size_t out_size, size_t offset, size_t count) {
        const float ratio = static_cast<float>(in_size) / static_cast<float>(out_size);
        std::vector<Tap> result(count);
        for (size_t k = 0; k < count; ++k) {
            const float src = std::max(ratio * (static_cast<float>(offset + k) + 0.5f) - 0.5f, 0.0f);
            const auto i0 = static_cast<size_t>(src);
            const size_t i1 = i0 + (i0 < in_size - 1 ? 1 : 0);
            const float l1 = src - static_cast<float>(i0);
            result[k] = {i0, i1, 1.0f - l1, l1};
        }
        return result;
    };
    const std::vector<Tap> rows = taps(src_h, new_h, top, height);
    const std::vector<Tap> cols = taps(src_w, new_w, left, width);

    ov::Tensor pixels(ov::element::f32, {1, 3, num_frames, height, width});
    float* out = pixels.data<float>();
    const uint8_t* in = frames.data<const uint8_t>();
    const size_t plane = num_frames * height * width;
    for (size_t f = 0; f < num_frames; ++f) {
        const uint8_t* frame = in + f * src_h * src_w * 3;
        for (size_t y = 0; y < height; ++y) {
            const Tap& r = rows[y];
            for (size_t x = 0; x < width; ++x) {
                const Tap& c = cols[x];
                for (size_t ch = 0; ch < 3; ++ch) {
                    auto at = [&](size_t yy, size_t xx) {
                        return static_cast<float>(frame[(yy * src_w + xx) * 3 + ch]);
                    };
                    const float value = r.l0 * (c.l0 * at(r.i0, c.i0) + c.l1 * at(r.i0, c.i1)) +
                                        r.l1 * (c.l0 * at(r.i1, c.i0) + c.l1 * at(r.i1, c.i1));
                    out[ch * plane + (f * height + y) * width + x] = value / 127.5f - 1.0f;
                }
            }
        }
    }
    return pixels;
}

/// Resizes a u8 image [1, H, W, 3] exactly like PIL's Image.resize, with GenAI's port of Pillow's fixed-point
/// resampling: diffusers prepares images with PIL, and a 1-level difference already changes what a model generates
inline ov::Tensor pillow_resize(const ov::Tensor& image, size_t height, size_t width, bool bicubic) {
    clip_image_u8 resized;
    const clip_image_u8 source = tensor_to_clip_image_u8(image);
    if (bicubic) {
        bicubic_resize(source, resized, static_cast<int>(width), static_cast<int>(height));
    } else {
        bilinear_resize(source, resized, static_cast<int>(width), static_cast<int>(height));
    }
    ov::Tensor out(ov::element::u8, {1, height, width, 3});
    std::copy(resized.buf.begin(), resized.buf.end(), out.data<uint8_t>());
    return out;
}

/// u8 image already resized to [1, H, W, 3] -> f32 [1, 3, 1, H, W] in [-1, 1], like diffusers' VideoProcessor
inline ov::Tensor image_to_pixels(const ov::Tensor& image) {
    const ov::Shape shape = image.get_shape();
    const size_t height = shape[1], width = shape[2];
    ov::Tensor pixels(ov::element::f32, {1, 3, 1, height, width});
    float* out = pixels.data<float>();
    const uint8_t* in = image.data<const uint8_t>();
    for (size_t i = 0; i < height * width; ++i) {
        for (size_t ch = 0; ch < 3; ++ch) {
            out[ch * height * width + i] = static_cast<float>(in[i * 3 + ch]) / 255.0f * 2.0f - 1.0f;
        }
    }
    return pixels;
}

/**
 * Positions [batch, 3, tokens, 2] of a condition appended as keyframe tokens (diffusers' _prepare_keyframe_coords):
 * pixel-space patch bounds without the causal first-frame fix, offset to 'pixel_frame_index', time in seconds. A
 * single-frame keyframe occupies one pixel frame.
 */
inline ov::Tensor keyframe_coords(size_t batch_size,
                                  size_t latent_frames,
                                  size_t latent_height,
                                  size_t latent_width,
                                  size_t pixel_frame_index,
                                  size_t num_pixel_frames,
                                  float frame_rate,
                                  const std::vector<int64_t>& scale_factors) {
    const size_t tokens = latent_frames * latent_height * latent_width;
    ov::Tensor coords(ov::element::f32, {batch_size, 3, tokens, 2});
    float* data = coords.data<float>();
    size_t token = 0;
    for (size_t f = 0; f < latent_frames; ++f) {
        for (size_t h = 0; h < latent_height; ++h) {
            for (size_t w = 0; w < latent_width; ++w, ++token) {
                float t0 = static_cast<float>(f * scale_factors[0] + pixel_frame_index);
                float t1 = static_cast<float>((f + 1) * scale_factors[0] + pixel_frame_index);
                if (num_pixel_frames == 1) {
                    t1 = t0 + 1.0f;
                }
                data[(0 * tokens + token) * 2] = t0 / frame_rate;
                data[(0 * tokens + token) * 2 + 1] = t1 / frame_rate;
                data[(1 * tokens + token) * 2] = static_cast<float>(h * scale_factors[1]);
                data[(1 * tokens + token) * 2 + 1] = static_cast<float>((h + 1) * scale_factors[1]);
                data[(2 * tokens + token) * 2] = static_cast<float>(w * scale_factors[2]);
                data[(2 * tokens + token) * 2 + 1] = static_cast<float>((w + 1) * scale_factors[2]);
            }
        }
    }
    const size_t batch_stride = 3 * tokens * 2;
    for (size_t b = 1; b < batch_size; ++b) {
        std::memcpy(data + b * batch_stride, data, batch_stride * sizeof(float));
    }
    return coords;
}

/// Concatenates [B, 3, N, 2] coordinates along the token axis
inline ov::Tensor concat_coords(const ov::Tensor& first, const ov::Tensor& second) {
    const ov::Shape a = first.get_shape(), b = second.get_shape();
    const size_t batch = a[0], na = a[2], nb = b[2];
    ov::Tensor out(ov::element::f32, {batch, 3, na + nb, 2});
    float* dst = out.data<float>();
    const float* pa = first.data<const float>();
    const float* pb = second.data<const float>();
    for (size_t i = 0; i < batch * 3; ++i) {
        std::memcpy(dst, pa + i * na * 2, na * 2 * sizeof(float));
        dst += na * 2;
        std::memcpy(dst, pb + i * nb * 2, nb * 2 * sizeof(float));
        dst += nb * 2;
    }
    return out;
}

}  // namespace ov::genai::ltx2_conditioning
