// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "blend_utils.hpp"
#include <algorithm>
#include <vector>
#include <cstdint>
#include "openvino/openvino.hpp"

namespace ov {
namespace genai {
namespace module {
namespace blend_utils {

// ============================================================================
// 4D Blend Functions for Images [N, C, H, W]
// ============================================================================

void blend_v_4d(ov::Tensor& a, ov::Tensor& b, size_t blend_extent) {
    // Blend along height dimension for 4D tensor [N, C, H, W]
    auto shape_a = a.get_shape();
    auto shape_b = b.get_shape();

    blend_extent = std::min({shape_a[2], shape_b[2], blend_extent});
    if (blend_extent == 0) return;

    const size_t N = shape_b[0];
    const size_t C = shape_b[1];
    const size_t H = shape_b[2];
    const size_t W = shape_b[3];
    const size_t H_a = shape_a[2];

    float* ptr_a = a.data<float>();
    float* ptr_b = b.data<float>();

    const size_t channel_stride_a = H_a * W;
    const size_t channel_stride_b = H * W;
    const size_t batch_stride_a = C * channel_stride_a;
    const size_t batch_stride_b = C * channel_stride_b;

    // Pre-compute blend weights
    const float inv_blend = 1.0f / static_cast<float>(blend_extent);

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t y = 0; y < blend_extent; ++y) {
                const float weight_b = static_cast<float>(y) * inv_blend;
                const float weight_a = 1.0f - weight_b;

                // a[:, :, -blend_extent + y, :]
                const size_t idx_a = n * batch_stride_a + c * channel_stride_a + (H_a - blend_extent + y) * W;
                // b[:, :, y, :]
                const size_t idx_b = n * batch_stride_b + c * channel_stride_b + y * W;

                for (size_t x = 0; x < W; ++x) {
                    ptr_b[idx_b + x] = ptr_a[idx_a + x] * weight_a + ptr_b[idx_b + x] * weight_b;
                }
            }
        }
    }
}

void blend_h_4d(ov::Tensor& a, ov::Tensor& b, size_t blend_extent) {
    // Blend along width dimension for 4D tensor [N, C, H, W]
    auto shape_a = a.get_shape();
    auto shape_b = b.get_shape();

    blend_extent = std::min({shape_a[3], shape_b[3], blend_extent});
    if (blend_extent == 0) return;

    const size_t N = shape_b[0];
    const size_t C = shape_b[1];
    const size_t H = shape_b[2];
    const size_t W = shape_b[3];
    const size_t W_a = shape_a[3];

    float* ptr_a = a.data<float>();
    float* ptr_b = b.data<float>();

    const size_t channel_stride_a = H * W_a;
    const size_t batch_stride_a = C * channel_stride_a;
    const size_t channel_stride_b = H * W;
    const size_t batch_stride_b = C * channel_stride_b;

    // Pre-compute blend weights
    const float inv_blend = 1.0f / static_cast<float>(blend_extent);

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t y = 0; y < H; ++y) {
                // a: take last blend_extent columns
                const size_t row_offset_a = n * batch_stride_a + c * channel_stride_a + y * W_a + (W_a - blend_extent);
                // b: take first blend_extent columns
                const size_t row_offset_b = n * batch_stride_b + c * channel_stride_b + y * W;

                for (size_t x = 0; x < blend_extent; ++x) {
                    const float weight_b = static_cast<float>(x) * inv_blend;
                    const float weight_a = 1.0f - weight_b;

                    ptr_b[row_offset_b + x] = ptr_a[row_offset_a + x] * weight_a + ptr_b[row_offset_b + x] * weight_b;
                }
            }
        }
    }
}

// ============================================================================
// 5D Blend Functions for Videos
// Supports both [B, T, H, W, C] (u8) and [B, C, T, H, W] (f32)
// ============================================================================

void blend_v_5d(ov::Tensor& a, ov::Tensor& b, size_t blend_extent) {
    // Blend along height dimension for 5D tensor
    // Output format after postprocess: [B, T, H, W, C] (u8)
    // Raw output format: [B, C, T, H, W] (f32)
    // OPTIMIZED: Pre-computed weights + row-wise processing + loop unrolling
    auto shape_a = a.get_shape();
    auto shape_b = b.get_shape();

    blend_extent = std::min({shape_a[2], shape_b[2], blend_extent});  // H dimension
    if (blend_extent == 0) return;

    // Pre-compute blend weights (avoid division in inner loop)
    std::vector<float> weight_b_arr(blend_extent);
    std::vector<float> weight_a_arr(blend_extent);
    const float inv_blend = 1.0f / static_cast<float>(blend_extent);
    for (size_t y = 0; y < blend_extent; ++y) {
        weight_b_arr[y] = static_cast<float>(y) * inv_blend;
        weight_a_arr[y] = 1.0f - weight_b_arr[y];
    }

    if (a.get_element_type() == ov::element::u8) {
        // Postprocessed: [B, T, H, W, C]
        const size_t B = shape_b[0], T = shape_b[1], H = shape_b[2], W = shape_b[3], C = shape_b[4];
        const size_t H_a = shape_a[2];
        const size_t row_stride = W * C;

        uint8_t* __restrict ptr_a = a.data<uint8_t>();
        uint8_t* __restrict ptr_b = b.data<uint8_t>();

        for (size_t b_idx = 0; b_idx < B; ++b_idx) {
            for (size_t t = 0; t < T; ++t) {
                const size_t frame_offset_a = (b_idx * T + t) * H_a;
                const size_t frame_offset_b = (b_idx * T + t) * H;

                for (size_t y = 0; y < blend_extent; ++y) {
                    const float wa = weight_a_arr[y];
                    const float wb = weight_b_arr[y];

                    const uint8_t* __restrict src = ptr_a + (frame_offset_a + H_a - blend_extent + y) * row_stride;
                    uint8_t* __restrict dst = ptr_b + (frame_offset_b + y) * row_stride;

                    // Process 4 elements at a time (loop unrolling)
                    size_t i = 0;
                    const size_t unroll_end = row_stride & ~3ULL;  // Round down to multiple of 4
                    for (; i < unroll_end; i += 4) {
                        float v0 = src[i]   * wa + dst[i]   * wb;
                        float v1 = src[i+1] * wa + dst[i+1] * wb;
                        float v2 = src[i+2] * wa + dst[i+2] * wb;
                        float v3 = src[i+3] * wa + dst[i+3] * wb;
                        dst[i]   = static_cast<uint8_t>(v0 < 0.f ? 0.f : (v0 > 255.f ? 255.f : v0));
                        dst[i+1] = static_cast<uint8_t>(v1 < 0.f ? 0.f : (v1 > 255.f ? 255.f : v1));
                        dst[i+2] = static_cast<uint8_t>(v2 < 0.f ? 0.f : (v2 > 255.f ? 255.f : v2));
                        dst[i+3] = static_cast<uint8_t>(v3 < 0.f ? 0.f : (v3 > 255.f ? 255.f : v3));
                    }
                    // Handle remainder
                    for (; i < row_stride; ++i) {
                        float v = src[i] * wa + dst[i] * wb;
                        dst[i] = static_cast<uint8_t>(v < 0.f ? 0.f : (v > 255.f ? 255.f : v));
                    }
                }
            }
        }
    } else {
        // Raw: [B, C, T, H, W]
        const size_t B = shape_b[0], C_dim = shape_b[1], T = shape_b[2], H = shape_b[3], W = shape_b[4];
        const size_t H_a = shape_a[3];

        float* __restrict ptr_a = a.data<float>();
        float* __restrict ptr_b = b.data<float>();

        for (size_t b_idx = 0; b_idx < B; ++b_idx) {
            for (size_t c = 0; c < C_dim; ++c) {
                for (size_t t = 0; t < T; ++t) {
                    const size_t plane_a = ((b_idx * C_dim + c) * T + t) * H_a * W;
                    const size_t plane_b = ((b_idx * C_dim + c) * T + t) * H * W;

                    for (size_t y = 0; y < blend_extent; ++y) {
                        const float wa = weight_a_arr[y];
                        const float wb = weight_b_arr[y];

                        const float* __restrict src = ptr_a + plane_a + (H_a - blend_extent + y) * W;
                        float* __restrict dst = ptr_b + plane_b + y * W;

                        // Process 4 elements at a time
                        size_t w = 0;
                        const size_t unroll_end = W & ~3ULL;
                        for (; w < unroll_end; w += 4) {
                            dst[w]   = src[w]   * wa + dst[w]   * wb;
                            dst[w+1] = src[w+1] * wa + dst[w+1] * wb;
                            dst[w+2] = src[w+2] * wa + dst[w+2] * wb;
                            dst[w+3] = src[w+3] * wa + dst[w+3] * wb;
                        }
                        for (; w < W; ++w) {
                            dst[w] = src[w] * wa + dst[w] * wb;
                        }
                    }
                }
            }
        }
    }
}

void blend_h_5d(ov::Tensor& a, ov::Tensor& b, size_t blend_extent) {
    // Blend along width dimension for 5D tensor
    // OPTIMIZED: Pre-computed weights + restrict pointers + loop unrolling
    auto shape_a = a.get_shape();
    auto shape_b = b.get_shape();

    blend_extent = std::min({shape_a[3], shape_b[3], blend_extent});  // W dimension
    if (blend_extent == 0) return;

    // Pre-compute blend weights
    std::vector<float> weight_b_arr(blend_extent);
    std::vector<float> weight_a_arr(blend_extent);
    const float inv_blend = 1.0f / static_cast<float>(blend_extent);
    for (size_t x = 0; x < blend_extent; ++x) {
        weight_b_arr[x] = static_cast<float>(x) * inv_blend;
        weight_a_arr[x] = 1.0f - weight_b_arr[x];
    }

    if (a.get_element_type() == ov::element::u8) {
        // Postprocessed: [B, T, H, W, C]
        const size_t B = shape_b[0], T = shape_b[1], H = shape_b[2], W = shape_b[3], C = shape_b[4];
        const size_t W_a = shape_a[3];

        uint8_t* __restrict ptr_a = a.data<uint8_t>();
        uint8_t* __restrict ptr_b = b.data<uint8_t>();

        for (size_t b_idx = 0; b_idx < B; ++b_idx) {
            for (size_t t = 0; t < T; ++t) {
                for (size_t h = 0; h < H; ++h) {
                    const size_t row_base_a = ((b_idx * T + t) * H + h) * W_a * C;
                    const size_t row_base_b = ((b_idx * T + t) * H + h) * W * C;

                    // For horizontal blend, process all blend positions
                    // Note: blend_extent is typically small (64 pixels), C=3
                    // So blend_extent * C elements per row
                    for (size_t x = 0; x < blend_extent; ++x) {
                        const float wa = weight_a_arr[x];
                        const float wb = weight_b_arr[x];

                        const uint8_t* __restrict src = ptr_a + row_base_a + (W_a - blend_extent + x) * C;
                        uint8_t* __restrict dst = ptr_b + row_base_b + x * C;

                        // C is typically 3 (RGB), unroll for common case
                        if (C == 3) {
                            float v0 = src[0] * wa + dst[0] * wb;
                            float v1 = src[1] * wa + dst[1] * wb;
                            float v2 = src[2] * wa + dst[2] * wb;
                            dst[0] = static_cast<uint8_t>(v0 < 0.f ? 0.f : (v0 > 255.f ? 255.f : v0));
                            dst[1] = static_cast<uint8_t>(v1 < 0.f ? 0.f : (v1 > 255.f ? 255.f : v1));
                            dst[2] = static_cast<uint8_t>(v2 < 0.f ? 0.f : (v2 > 255.f ? 255.f : v2));
                        } else {
                            for (size_t c = 0; c < C; ++c) {
                                float v = src[c] * wa + dst[c] * wb;
                                dst[c] = static_cast<uint8_t>(v < 0.f ? 0.f : (v > 255.f ? 255.f : v));
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Raw: [B, C, T, H, W]
        const size_t B = shape_b[0], C_dim = shape_b[1], T = shape_b[2], H = shape_b[3], W = shape_b[4];
        const size_t W_a = shape_a[4];

        float* __restrict ptr_a = a.data<float>();
        float* __restrict ptr_b = b.data<float>();

        for (size_t b_idx = 0; b_idx < B; ++b_idx) {
            for (size_t c = 0; c < C_dim; ++c) {
                for (size_t t = 0; t < T; ++t) {
                    for (size_t h = 0; h < H; ++h) {
                        const float* __restrict src_row = ptr_a + ((b_idx * C_dim + c) * T + t) * H * W_a + h * W_a + (W_a - blend_extent);
                        float* __restrict dst_row = ptr_b + ((b_idx * C_dim + c) * T + t) * H * W + h * W;

                        // blend_extent is typically 8 (latent) or 64 (sample)
                        // Unroll by 4 for typical cases
                        size_t x = 0;
                        const size_t unroll_end = blend_extent & ~3ULL;
                        for (; x < unroll_end; x += 4) {
                            dst_row[x]   = src_row[x]   * weight_a_arr[x]   + dst_row[x]   * weight_b_arr[x];
                            dst_row[x+1] = src_row[x+1] * weight_a_arr[x+1] + dst_row[x+1] * weight_b_arr[x+1];
                            dst_row[x+2] = src_row[x+2] * weight_a_arr[x+2] + dst_row[x+2] * weight_b_arr[x+2];
                            dst_row[x+3] = src_row[x+3] * weight_a_arr[x+3] + dst_row[x+3] * weight_b_arr[x+3];
                        }
                        for (; x < blend_extent; ++x) {
                            dst_row[x] = src_row[x] * weight_a_arr[x] + dst_row[x] * weight_b_arr[x];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Auto-dispatch Functions
// ============================================================================

void blend_v(ov::Tensor& a, ov::Tensor& b, size_t blend_extent) {
    size_t ndim = a.get_shape().size();

    if (ndim == 4) {
        blend_v_4d(a, b, blend_extent);
    } else if (ndim == 5) {
        blend_v_5d(a, b, blend_extent);
    } else {
        OPENVINO_THROW("blend_v: Unsupported tensor rank " + std::to_string(ndim) + ", expected 4 or 5");
    }
}

void blend_h(ov::Tensor& a, ov::Tensor& b, size_t blend_extent) {
    size_t ndim = a.get_shape().size();

    if (ndim == 4) {
        blend_h_4d(a, b, blend_extent);
    } else if (ndim == 5) {
        blend_h_5d(a, b, blend_extent);
    } else {
        OPENVINO_THROW("blend_h: Unsupported tensor rank " + std::to_string(ndim) + ", expected 4 or 5");
    }
}

}  // namespace blend_utils
}  // namespace module
}  // namespace genai
}  // namespace ov
