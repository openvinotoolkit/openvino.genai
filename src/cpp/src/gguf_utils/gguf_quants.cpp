// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <openvino/core/parallel.hpp>
#include <sstream>

#include "gguf_utils/gguf.hpp"

using namespace std;

// Read a little-endian IEEE half-precision value from a (possibly unaligned) byte
// pointer. Using memcpy avoids the strict-aliasing/alignment undefined behavior of
// dereferencing a reinterpreted `uint16_t*` and keeps the readers portable.
static inline ov::float16 read_f16(const uint8_t* p) {
    uint16_t bits;
    std::memcpy(&bits, p, sizeof(bits));
    return ov::float16::from_bits(bits);
}

void unpack_32_4(uint8_t* data, uint8_t* dst) {
    std::fill_n(dst, 16, 0);
    for (int j = 0; j < 16; ++j) {
        uint8_t x = (data[j] & 0x0F);
        uint8_t y = (data[j] >> 4);
        if (j % 2 != 0) {
            x <<= 4;
            y <<= 4;
        }
        dst[j / 2] |= x;
        dst[8 + j / 2] |= y;  // Last 16 weights are in the higher bits
    }
}

// Extracts (weight, scales, biases) from Q4_0 tensors.
// Data layout is: |16 bit scale|32 x 4bit weights|.
void extract_q4_0_data(const gguf_tensor& tensor,
                       ov::Tensor& weights_arr,
                       ov::Tensor& scales_arr,
                       ov::Tensor& biases_arr) {
    const uint64_t bytes_per_block = 18;  // 2 bytes scale, 32x0.5 byte weights
    auto data = static_cast<uint8_t*>(tensor.weights_data);
    auto weights = static_cast<uint8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto biases = biases_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
        scales[i] = read_f16(data + i * bytes_per_block);
        biases[i] = ov::float16(-8.f * static_cast<float>(scales[i]));
        unpack_32_4(data + i * bytes_per_block + 2, weights + i * 16);
    });
}

// Extracts (weight, scales, biases) from Q4_1 tensors.
// Data layout is: |16 bit scale|16 bit bias|32 x 4bit weights|.
void extract_q4_1_data(const gguf_tensor& tensor,
                       ov::Tensor& weights_arr,
                       ov::Tensor& scales_arr,
                       ov::Tensor& biases_arr) {
    const uint64_t bytes_per_block = 20;  // 2 bytes scale, 2 bytes bias, 32x0.5 byte weights
    auto data = static_cast<uint8_t*>(tensor.weights_data);
    auto weights = static_cast<uint8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto biases = biases_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
        scales[i] = read_f16(data + i * bytes_per_block);
        biases[i] = read_f16(data + i * bytes_per_block + 2);
        unpack_32_4(data + i * bytes_per_block + 4, weights + i * 16);
    });
}

// Extracts (weight, scales, biases) from Q8_0 tensors.
// Data layout is: |16 bit scale|32 x 8bit weights|.
// Emitted at group size 16 (the single 32-weight block scale duplicated across its two
// halves) so the layout matches the other u8 quants (Q5_0/Q5_1/Q5_K/Q6_K). Keeping all
// u8 quant types at one group size lets the GPU plugin's FullyConnectedHorizontalFusion
// concat q/k/v scales when a model mixes them across those projections.
void extract_q8_0_data(const gguf_tensor& tensor,
                       ov::Tensor& weights_arr,
                       ov::Tensor& scales_arr,
                       ov::Tensor& biases_arr) {
    const uint64_t weights_per_block = 32;
    const uint64_t bytes_per_block = 34;  // 2 bytes scale, 32x1 byte weights
    const uint64_t n_block = tensor.bsize / bytes_per_block;
    auto data = static_cast<uint8_t*>(tensor.weights_data);
    auto weights = static_cast<uint8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto biases = biases_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    // Two scale/bias entries per 32-weight block (group 16). Guards against the
    // scales tensor being sized for a different group (would overflow below).
    OPENVINO_ASSERT(scales_arr.get_size() == n_block * 2,
                    "[load_gguf] Q8_0 scales size mismatch (expected group 16)");
    for (int64_t i = 0; i < static_cast<int64_t>(n_block); i++) {
        uint8_t* block_data = data + i * bytes_per_block;
        ov::float16 sc = read_f16(block_data);
        ov::float16 bs(-128.f * static_cast<float>(sc));
        scales[i * 2] = sc;
        scales[i * 2 + 1] = sc;
        biases[i * 2] = bs;
        biases[i * 2 + 1] = bs;
        for (int64_t j = 0; j < static_cast<int64_t>(weights_per_block); ++j) {
            uint8_t x = block_data[j + 2];  // j+2 to skip the scale bytes.
            // Original data is in int8_t, so we add a bias of -128 and invert the
            // first bit.
            x ^= 1 << 7;
            weights[i * weights_per_block + j] = x;
        }
    }
}

void unpack_256_4(const uint8_t* data, uint8_t* dst) {
    // Initialize the output array with zeros
    std::fill_n(dst, 128, 0);

    for (size_t i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            uint8_t x = (data[i * 32 + j] & 0x0F);
            uint8_t y = (data[i * 32 + j] >> 4);
            if (j % 2 != 0) {
                x <<= 4;
                y <<= 4;
            }
            dst[i * 32 + j / 2] |= x;
            dst[i * 32 + 16 + j / 2] |= y;  // Last 16 weights are in the higher bits
        }
    }
}

void extract_q4_k_data(const gguf_tensor& tensor,
                       ov::Tensor& weights_arr,
                       ov::Tensor& scales_arr,
                       ov::Tensor& biases_arr) {
    const uint64_t bytes_per_block = 2 + 2 + 12 + 128;
    const uint64_t n_super_block = tensor.bsize / bytes_per_block;
    auto data = static_cast<uint8_t*>(tensor.weights_data);
    auto weights = static_cast<uint8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto biases = biases_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    ov::parallel_for(n_super_block, [&](size_t i) {
        uint8_t* block_data = data + i * bytes_per_block;

        // Extract scale factors and offsets
        float scale_scales = static_cast<float>(read_f16(block_data));
        float scale_biases = static_cast<float>(read_f16(block_data + 2));

        // Extract qs1 and qs2
        uint8_t* qs1 = block_data + 4;
        uint8_t* qs2 = block_data + 16;

        scales[i * 8] = ov::float16(scale_scales * static_cast<float>((*(qs1) & 0b111111)));
        scales[i * 8 + 1] = ov::float16(scale_scales * static_cast<float>((*(qs1 + 1) & 0b111111)));
        scales[i * 8 + 2] = ov::float16(scale_scales * static_cast<float>((*(qs1 + 2) & 0b111111)));
        scales[i * 8 + 3] = ov::float16(scale_scales * static_cast<float>((*(qs1 + 3) & 0b111111)));
        scales[i * 8 + 4] =
            ov::float16(scale_scales * static_cast<float>((*(qs1 + 8) & 0b00001111) | ((*(qs1) >> 6) << 4)));
        scales[i * 8 + 5] =
            ov::float16(scale_scales * static_cast<float>((*(qs1 + 9) & 0b00001111) | ((*(qs1 + 1) >> 6) << 4)));
        scales[i * 8 + 6] =
            ov::float16(scale_scales * static_cast<float>((*(qs1 + 10) & 0b00001111) | ((*(qs1 + 2) >> 6) << 4)));
        scales[i * 8 + 7] =
            ov::float16(scale_scales * static_cast<float>((*(qs1 + 11) & 0b00001111) | ((*(qs1 + 3) >> 6) << 4)));

        biases[i * 8] = ov::float16(-1.f * scale_biases * static_cast<float>((*(qs1 + 4) & 0b111111)));
        biases[i * 8 + 1] = ov::float16(-1.f * scale_biases * static_cast<float>((*(qs1 + 5) & 0b111111)));
        biases[i * 8 + 2] = ov::float16(-1.f * scale_biases * static_cast<float>((*(qs1 + 6) & 0b111111)));
        biases[i * 8 + 3] = ov::float16(-1.f * scale_biases * static_cast<float>((*(qs1 + 7) & 0b111111)));
        biases[i * 8 + 4] =
            ov::float16(-1.f * scale_biases * static_cast<float>((*(qs1 + 8) >> 4) | ((*(qs1 + 4) >> 6) << 4)));
        biases[i * 8 + 5] =
            ov::float16(-1.f * scale_biases * static_cast<float>((*(qs1 + 9) >> 4) | ((*(qs1 + 5) >> 6) << 4)));
        biases[i * 8 + 6] =
            ov::float16(-1.f * scale_biases * static_cast<float>((*(qs1 + 10) >> 4) | ((*(qs1 + 6) >> 6) << 4)));
        biases[i * 8 + 7] =
            ov::float16(-1.f * scale_biases * static_cast<float>((*(qs1 + 11) >> 4) | ((*(qs1 + 7) >> 6) << 4)));
        unpack_256_4(block_data + 16, weights + i * 128);
    });
}

void extract_q6_k_data(const gguf_tensor& tensor,
                       ov::Tensor& weights_arr,
                       ov::Tensor& scales_arr,
                       ov::Tensor& biases_arr) {
    const uint64_t bytes_per_block = 128 + 64 + 16 + 2;
    const uint64_t n_super_block = tensor.bsize / bytes_per_block;
    auto data = static_cast<uint8_t*>(tensor.weights_data);
    auto weights = static_cast<uint8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto biases = biases_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    // std::string name(tensor.name, tensor.namelen);
    for (int64_t i = 0; i < n_super_block; i++) {
        uint8_t* block_data = data + i * bytes_per_block;

        float scale_factor =
            static_cast<float>(read_f16(block_data + 208));  // (128+64+16)/2

        for (size_t j = 0; j < 16; j++) {
            scales[j + i * 16] =
                ov::float16(scale_factor * static_cast<float>(*((int8_t*)(block_data + 128 + 64 + j))));
            biases[j + i * 16] = ov::float16(-32.f * static_cast<float>(scales[j + i * 16]));
        }

        // Extract ql and qh
        uint8_t* ql = block_data;
        uint8_t* qh = block_data + 128;

        // Extract weights
        for (int64_t j = 0; j < 32; ++j) {
            weights[i * 256 + j] = (ql[j] & 0xF) | (((qh[j] >> 0) & 3) << 4);
            weights[i * 256 + j + 32] = (ql[32 + j] & 0xF) | (((qh[j] >> 2) & 3) << 4);
            weights[i * 256 + j + 64] = (ql[j] >> 4) | (((qh[j] >> 4) & 3) << 4);
            weights[i * 256 + j + 96] = (ql[32 + j] >> 4) | (((qh[j] >> 6) & 3) << 4);
            weights[i * 256 + j + 128] = (ql[64 + j] & 0xF) | (((qh[32 + j] >> 0) & 3) << 4);
            weights[i * 256 + j + 160] = (ql[96 + j] & 0xF) | (((qh[32 + j] >> 2) & 3) << 4);
            weights[i * 256 + j + 192] = (ql[64 + j] >> 4) | (((qh[32 + j] >> 4) & 3) << 4);
            weights[i * 256 + j + 224] = (ql[96 + j] >> 4) | (((qh[32 + j] >> 6) & 3) << 4);
        }
    }
}

// Extracts (weight, scales, biases) from Q5_K tensors.
// Super-block layout (256 weights, 176 bytes):
//   |f16 d|f16 dmin|12 x 6bit packed scales/mins|32 x qh (high bits)|128 x qs (low nibbles)|
// The 6-bit scale/min packing is identical to Q4_K; only the weights differ:
// each weight is 5-bit = (low nibble from qs) | (high bit from qh).
//
// Q5_K has 8 scale/min sub-blocks of 32 weights each. We deliberately emit the
// scales/biases at group size 16 (16 entries per super-block, each of the 8
// sub-block values duplicated across its two halves). This is numerically
// identical -- all 32 weights of a sub-block share one scale -- but it makes
// the Q5_K weight layout match Q6_K (also group 16). Without this, a model that
// mixes Q5_K (q/k) and Q6_K (v) projections -- both u8 -- produces sibling
// FullyConnected ops with different group counts, and the GPU plugin's
// FullyConnectedHorizontalFusion fails to concat their scales. Matching the
// group size keeps q/k/v fusion-compatible on GPU.
void extract_q5_k_data(const gguf_tensor& tensor,
                       ov::Tensor& weights_arr,
                       ov::Tensor& scales_arr,
                       ov::Tensor& biases_arr) {
    const uint64_t bytes_per_block = 2 + 2 + 12 + 32 + 128;  // 176
    const uint64_t n_super_block = tensor.bsize / bytes_per_block;
    auto data = static_cast<uint8_t*>(tensor.weights_data);
    auto weights = static_cast<uint8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto biases = biases_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    // 16 scale/bias entries per 256-weight super-block (group 16). Guards against the
    // scales tensor being sized for a different group (would overflow below).
    OPENVINO_ASSERT(scales_arr.get_size() == n_super_block * 16,
                    "[load_gguf] Q5_K scales size mismatch (expected group 16)");
    ov::parallel_for(n_super_block, [&](size_t i) {
        uint8_t* block_data = data + i * bytes_per_block;

        // Extract scale factors and offsets (identical packing to Q4_K).
        float scale_scales = static_cast<float>(read_f16(block_data));
        float scale_biases = static_cast<float>(read_f16(block_data + 2));

        // 12-byte packed 6-bit scales/mins start right after d and dmin.
        uint8_t* qs1 = block_data + 4;

        // Decode the 8 sub-block (32-weight) scales and mins.
        float sub_scale[8], sub_bias[8];
        sub_scale[0] = scale_scales * static_cast<float>((*(qs1) & 0b111111));
        sub_scale[1] = scale_scales * static_cast<float>((*(qs1 + 1) & 0b111111));
        sub_scale[2] = scale_scales * static_cast<float>((*(qs1 + 2) & 0b111111));
        sub_scale[3] = scale_scales * static_cast<float>((*(qs1 + 3) & 0b111111));
        sub_scale[4] = scale_scales * static_cast<float>((*(qs1 + 8) & 0b00001111) | ((*(qs1) >> 6) << 4));
        sub_scale[5] = scale_scales * static_cast<float>((*(qs1 + 9) & 0b00001111) | ((*(qs1 + 1) >> 6) << 4));
        sub_scale[6] = scale_scales * static_cast<float>((*(qs1 + 10) & 0b00001111) | ((*(qs1 + 2) >> 6) << 4));
        sub_scale[7] = scale_scales * static_cast<float>((*(qs1 + 11) & 0b00001111) | ((*(qs1 + 3) >> 6) << 4));

        sub_bias[0] = -1.f * scale_biases * static_cast<float>((*(qs1 + 4) & 0b111111));
        sub_bias[1] = -1.f * scale_biases * static_cast<float>((*(qs1 + 5) & 0b111111));
        sub_bias[2] = -1.f * scale_biases * static_cast<float>((*(qs1 + 6) & 0b111111));
        sub_bias[3] = -1.f * scale_biases * static_cast<float>((*(qs1 + 7) & 0b111111));
        sub_bias[4] = -1.f * scale_biases * static_cast<float>((*(qs1 + 8) >> 4) | ((*(qs1 + 4) >> 6) << 4));
        sub_bias[5] = -1.f * scale_biases * static_cast<float>((*(qs1 + 9) >> 4) | ((*(qs1 + 5) >> 6) << 4));
        sub_bias[6] = -1.f * scale_biases * static_cast<float>((*(qs1 + 10) >> 4) | ((*(qs1 + 6) >> 6) << 4));
        sub_bias[7] = -1.f * scale_biases * static_cast<float>((*(qs1 + 11) >> 4) | ((*(qs1 + 7) >> 6) << 4));

        // Emit at group size 16: duplicate each 32-weight sub-block value across
        // its two 16-weight halves (16 entries per super-block).
        for (int s = 0; s < 8; ++s) {
            scales[i * 16 + 2 * s] = ov::float16(sub_scale[s]);
            scales[i * 16 + 2 * s + 1] = ov::float16(sub_scale[s]);
            biases[i * 16 + 2 * s] = ov::float16(sub_bias[s]);
            biases[i * 16 + 2 * s + 1] = ov::float16(sub_bias[s]);
        }

        // Weights: 5-bit, low nibble from qs, high bit from qh. Sequential y-order,
        // so weight k belongs to sub-block k/32 (i.e. group-16 indices 2*(k/32) and
        // 2*(k/32)+1, which carry the same scale/bias).
        uint8_t* qh = block_data + 16;   // 32 bytes of high bits (one bit per weight)
        uint8_t* ql = block_data + 48;   // 128 bytes of low nibbles
        uint8_t u1 = 1, u2 = 2;
        for (int jb = 0; jb < 256; jb += 64) {
            for (int l = 0; l < 32; ++l) {
                weights[i * 256 + jb + l] = (ql[l] & 0x0F) | ((qh[l] & u1) ? 16 : 0);
                weights[i * 256 + jb + 32 + l] = (ql[l] >> 4) | ((qh[l] & u2) ? 16 : 0);
            }
            ql += 32;
            u1 <<= 2;
            u2 <<= 2;
        }
    });
}

// Extracts (weight, scales, biases) from Q5_0 tensors.
// Block layout (32 weights, 22 bytes): |f16 d|4 x qh (high bits)|16 x qs (low nibbles)|
// Each weight is 5-bit = (qs nibble) | (qh bit << 4), dequantized as (w - 16) * d.
// Emitted at group size 16 (the single 32-weight block scale duplicated across its two
// halves) so the layout matches Q5_K/Q6_K and stays GPU horizontal-fusion compatible.
void extract_q5_0_data(const gguf_tensor& tensor,
                       ov::Tensor& weights_arr,
                       ov::Tensor& scales_arr,
                       ov::Tensor& biases_arr) {
    const uint64_t bytes_per_block = 2 + 4 + 16;  // 22
    const uint64_t n_block = tensor.bsize / bytes_per_block;
    auto data = static_cast<uint8_t*>(tensor.weights_data);
    auto weights = static_cast<uint8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto biases = biases_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    // Two scale/bias entries per 32-weight block (group 16). Guards against the
    // scales tensor being sized for a different group (would overflow below).
    OPENVINO_ASSERT(scales_arr.get_size() == n_block * 2,
                    "[load_gguf] Q5_0 scales size mismatch (expected group 16)");
    ov::parallel_for(n_block, [&](size_t i) {
        uint8_t* block_data = data + i * bytes_per_block;
        float d = static_cast<float>(read_f16(block_data));

        ov::float16 sc(d);
        ov::float16 bs(-16.f * d);  // zero point 16 -> dequant is (w - 16) * d
        scales[i * 2] = sc;
        scales[i * 2 + 1] = sc;
        biases[i * 2] = bs;
        biases[i * 2 + 1] = bs;

        uint32_t qh;
        std::memcpy(&qh, block_data + 2, sizeof(qh));
        uint8_t* qs = block_data + 6;
        for (int j = 0; j < 16; ++j) {
            uint8_t xh0 = static_cast<uint8_t>(((qh >> j) & 1) << 4);
            uint8_t xh1 = static_cast<uint8_t>(((qh >> (j + 16)) & 1) << 4);
            weights[i * 32 + j] = (qs[j] & 0x0F) | xh0;
            weights[i * 32 + j + 16] = (qs[j] >> 4) | xh1;
        }
    });
}

// Extracts (weight, scales, biases) from Q5_1 tensors.
// Block layout (32 weights, 24 bytes): |f16 d|f16 m|4 x qh|16 x qs|
// Each weight is 5-bit = (qs nibble) | (qh bit << 4), dequantized as w * d + m.
// Emitted at group size 16, as for Q5_0.
void extract_q5_1_data(const gguf_tensor& tensor,
                       ov::Tensor& weights_arr,
                       ov::Tensor& scales_arr,
                       ov::Tensor& biases_arr) {
    const uint64_t bytes_per_block = 2 + 2 + 4 + 16;  // 24
    const uint64_t n_block = tensor.bsize / bytes_per_block;
    auto data = static_cast<uint8_t*>(tensor.weights_data);
    auto weights = static_cast<uint8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto biases = biases_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    // Two scale/bias entries per 32-weight block (group 16). Guards against the
    // scales tensor being sized for a different group (would overflow below).
    OPENVINO_ASSERT(scales_arr.get_size() == n_block * 2,
                    "[load_gguf] Q5_1 scales size mismatch (expected group 16)");
    ov::parallel_for(n_block, [&](size_t i) {
        uint8_t* block_data = data + i * bytes_per_block;
        float d = static_cast<float>(read_f16(block_data));
        float m = static_cast<float>(read_f16(block_data + 2));

        ov::float16 sc(d);
        ov::float16 bs(m);  // dequant is w * d + m -> zero point round(-m/d)
        scales[i * 2] = sc;
        scales[i * 2 + 1] = sc;
        biases[i * 2] = bs;
        biases[i * 2 + 1] = bs;

        uint32_t qh;
        std::memcpy(&qh, block_data + 4, sizeof(qh));
        uint8_t* qs = block_data + 8;
        for (int j = 0; j < 16; ++j) {
            uint8_t xh0 = static_cast<uint8_t>(((qh >> j) & 1) << 4);
            uint8_t xh1 = static_cast<uint8_t>(((qh >> (j + 16)) & 1) << 4);
            weights[i * 32 + j] = (qs[j] & 0x0F) | xh0;
            weights[i * 32 + j + 16] = (qs[j] >> 4) | xh1;
        }
    });
}

void gguf_load_quantized(std::unordered_map<std::string, ov::Tensor>& a,
                         std::unordered_map<std::string, gguf_tensor_type>& qtype_map,
                         const gguf_tensor& tensor) {
    uint64_t weights_per_byte;
    if (tensor.type == GGUF_TYPE_Q4_0 || tensor.type == GGUF_TYPE_Q4_1 || tensor.type == GGUF_TYPE_Q4_K) {
        weights_per_byte = 2;
    } else {  // tensor.type == GGUF_TYPE_Q8_0 || tensor.type == GGUF_TYPE_Q6_K
        weights_per_byte = 1;
    }

    std::string name(tensor.name, tensor.namelen);

    auto shape = get_shape(tensor);

    uint64_t weights_per_block;
    // All u8-path quant types (Q8_0/Q5_0/Q5_1/Q5_K/Q6_K) are emitted at group 16 so they
    // share one scale layout; this lets the GPU plugin's FullyConnectedHorizontalFusion
    // concat q/k/v scales when a model mixes these types across those projections. The
    // u4-path types (Q4_0/Q4_1/Q4_K) stay at group 32 (internally consistent, different
    // element type so never fused with the u8 ones).
    if (tensor.type == GGUF_TYPE_Q8_0 || tensor.type == GGUF_TYPE_Q6_K ||
        tensor.type == GGUF_TYPE_Q5_K || tensor.type == GGUF_TYPE_Q5_0 ||
        tensor.type == GGUF_TYPE_Q5_1) {
        weights_per_block = 16;
    } else {
        weights_per_block = 32;
    }

    OPENVINO_ASSERT(shape.back() % weights_per_block == 0,
                    "[load_gguf] tensor ",
                    name,
                    " has incompatible last dim shape: ",
                    shape.back());

    // K-quants (Q4_K/Q5_K/Q6_K) are dequantized one 256-weight super-block at a time:
    // the extract_*_k_data() loops iterate tensor.bsize / bytes_per_block super-blocks
    // and write a full super-block (256 weights) each.
    // The output buffers below, however, are sized from shape.back(), which is only
    // guarded to be a multiple of the sub-block size (32 / 16) above. A last dim that
    // is a multiple of the sub-block size but not of the 256-weight super-block (e.g.
    // 32) makes gguflib pad the data up to a full super-block, so the loop writes past
    // the end of the smaller output buffer -- an attacker-controlled heap overflow
    // (the dequantized nibbles come straight from the file). Require whole super-blocks.
    if (tensor.type == GGUF_TYPE_Q4_K || tensor.type == GGUF_TYPE_Q5_K || tensor.type == GGUF_TYPE_Q6_K) {
        constexpr uint64_t weights_per_super_block = 256;
        OPENVINO_ASSERT(shape.back() % weights_per_super_block == 0,
                        "[load_gguf] K-quant tensor ",
                        name,
                        " has a last dim of ",
                        shape.back(),
                        " which is not a multiple of the super-block size ",
                        weights_per_super_block,
                        ". The GGUF file is malformed or corrupted.");
    }

    auto weights_shape = shape;
    weights_shape.back() /= (weights_per_byte * 4);  // means u32 type can store 8 q4 or 4 q8

    ov::Tensor weights(ov::element::u32, std::move(weights_shape));
    // For scales and bias
    shape[shape.size() - 1] = shape[shape.size() - 1] / weights_per_block;

    ov::Tensor scales(ov::element::f16, shape);
    ov::Tensor biases(ov::element::f16, std::move(shape));
    if (tensor.type == GGUF_TYPE_Q4_0) {
        extract_q4_0_data(tensor, weights, scales, biases);
    } else if (tensor.type == GGUF_TYPE_Q4_1) {
        extract_q4_1_data(tensor, weights, scales, biases);
    } else if (tensor.type == GGUF_TYPE_Q8_0) {
        extract_q8_0_data(tensor, weights, scales, biases);
    } else if (tensor.type == GGUF_TYPE_Q6_K) {
        // due to WA #2135, this case will not be used, extract_q6_k_data temporarily disabled.
        extract_q6_k_data(tensor, weights, scales, biases);
    } else if (tensor.type == GGUF_TYPE_Q4_K) {
        extract_q4_k_data(tensor, weights, scales, biases);
    } else if (tensor.type == GGUF_TYPE_Q5_K) {
        extract_q5_k_data(tensor, weights, scales, biases);
    } else if (tensor.type == GGUF_TYPE_Q5_0) {
        extract_q5_0_data(tensor, weights, scales, biases);
    } else if (tensor.type == GGUF_TYPE_Q5_1) {
        extract_q5_1_data(tensor, weights, scales, biases);
    } else {
        OPENVINO_ASSERT(false, "Unsupported tensor type in 'gguf_load_quantized'");
    }

    a.emplace(name, std::move(weights));

    auto check_insert = [](const auto& inserted) {
        OPENVINO_ASSERT(inserted.second,
                        "[load_gguf] Duplicate parameter name ",
                        inserted.first->second.data(),
                        ". This can happen when loading quantized tensors");
    };

    constexpr std::string_view weight_suffix = ".weight";
    const std::string name_prefix = name.substr(0, name.length() - weight_suffix.length());
    check_insert(a.emplace(name_prefix + ".scales", std::move(scales)));
    check_insert(a.emplace(name_prefix + ".biases", std::move(biases)));

    qtype_map.emplace(name_prefix + ".qtype", static_cast<gguf_tensor_type>(tensor.type));
}
