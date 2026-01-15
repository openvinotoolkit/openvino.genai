// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <openvino/core/parallel.hpp>
#include <sstream>

#include "gguf_utils/gguf.hpp"

using namespace std;

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
        scales[i] = ov::float16::from_bits(*((uint16_t*)(data + i * bytes_per_block)));
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
        scales[i] = ov::float16::from_bits(*((uint16_t*)(data + i * bytes_per_block)));
        biases[i] = ov::float16::from_bits(*((uint16_t*)(data + i * bytes_per_block + 2)));
        unpack_32_4(data + i * bytes_per_block + 4, weights + i * 16);
    });
}

// Extracts (weight, scales, biases) from Q8_0 tensors.
// Data layout is: |16 bit scale|32 x 8bit weights|.
void extract_q8_0_data(const gguf_tensor& tensor,
                       ov::Tensor& weights_arr,
                       ov::Tensor& scales_arr,
                       ov::Tensor& biases_arr) {
    const uint64_t weights_per_block = 32;
    const uint64_t bytes_per_block = 34;  // 2 bytes scale, 32x1 byte weights
    auto data = static_cast<uint8_t*>(tensor.weights_data);
    auto weights = static_cast<uint8_t*>(weights_arr.data());
    auto scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    auto biases = biases_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    for (int64_t i = 0; i < scales_arr.get_size(); i++) {
        uint8_t* block_data = data + i * bytes_per_block;
        scales[i] = ov::float16::from_bits(*(uint16_t*)block_data);
        biases[i] = ov::float16(-128.f * static_cast<float>(scales[i]));
        for (int64_t j = 0; j < weights_per_block; ++j) {
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
        float scale_scales = static_cast<float>(ov::float16::from_bits(*((uint16_t*)block_data)));
        float scale_biases = static_cast<float>(ov::float16::from_bits(*((uint16_t*)block_data + 1)));

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
            static_cast<float>(ov::float16::from_bits(*((uint16_t*)block_data + 104)));  // (128+64+16)/2

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
    // here we only consider sub block, q6k:16 q4k:32
    if (tensor.type == GGUF_TYPE_Q6_K) {
        weights_per_block = 16;
    } else {
        weights_per_block = 32;
    }

    OPENVINO_ASSERT(shape.back() % weights_per_block == 0,
                    "[load_gguf] tensor ",
                    name,
                    " has incompatible last dim shape: ",
                    shape.back());

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
    } else {
        OPENVINO_ASSERT("Unsupported tensor type in 'gguf_load_quantized'");
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

// ====================================================================
// Quantization Functions (for requantization during GGUF load)
// ====================================================================

// Use gguflib's FP16 conversion functions
extern "C" {
    uint16_t to_half(float f);
    float from_half(uint16_t h);
}

// Quantize FP32 weights to Q4_0 format (symmetric, 4-bit)
// Algorithm: Find max value in each block, map to [-8, 7] range
// Output: U4 packed weights (2 per byte), FP16 scales, single FP16 bias
void quantize_q4_0(const float* src,
                   ov::Tensor& weights_out,
                   ov::Tensor& scales_out,
                   ov::Tensor& biases_out,
                   int64_t n_elements,
                   int64_t block_size) {
    OPENVINO_ASSERT(n_elements % block_size == 0, "n_elements must be divisible by block_size");
    
    const int64_t num_blocks = n_elements / block_size;
    
    auto* weights = static_cast<uint32_t*>(weights_out.data());  // u32 packed format
    // scales_out and biases_out are f16 type, but we write uint16_t bit representation
    auto* scales = reinterpret_cast<uint16_t*>(scales_out.data());
    auto* biases = reinterpret_cast<uint16_t*>(biases_out.data());
    
    for (int64_t i = 0; i < num_blocks; i++) {
        // Find absolute max value in block
        float amax = 0.0f;
        float max_val = 0.0f;
        
        for (int64_t j = 0; j < block_size; j++) {
            float v = src[i * block_size + j];
            if (fabsf(v) > amax) {
                amax = fabsf(v);
                max_val = v;
            }
        }
        
        // Calculate scale (symmetric: maps max to -8)
        float d = max_val / -8.0f;
        
        if (d == 0.0f) {
            // Zero block - use dummy scale and zero point at 8
            scales[i] = to_half(1.0f);
            biases[i] = to_half(-8.0f);
            // Fill with zeros packed into u32 (zp=8 means 0x88 bytes)
            uint32_t zp_packed = 0x88888888;  // 4 bytes of 0x88
            for (int64_t j = 0; j < block_size / 8; j++) {
                weights[i * block_size / 8 + j] = zp_packed;
            }
            continue;
        }
        
        float id = 1.0f / d;
        scales[i] = to_half(d);
        
        // Symmetric quantization: replicate bias for all blocks
        // (building_blocks.cpp expects biases.shape == scales.shape)
        biases[i] = to_half(-8.0f * d);
        
        // Quantize and pack weights: 2 q4 per byte, 4 bytes per u32 (8 q4 per u32)
        for (int64_t j = 0; j < block_size; j += 8) {
            uint32_t packed = 0;
            for (int k = 0; k < 4; k++) {  // 4 bytes per u32
                int idx0 = i * block_size + j + k * 2;
                int idx1 = i * block_size + j + k * 2 + 1;
                
                float x0 = src[idx0] * id;
                float x1 = src[idx1] * id;
                
                // Clamp to [0, 15] range (4-bit)
                uint8_t q0 = std::min(15, static_cast<int>(x0 + 8.5f));
                uint8_t q1 = std::min(15, static_cast<int>(x1 + 8.5f));
                
                // Pack: low 4 bits = q0, high 4 bits = q1
                uint8_t byte_val = q0 | (q1 << 4);
                packed |= (static_cast<uint32_t>(byte_val) << (k * 8));
            }
            weights[(i * block_size + j) / 8] = packed;
        }
    }
}

// Quantize FP32 weights to Q8_0 format (symmetric, 8-bit)
// Algorithm: Find max absolute value in each block, map to [-128, 127] range
// Output: U8 weights, FP16 scales, FP16 biases (replicated for building_blocks compatibility)
void quantize_q8_0(const float* src,
                   ov::Tensor& weights_out,
                   ov::Tensor& scales_out,
                   ov::Tensor& biases_out,
                   int64_t n_elements,
                   int64_t block_size) {
    OPENVINO_ASSERT(n_elements % block_size == 0, "n_elements must be divisible by block_size");
    
    const int64_t num_blocks = n_elements / block_size;
    
    auto* weights = static_cast<uint32_t*>(weights_out.data());  // u32 packed format
    // scales_out and biases_out are f16 type, but we write uint16_t bit representation
    auto* scales = reinterpret_cast<uint16_t*>(scales_out.data());
    auto* biases = reinterpret_cast<uint16_t*>(biases_out.data());
    
    for (int64_t i = 0; i < num_blocks; i++) {
        // Find absolute max value in block
        float amax = 0.0f;
        
        for (int64_t j = 0; j < block_size; j++) {
            float v = src[i * block_size + j];
            if (fabsf(v) > amax) {
                amax = fabsf(v);
            }
        }
        
        // Calculate scale
        float d = amax / 127.0f;
        float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        
        scales[i] = to_half(d);
        
        // Symmetric quantization: replicate bias for all blocks
        // (building_blocks.cpp expects biases.shape == scales.shape)
        biases[i] = to_half(-128.0f * d);
        
        // Quantize weights to [0, 255] range (offset by 128), pack 4 u8 per u32
        for (int64_t j = 0; j < block_size; j += 4) {
            uint32_t packed = 0;
            for (int k = 0; k < 4; k++) {
                float x0 = src[i * block_size + j + k] * id;
                int8_t xi0 = static_cast<int8_t>(roundf(x0));
                uint8_t u8_val = static_cast<uint8_t>(xi0 + 128);
                packed |= (static_cast<uint32_t>(u8_val) << (k * 8));
            }
            weights[(i * block_size + j) / 4] = packed;
        }
    }
}
