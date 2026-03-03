// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file rtn_quantize.hpp
 * @brief Round-To-Nearest (RTN) weight quantization for in-flight compression
 * 
 * This implements a simple but effective RTN algorithm that matches NNCF's
 * weight compression. It quantizes FP16/BF16 weights to INT4 or INT8.
 * 
 * NNCF Algorithm Reference (weight_lowering.py):
 *   - _calculate_signed_scale(): scale = max(|min|, -max) / 2^(bits-1)
 *   - _calculate_integer_quantized_weight(): round(weight / scale), clip to range
 * 
 * Performance optimizations:
 *   - Type dispatch outside hot loops
 *   - Precomputed inverse scale (multiply instead of divide)
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/core/parallel.hpp"

namespace ov {
namespace genai {
namespace rtn {

/**
 * @brief Result of RTN quantization
 */
struct QuantizedWeight {
    ov::Tensor compressed;      ///< Quantized weights (INT4 or INT8)
    ov::Tensor scale;           ///< Scale tensor for dequantization
    ov::Tensor zero_point;      ///< Zero point (for asymmetric quantization)
    ov::element::Type compressed_type; ///< Data type of compressed weights
    bool has_zero_point;        ///< True if asymmetric quantization was used
};

/**
 * @brief Convert BF16 to float
 */
inline float bf16_to_float(uint16_t bf16_val) {
    uint32_t fp32_bits = static_cast<uint32_t>(bf16_val) << 16;
    float result;
    std::memcpy(&result, &fp32_bits, sizeof(float));
    return result;
}

/**
 * @brief Convert FP16 to float
 */
inline float fp16_to_float(uint16_t fp16_val) {
    // Extract components
    uint32_t sign = (fp16_val >> 15) & 0x1;
    uint32_t exp = (fp16_val >> 10) & 0x1F;
    uint32_t mant = fp16_val & 0x3FF;
    
    uint32_t fp32_bits;
    if (exp == 0) {
        if (mant == 0) {
            // Zero
            fp32_bits = sign << 31;
        } else {
            // Subnormal
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            fp32_bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        // Inf or NaN
        fp32_bits = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        // Normal
        fp32_bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    
    float result;
    std::memcpy(&result, &fp32_bits, sizeof(float));
    return result;
}

/**
 * @brief Convert float to FP16
 */
inline uint16_t float_to_fp16(float value) {
    uint32_t fp32_bits;
    std::memcpy(&fp32_bits, &value, sizeof(float));
    uint32_t sign = (fp32_bits >> 31) & 0x1;
    int32_t exp = ((fp32_bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (fp32_bits >> 13) & 0x3FF;
    
    if (exp <= 0) {
        return static_cast<uint16_t>(sign << 15);  // Underflow to zero
    } else if (exp >= 31) {
        return static_cast<uint16_t>((sign << 15) | 0x7C00);  // Overflow to inf
    } else {
        return static_cast<uint16_t>((sign << 15) | (exp << 10) | mant);
    }
}

/**
 * @brief Get float value from tensor at given index
 */
inline float get_float_value(const ov::Tensor& tensor, size_t idx) {
    auto dtype = tensor.get_element_type();
    if (dtype == ov::element::f32) {
        return static_cast<const float*>(tensor.data())[idx];
    } else if (dtype == ov::element::f16) {
        return fp16_to_float(static_cast<const uint16_t*>(tensor.data())[idx]);
    } else if (dtype == ov::element::bf16) {
        return bf16_to_float(static_cast<const uint16_t*>(tensor.data())[idx]);
    }
    throw std::runtime_error("Unsupported tensor dtype for quantization");
}

// Check for AVX2 support
#if defined(__AVX2__) || (defined(_MSC_VER) && !defined(_M_ARM64))
#   define OV_GENAI_RTN_AVX2 1
#   include <immintrin.h>
#endif

// Conversion function wrappers
inline float convert_f32(float val) { return val; }
inline float convert_f16(uint16_t val) { return fp16_to_float(val); }
inline float convert_bf16(uint16_t val) { return bf16_to_float(val); }

enum class InputType { F32, F16, BF16 };

/**
 * @brief Optimized Min/Max finder with AVX2 support
 */
template<typename DataType, InputType InType>
inline void find_min_max(const DataType* data, size_t count, float& min_val, float& max_val) {
    size_t i = 0;
    float l_min = min_val;
    float l_max = max_val;

#ifdef OV_GENAI_RTN_AVX2
    size_t simd_step = 8; // Process 8 floats at a time (AVX2 256-bit)
    if (count >= simd_step) {
        __m256 v_min = _mm256_set1_ps(l_min);
        __m256 v_max = _mm256_set1_ps(l_max);
        
        size_t count_aligned = count - (count % simd_step);
        for (; i < count_aligned; i += simd_step) {
            __m256 vals;
            
            if constexpr (InType == InputType::F32) {
                vals = _mm256_loadu_ps(reinterpret_cast<const float*>(data + i));
            } 
            else if constexpr (InType == InputType::F16) {
                // Load 8 FP16 (128-bit), convert to FP32 (256-bit)
                __m128i v_fp16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + i));
                vals = _mm256_cvtph_ps(v_fp16);
            } 
            else if constexpr (InType == InputType::BF16) {
                // Load 8 BF16 (128-bit)
                __m128i v_bf16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data + i));
                // Reshuffle/unpack to get to upper 16 bits of 32-bit integers
                // BF16: [b0, b1, b2... b7] -> FP32: [b0<<16, b1<<16... ]
                
                // Convert to 32-bit int (zero extended)
                __m256i v_32 = _mm256_cvtepu16_epi32(v_bf16);
                // Shift left by 16
                v_32 = _mm256_slli_epi32(v_32, 16);
                // Cast to float
                vals = _mm256_castsi256_ps(v_32);
            }
            
            v_min = _mm256_min_ps(v_min, vals);
            v_max = _mm256_max_ps(v_max, vals);
        }
        
        // Horizontal reduction
        float tmp_min[8], tmp_max[8];
        _mm256_storeu_ps(tmp_min, v_min);
        _mm256_storeu_ps(tmp_max, v_max);
        
        for (int k = 0; k < 8; ++k) {
            l_min = std::min(l_min, tmp_min[k]);
            l_max = std::max(l_max, tmp_max[k]);
        }
    }
#endif

    // Scalar fallback/remainder
    for (; i < count; ++i) {
        float val;
        if constexpr (InType == InputType::F32) val = convert_f32(data[i]);
        else if constexpr (InType == InputType::F16) val = convert_f16(data[i]);
        else if constexpr (InType == InputType::BF16) val = convert_bf16(data[i]);
        
        l_min = std::min(l_min, val);
        l_max = std::max(l_max, val);
    }
    
    min_val = l_min;
    max_val = l_max;
}

/**
 * @brief Templated quantization for INT4 symmetric - optimized inner loop
 */
template<typename DataType, InputType InType, float (*ConvertFunc)(DataType)>
inline void quantize_int4_sym_typed(
    const DataType* src_data,
    uint8_t* compressed_data,
    uint16_t* scale_data,
    size_t out_features,
    size_t in_features,
    size_t packed_in_features,
    size_t num_groups,
    int group_size
) {
    constexpr float factor = 8.0f;  // 2^(4-1) = 8
    constexpr int8_t level_low = -8;
    constexpr int8_t level_high = 7;
    constexpr float eps = 1.1920929e-7f;
    
    ov::parallel_for(out_features, [&](size_t row) {
        const DataType* row_data = src_data + row * in_features;
        uint8_t* row_compressed = compressed_data + row * packed_in_features;
        uint16_t* row_scale = scale_data + row * num_groups;
        
        for (size_t g = 0; g < num_groups; g++) {
            size_t group_start = g * group_size;
            size_t group_end = std::min(group_start + static_cast<size_t>(group_size), in_features);
            
            // Single pass: find min/max
            float w_min = std::numeric_limits<float>::max();
            float w_max = std::numeric_limits<float>::lowest();
            
            find_min_max<DataType, InType>(row_data + group_start, group_end - group_start, w_min, w_max);
            
            // Compute scale
            float max_abs = std::max(std::abs(w_min), std::abs(w_max));
            float scale_val = max_abs / factor;
            if (scale_val < eps) scale_val = eps;
            float inv_scale = 1.0f / scale_val;
            
            // Store scale
            row_scale[g] = float_to_fp16(scale_val);
            
            // Quantize elements with loop unrolling
            size_t col = group_start;
            
#ifdef OV_GENAI_RTN_AVX2
            if (col + 7 < group_end) {
                __m256 v_inv_scale = _mm256_set1_ps(inv_scale);
                __m256i v_low = _mm256_set1_epi32(level_low);
                __m256i v_high = _mm256_set1_epi32(level_high);

                for (; col + 7 < group_end; col += 8) {
                    __m256 vals;
                    if constexpr (InType == InputType::F32) {
                        vals = _mm256_loadu_ps(reinterpret_cast<const float*>(row_data + col));
                    } else if constexpr (InType == InputType::F16) {
                        __m128i v_fp16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row_data + col));
                        vals = _mm256_cvtph_ps(v_fp16);
                    } else if constexpr (InType == InputType::BF16) {
                        __m128i v_bf16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row_data + col));
                        __m256i v_32 = _mm256_cvtepu16_epi32(v_bf16);
                        v_32 = _mm256_slli_epi32(v_32, 16);
                        vals = _mm256_castsi256_ps(v_32);
                    }

                    // Quantize
                    vals = _mm256_mul_ps(vals, v_inv_scale);
                    vals = _mm256_round_ps(vals, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    __m256i v_q = _mm256_cvtps_epi32(vals);
                    
                    // Clamp
                    v_q = _mm256_max_epi32(v_q, v_low);
                    v_q = _mm256_min_epi32(v_q, v_high);

                    // Store temp to pack scalar-wise (faster than complex AVX bit packing for 4-bit)
                    int32_t tmp[8];
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), v_q);
                    
                    uint8_t* d = row_compressed + col / 2;
                    d[0] = (tmp[0] & 0x0F) | ((tmp[1] & 0x0F) << 4);
                    d[1] = (tmp[2] & 0x0F) | ((tmp[3] & 0x0F) << 4);
                    d[2] = (tmp[4] & 0x0F) | ((tmp[5] & 0x0F) << 4);
                    d[3] = (tmp[6] & 0x0F) | ((tmp[7] & 0x0F) << 4);
                }
            }
#endif

            // Handle start alignment if off-boundary (odd)
            if (col % 2 != 0 && col < group_end) {
                float val = ConvertFunc(row_data[col]);
                int8_t quantized = static_cast<int8_t>(std::round(val * inv_scale));
                quantized = std::max(level_low, std::min(level_high, quantized));
                uint8_t packed = static_cast<uint8_t>(quantized) & 0x0F;
                row_compressed[col / 2] |= (packed << 4);
                col++;
            }
            
            // Process pairs (unrolled)
            for (; col + 1 < group_end; col += 2) {
                float val0 = ConvertFunc(row_data[col]);
                float val1 = ConvertFunc(row_data[col + 1]);
                
                int8_t q0 = static_cast<int8_t>(std::round(val0 * inv_scale));
                q0 = std::max(level_low, std::min(level_high, q0));
                
                int8_t q1 = static_cast<int8_t>(std::round(val1 * inv_scale));
                q1 = std::max(level_low, std::min(level_high, q1));
                
                uint8_t packed = (static_cast<uint8_t>(q0) & 0x0F) | ((static_cast<uint8_t>(q1) & 0x0F) << 4);
                row_compressed[col / 2] = packed;
            }
            
            // Handle last element
            if (col < group_end) {
                float val = ConvertFunc(row_data[col]);
                int8_t quantized = static_cast<int8_t>(std::round(val * inv_scale));
                quantized = std::max(level_low, std::min(level_high, quantized));
                
                uint8_t packed = static_cast<uint8_t>(quantized) & 0x0F;
                row_compressed[col / 2] = packed;
            }
        }
    });
}

// Conversion function wrappers for template - moved up


/**
 * @brief Templated quantization for INT4 asymmetric - optimized inner loop
 */
template<typename DataType, InputType InType, float (*ConvertFunc)(DataType)>
inline void quantize_int4_asym_typed(
    const DataType* src_data,
    uint8_t* compressed_data,
    uint16_t* scale_data,
    uint8_t* zero_point_data,
    size_t out_features,
    size_t in_features,
    size_t packed_in_features,
    size_t num_groups,
    int group_size
) {
    constexpr float levels = 15.0f;  // 2^4 - 1
    constexpr int32_t level_low = 0;
    constexpr int32_t level_high = 15;
    constexpr float eps = 1.1920929e-7f;
    
    ov::parallel_for(out_features, [&](size_t row) {
        const DataType* row_data = src_data + row * in_features;
        uint8_t* row_compressed = compressed_data + row * packed_in_features;
        uint16_t* row_scale = scale_data + row * num_groups;
        uint8_t* row_zp = zero_point_data + row * num_groups;
        
        for (size_t g = 0; g < num_groups; g++) {
            size_t group_start = g * group_size;
            size_t group_end = std::min(group_start + static_cast<size_t>(group_size), in_features);
            
            float w_min = std::numeric_limits<float>::max();
            float w_max = std::numeric_limits<float>::lowest();
            
            find_min_max<DataType, InType>(row_data + group_start, group_end - group_start, w_min, w_max);
            
            float scale_val = (w_max - w_min) / levels;
            if (scale_val < eps) scale_val = eps;
            float inv_scale = 1.0f / scale_val;
            
            int32_t zp = static_cast<int32_t>(std::round(-w_min * inv_scale));
            zp = std::max(level_low, std::min(level_high, zp));
            
            row_scale[g] = float_to_fp16(scale_val);
            row_zp[g] = static_cast<uint8_t>(zp);
            
            size_t col = group_start;

#ifdef OV_GENAI_RTN_AVX2
            if (col + 7 < group_end) {
                __m256 v_inv_scale = _mm256_set1_ps(inv_scale);
                __m256i v_zp = _mm256_set1_epi32(zp);
                __m256i v_low = _mm256_set1_epi32(level_low);
                __m256i v_high = _mm256_set1_epi32(level_high);

                for (; col + 7 < group_end; col += 8) {
                    __m256 vals;
                    if constexpr (InType == InputType::F32) {
                        vals = _mm256_loadu_ps(reinterpret_cast<const float*>(row_data + col));
                    } else if constexpr (InType == InputType::F16) {
                        __m128i v_fp16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row_data + col));
                        vals = _mm256_cvtph_ps(v_fp16);
                    } else if constexpr (InType == InputType::BF16) {
                        __m128i v_bf16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row_data + col));
                        __m256i v_32 = _mm256_cvtepu16_epi32(v_bf16);
                        v_32 = _mm256_slli_epi32(v_32, 16);
                        vals = _mm256_castsi256_ps(v_32);
                    }

                    // Quantize
                    vals = _mm256_mul_ps(vals, v_inv_scale);
                    vals = _mm256_round_ps(vals, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    __m256i v_q = _mm256_cvtps_epi32(vals);
                    v_q = _mm256_add_epi32(v_q, v_zp);
                    
                    // Clamp
                    v_q = _mm256_max_epi32(v_q, v_low);
                    v_q = _mm256_min_epi32(v_q, v_high);

                    // Store temp to pack scalar-wise
                    int32_t tmp[8];
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), v_q);
                    
                    uint8_t* d = row_compressed + col / 2;
                    d[0] = (tmp[0] & 0x0F) | ((tmp[1] & 0x0F) << 4);
                    d[1] = (tmp[2] & 0x0F) | ((tmp[3] & 0x0F) << 4);
                    d[2] = (tmp[4] & 0x0F) | ((tmp[5] & 0x0F) << 4);
                    d[3] = (tmp[6] & 0x0F) | ((tmp[7] & 0x0F) << 4);
                }
            }
#endif

            if (col % 2 != 0 && col < group_end) {
                float val = ConvertFunc(row_data[col]);
                int32_t quantized = static_cast<int32_t>(std::round(val * inv_scale)) + zp;
                quantized = std::max(level_low, std::min(level_high, quantized));
                uint8_t packed = static_cast<uint8_t>(quantized) & 0x0F;
                row_compressed[col / 2] |= (packed << 4);
                col++;
            }
            
            for (; col + 1 < group_end; col += 2) {
                float val0 = ConvertFunc(row_data[col]);
                float val1 = ConvertFunc(row_data[col+1]);
                
                int32_t q0 = static_cast<int32_t>(std::round(val0 * inv_scale)) + zp;
                q0 = std::max(level_low, std::min(level_high, q0));
                
                int32_t q1 = static_cast<int32_t>(std::round(val1 * inv_scale)) + zp;
                q1 = std::max(level_low, std::min(level_high, q1));
                
                uint8_t packed = (static_cast<uint8_t>(q0) & 0x0F) | ((static_cast<uint8_t>(q1) & 0x0F) << 4);
                row_compressed[col / 2] = packed;
            }
            
            if (col < group_end) {
                float val = ConvertFunc(row_data[col]);
                int32_t quantized = static_cast<int32_t>(std::round(val * inv_scale)) + zp;
                quantized = std::max(level_low, std::min(level_high, quantized));
                uint8_t packed = static_cast<uint8_t>(quantized) & 0x0F;
                row_compressed[col / 2] = packed;
            }
        }
    });
}

/**
 * @brief Quantize weights using symmetric INT4 RTN algorithm (NNCF-compatible)
 * 
 * NNCF Algorithm:
 *   factor = 2^(num_bits - 1) = 8 for INT4
 *   scale = max(|min|, |max|) / factor
 *   quantized = round(weight / scale)
 *   quantized = clamp(quantized, -8, 7)
 * 
 * Optimized version with:
 *   - Type dispatch outside hot loops
 *   - Precomputed inverse scale (multiply instead of divide)
 * 
 * @param weight Input weight tensor (FP16 or BF16), shape [out_features, in_features] or [batch, out_features, in_features]
 * @param group_size Number of elements per quantization group (default: 128)
 * @return QuantizedWeight with INT4 weights packed as U8 and FP16 scales
 */
inline QuantizedWeight quantize_int4_sym(const ov::Tensor& weight, int group_size = 128) {
    auto shape = weight.get_shape();
    if (shape.size() != 1 && shape.size() != 2 && shape.size() != 3) {
        throw std::runtime_error("Weight tensor must be 1D, 2D or 3D for quantization");
    }
    
    // Support 1D [in_features], 2D [out_features, in_features] and 3D [batch, out_features, in_features]
    size_t out_features = 1;
    size_t in_features = shape[0];
    ov::Shape output_shape = {};

    if (shape.size() == 2) {
        out_features = shape[0];
        in_features = shape[1];
        output_shape = {shape[0]};
    } else if (shape.size() == 3) {
        out_features = shape[0] * shape[1];
        in_features = shape[2];
        output_shape = {shape[0], shape[1]};
    }
    
    // Handle channel-wise quantization (group_size = -1)
    if (group_size <= 0) {
        group_size = static_cast<int>(in_features);
    }
    
    size_t num_groups = (in_features + group_size - 1) / group_size;
    size_t packed_in_features = (in_features + 1) / 2;
    
    // Output shapes: append packed_in_features or num_groups to output_shape
    ov::Shape compressed_shape = output_shape;
    compressed_shape.push_back(packed_in_features);
    ov::Shape scale_shape = output_shape;
    scale_shape.push_back(num_groups);
    
    ov::Tensor compressed(ov::element::u8, compressed_shape);
    ov::Tensor scale(ov::element::f16, scale_shape);
    
    auto* compressed_data = static_cast<uint8_t*>(compressed.data());
    auto* scale_data = static_cast<uint16_t*>(scale.data());
    
    // Initialize compressed data to zero (important for proper nibble packing)
    std::memset(compressed_data, 0, out_features * packed_in_features);
    
    // Dispatch based on dtype - type check happens ONCE, not per element
    auto dtype = weight.get_element_type();
    if (dtype == ov::element::f32) {
        quantize_int4_sym_typed<float, InputType::F32, convert_f32>(
            static_cast<const float*>(weight.data()),
            compressed_data, scale_data,
            out_features, in_features, packed_in_features, num_groups, group_size);
    } else if (dtype == ov::element::f16) {
        quantize_int4_sym_typed<uint16_t, InputType::F16, convert_f16>(
            static_cast<const uint16_t*>(weight.data()),
            compressed_data, scale_data,
            out_features, in_features, packed_in_features, num_groups, group_size);
    } else if (dtype == ov::element::bf16) {
        quantize_int4_sym_typed<uint16_t, InputType::BF16, convert_bf16>(
            static_cast<const uint16_t*>(weight.data()),
            compressed_data, scale_data,
            out_features, in_features, packed_in_features, num_groups, group_size);
    } else {
        throw std::runtime_error("Unsupported tensor dtype for quantization");
    }
    
    QuantizedWeight result;
    result.compressed = compressed;
    result.scale = scale;
    result.has_zero_point = false;
    result.compressed_type = ov::element::i4;
    return result;
}

/**
 * @brief Quantize weights using asymmetric INT4 RTN algorithm (NNCF-compatible)
 * 
 * NNCF Algorithm:
 *   levels = 2^num_bits - 1 = 15 for INT4
 *   scale = (max - min) / levels
 *   zero_point = round(-min / scale)
 *   quantized = round(weight / scale) + zero_point
 *   quantized = clamp(quantized, 0, 15)
 * 
 * Optimized version with:
 *   - Type dispatch outside hot loops
 *   - Precomputed inverse scale (multiply instead of divide)
 * 
 * @param weight Input weight tensor (FP16 or BF16), shape [out_features, in_features] or [batch, out_features, in_features]
 * @param group_size Number of elements per quantization group (default: 128)
 * @return QuantizedWeight with INT4 weights packed as U8, FP16 scales, and U8 zero points
 */
inline QuantizedWeight quantize_int4_asym(const ov::Tensor& weight, int group_size = 128) {
    auto shape = weight.get_shape();
    if (shape.size() != 1 && shape.size() != 2 && shape.size() != 3) {
        throw std::runtime_error("Weight tensor must be 1D, 2D or 3D for quantization");
    }
    
    // Support 1D [in_features], 2D [out_features, in_features] and 3D [batch, out_features, in_features]
    size_t out_features = 1;
    size_t in_features = shape[0];
    ov::Shape output_shape = {};

    if (shape.size() == 2) {
        out_features = shape[0];
        in_features = shape[1];
        output_shape = {shape[0]};
    } else if (shape.size() == 3) {
        out_features = shape[0] * shape[1];
        in_features = shape[2];
        output_shape = {shape[0], shape[1]};
    }
    
    // Handle channel-wise quantization (group_size = -1)
    if (group_size <= 0) {
        group_size = static_cast<int>(in_features);
    }
    
    size_t num_groups = (in_features + group_size - 1) / group_size;
    size_t packed_in_features = (in_features + 1) / 2;
    
    // Output shapes: append packed_in_features or num_groups to output_shape
    ov::Shape compressed_shape = output_shape;
    compressed_shape.push_back(packed_in_features);
    ov::Shape scale_shape = output_shape;
    scale_shape.push_back(num_groups);
    
    ov::Tensor compressed(ov::element::u8, compressed_shape);
    ov::Tensor scale(ov::element::f16, scale_shape);
    ov::Tensor zero_point(ov::element::u8, scale_shape);
    
    auto* compressed_data = static_cast<uint8_t*>(compressed.data());
    auto* scale_data = static_cast<uint16_t*>(scale.data());
    auto* zero_point_data = static_cast<uint8_t*>(zero_point.data());
    
    // Initialize compressed data to zero (for nibble packing)
    std::memset(compressed_data, 0, out_features * packed_in_features);
    
    auto dtype = weight.get_element_type();
    if (dtype == ov::element::f32) {
        quantize_int4_asym_typed<float, InputType::F32, convert_f32>(
            static_cast<const float*>(weight.data()),
            compressed_data, scale_data, zero_point_data,
            out_features, in_features, packed_in_features, num_groups, group_size);
    } else if (dtype == ov::element::f16) {
        quantize_int4_asym_typed<uint16_t, InputType::F16, convert_f16>(
            static_cast<const uint16_t*>(weight.data()),
            compressed_data, scale_data, zero_point_data,
            out_features, in_features, packed_in_features, num_groups, group_size);
    } else if (dtype == ov::element::bf16) {
        quantize_int4_asym_typed<uint16_t, InputType::BF16, convert_bf16>(
            static_cast<const uint16_t*>(weight.data()),
            compressed_data, scale_data, zero_point_data,
            out_features, in_features, packed_in_features, num_groups, group_size);
    } else {
        throw std::runtime_error("Unsupported tensor dtype for quantization");
    }
    
    QuantizedWeight result;
    result.compressed = compressed;
    result.scale = scale;
    result.zero_point = zero_point;
    result.has_zero_point = true;
    result.compressed_type = ov::element::u4;
    return result;
}

/**
 * @brief Templated quantization for INT8 symmetric - optimized inner loop
 */
template<typename DataType, InputType InType, float (*ConvertFunc)(DataType)>
inline void quantize_int8_sym_typed(
    const DataType* src_data,
    int8_t* compressed_data,
    uint16_t* scale_data,
    size_t out_features,
    size_t in_features,
    size_t num_groups,
    int group_size
) {
    constexpr float factor = 128.0f;  // 2^(8-1) = 128
    constexpr int32_t level_low = -128;
    constexpr int32_t level_high = 127;
    constexpr float eps = 1.1920929e-7f;
    
    ov::parallel_for(out_features, [&](size_t row) {
        const DataType* row_data = src_data + row * in_features;
        int8_t* row_compressed = compressed_data + row * in_features;
        uint16_t* row_scale = scale_data + row * num_groups;
        
        for (size_t g = 0; g < num_groups; g++) {
            size_t group_start = g * group_size;
            size_t group_end = std::min(group_start + static_cast<size_t>(group_size), in_features);
            
            // Single pass: find min/max
            float w_min = std::numeric_limits<float>::max();
            float w_max = std::numeric_limits<float>::lowest();
            
            find_min_max<DataType, InType>(row_data + group_start, group_end - group_start, w_min, w_max);
            
            // Compute scale
            float max_abs = std::max(std::abs(w_min), std::abs(w_max));
            float scale_val = max_abs / factor;
            if (scale_val < eps) scale_val = eps;
            float inv_scale = 1.0f / scale_val;
            
            // Store scale
            row_scale[g] = float_to_fp16(scale_val);
            
#ifdef OV_GENAI_RTN_AVX2
            size_t col = group_start;
            if (col + 7 < group_end) {
                __m256 v_inv_scale = _mm256_set1_ps(inv_scale);
                __m256i v_low = _mm256_set1_epi32(level_low);
                __m256i v_high = _mm256_set1_epi32(level_high);

                for (; col + 7 < group_end; col += 8) {
                    __m256 vals;
                    if constexpr (InType == InputType::F32) {
                        vals = _mm256_loadu_ps(reinterpret_cast<const float*>(row_data + col));
                    } else if constexpr (InType == InputType::F16) {
                        __m128i v_fp16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row_data + col));
                        vals = _mm256_cvtph_ps(v_fp16);
                    } else if constexpr (InType == InputType::BF16) {
                        __m128i v_bf16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row_data + col));
                        __m256i v_32 = _mm256_cvtepu16_epi32(v_bf16);
                        v_32 = _mm256_slli_epi32(v_32, 16);
                        vals = _mm256_castsi256_ps(v_32);
                    }

                    // Quantize
                    vals = _mm256_mul_ps(vals, v_inv_scale);
                    vals = _mm256_round_ps(vals, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    __m256i v_q = _mm256_cvtps_epi32(vals);
                    
                    // Clamp
                    v_q = _mm256_max_epi32(v_q, v_low);
                    v_q = _mm256_min_epi32(v_q, v_high);

                    // Store elements
                    int8_t* d = row_compressed + col;
                    int32_t tmp[8];
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), v_q);
                    
                    for(int k=0; k<8; ++k) d[k] = static_cast<int8_t>(tmp[k]);
                }
            }
            // Fallback loop continues from 'col'
            for (; col < group_end; col++) {
#else
            // Quantize elements
            for (size_t col = group_start; col < group_end; col++) {
#endif
                float val = ConvertFunc(row_data[col]);
                int32_t quantized = static_cast<int32_t>(std::round(val * inv_scale));
                quantized = std::max(level_low, std::min(level_high, quantized));
                row_compressed[col] = static_cast<int8_t>(quantized);
            }
        }
    });
}

/**
 * @brief Templated quantization for INT8 asymmetric - optimized inner loop
 */
template<typename DataType, InputType InType, float (*ConvertFunc)(DataType)>
inline void quantize_int8_asym_typed(
    const DataType* src_data,
    uint8_t* compressed_data,
    uint16_t* scale_data,
    uint8_t* zero_point_data,
    size_t out_features,
    size_t in_features,
    size_t num_groups,
    int group_size
) {
    constexpr float levels = 255.0f;  // 2^8 - 1
    constexpr int32_t level_low = 0;
    constexpr int32_t level_high = 255;
    constexpr float eps = 1.1920929e-7f;
    
    ov::parallel_for(out_features, [&](size_t row) {
        const DataType* row_data = src_data + row * in_features;
        uint8_t* row_compressed = compressed_data + row * in_features;
        uint16_t* row_scale = scale_data + row * num_groups;
        uint8_t* row_zp = zero_point_data + row * num_groups;
        
        for (size_t g = 0; g < num_groups; g++) {
            size_t group_start = g * group_size;
            size_t group_end = std::min(group_start + static_cast<size_t>(group_size), in_features);
            
            float w_min = std::numeric_limits<float>::max();
            float w_max = std::numeric_limits<float>::lowest();
            
            find_min_max<DataType, InType>(row_data + group_start, group_end - group_start, w_min, w_max);
            
            float scale_val = (w_max - w_min) / levels;
            if (scale_val < eps) scale_val = eps;
            float inv_scale = 1.0f / scale_val;
            
            int32_t zp = static_cast<int32_t>(std::round(-w_min * inv_scale));
            zp = std::max(level_low, std::min(level_high, zp));
            
            row_scale[g] = float_to_fp16(scale_val);
            row_zp[g] = static_cast<uint8_t>(zp);
            
            size_t col = group_start;

#ifdef OV_GENAI_RTN_AVX2
            if (col + 7 < group_end) {
                __m256 v_inv_scale = _mm256_set1_ps(inv_scale);
                __m256i v_zp = _mm256_set1_epi32(zp);
                __m256i v_low = _mm256_set1_epi32(level_low);
                __m256i v_high = _mm256_set1_epi32(level_high);

                for (; col + 7 < group_end; col += 8) {
                    __m256 vals;
                    if constexpr (InType == InputType::F32) {
                        vals = _mm256_loadu_ps(reinterpret_cast<const float*>(row_data + col));
                    } else if constexpr (InType == InputType::F16) {
                        __m128i v_fp16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row_data + col));
                        vals = _mm256_cvtph_ps(v_fp16);
                    } else if constexpr (InType == InputType::BF16) {
                        __m128i v_bf16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row_data + col));
                        __m256i v_32 = _mm256_cvtepu16_epi32(v_bf16);
                        v_32 = _mm256_slli_epi32(v_32, 16);
                        vals = _mm256_castsi256_ps(v_32);
                    }

                    // Quantize
                    vals = _mm256_mul_ps(vals, v_inv_scale);
                    vals = _mm256_round_ps(vals, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                    __m256i v_q = _mm256_cvtps_epi32(vals);
                    v_q = _mm256_add_epi32(v_q, v_zp);
                    
                    // Clamp
                    v_q = _mm256_max_epi32(v_q, v_low);
                    v_q = _mm256_min_epi32(v_q, v_high);

                    // Store elements
                    uint8_t* d = row_compressed + col;
                    int32_t tmp[8];
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), v_q);
                    
                    for(int k=0; k<8; ++k) d[k] = static_cast<uint8_t>(tmp[k]);
                }
            }
#endif
            
            for (; col < group_end; col++) {
                float val = ConvertFunc(row_data[col]);
                int32_t quantized = static_cast<int32_t>(std::round(val * inv_scale)) + zp;
                quantized = std::max(level_low, std::min(level_high, quantized));
                row_compressed[col] = static_cast<uint8_t>(quantized);
            }
        }
    });
}

/**
 * @brief Quantize weights using symmetric INT8 RTN algorithm (NNCF-compatible)
 * 
 * NNCF Algorithm:
 *   factor = 2^(num_bits - 1) = 128 for INT8
 *   scale = max(|min|, |max|) / factor
 *   quantized = round(weight / scale)
 *   quantized = clamp(quantized, -128, 127)
 * 
 * Optimized version with:
 *   - Type dispatch outside hot loops
 *   - Precomputed inverse scale (multiply instead of divide)
 * 
 * @param weight Input weight tensor (FP16 or BF16), shape [out_features, in_features] or [batch, out_features, in_features]
 * @param group_size Number of elements per quantization group (default: 128)
 * @return QuantizedWeight with INT8 weights and FP16 scales
 */
inline QuantizedWeight quantize_int8_sym(const ov::Tensor& weight, int group_size = 128) {
    auto shape = weight.get_shape();
    if (shape.size() != 1 && shape.size() != 2 && shape.size() != 3) {
        throw std::runtime_error("Weight tensor must be 1D, 2D or 3D for quantization");
    }
    
    // Support 1D [in_features], 2D [out_features, in_features] and 3D [batch, out_features, in_features]
    size_t out_features = 1;
    size_t in_features = shape[0];
    ov::Shape output_shape = {};

    if (shape.size() == 2) {
        out_features = shape[0];
        in_features = shape[1];
        output_shape = {shape[0]};
    } else if (shape.size() == 3) {
        out_features = shape[0] * shape[1];
        in_features = shape[2];
        output_shape = {shape[0], shape[1]};
    }
    
    // Handle channel-wise quantization (group_size = -1)
    if (group_size <= 0) {
        group_size = static_cast<int>(in_features);
    }
    
    size_t num_groups = (in_features + group_size - 1) / group_size;
    
    // Output shapes: append in_features or num_groups to output_shape
    ov::Shape compressed_shape = output_shape;
    compressed_shape.push_back(in_features);
    ov::Shape scale_shape = output_shape;
    scale_shape.push_back(num_groups);
    
    ov::Tensor compressed(ov::element::i8, compressed_shape);
    ov::Tensor scale(ov::element::f16, scale_shape);
    
    auto* compressed_data = static_cast<int8_t*>(compressed.data());
    auto* scale_data = static_cast<uint16_t*>(scale.data());
    
    // Dispatch based on dtype - type check happens ONCE, not per element
    auto dtype = weight.get_element_type();
    if (dtype == ov::element::f32) {
        quantize_int8_sym_typed<float, InputType::F32, convert_f32>(
            static_cast<const float*>(weight.data()),
            compressed_data, scale_data,
            out_features, in_features, num_groups, group_size);
    } else if (dtype == ov::element::f16) {
        quantize_int8_sym_typed<uint16_t, InputType::F16, convert_f16>(
            static_cast<const uint16_t*>(weight.data()),
            compressed_data, scale_data,
            out_features, in_features, num_groups, group_size);
    } else if (dtype == ov::element::bf16) {
        quantize_int8_sym_typed<uint16_t, InputType::BF16, convert_bf16>(
            static_cast<const uint16_t*>(weight.data()),
            compressed_data, scale_data,
            out_features, in_features, num_groups, group_size);
    } else {
        throw std::runtime_error("Unsupported tensor dtype for quantization");
    }
    
    QuantizedWeight result;
    result.compressed = compressed;
    result.scale = scale;
    result.compressed_type = ov::element::i8;
    result.has_zero_point = false;
    return result;
}

/**
 * @brief Quantize weights using asymmetric INT8 RTN algorithm (NNCF-compatible)
 * 
 * NNCF Algorithm:
 *   levels = 2^num_bits - 1 = 255 for INT8
 *   scale = (max - min) / levels
 *   zero_point = round(-min / scale)
 *   quantized = round(weight / scale) + zero_point
 *   quantized = clamp(quantized, 0, 255)
 * 
 * Optimized version with:
 *   - Type dispatch outside hot loops
 *   - Precomputed inverse scale (multiply instead of divide)
 * 
 * @param weight Input weight tensor (FP16 or BF16), shape [out_features, in_features] or [batch, out_features, in_features]
 * @param group_size Number of elements per quantization group (default: 128)
 * @return QuantizedWeight with INT8 weights, FP16 scales, and U8 zero points
 */
inline QuantizedWeight quantize_int8_asym(const ov::Tensor& weight, int group_size = 128) {
    auto shape = weight.get_shape();
    if (shape.size() != 1 && shape.size() != 2 && shape.size() != 3) {
        throw std::runtime_error("Weight tensor must be 1D, 2D or 3D for quantization");
    }
    
    // Support 1D [in_features], 2D [out_features, in_features] and 3D [batch, out_features, in_features]
    size_t out_features = 1;
    size_t in_features = shape[0];
    ov::Shape output_shape = {};

    if (shape.size() == 2) {
        out_features = shape[0];
        in_features = shape[1];
        output_shape = {shape[0]};
    } else if (shape.size() == 3) {
        out_features = shape[0] * shape[1];
        in_features = shape[2];
        output_shape = {shape[0], shape[1]};
    }

    // Handle channel-wise quantization (group_size = -1)
    if (group_size <= 0) {
        group_size = static_cast<int>(in_features);
    }
    
    size_t num_groups = (in_features + group_size - 1) / group_size;
    
    // Output shapes: append in_features or num_groups to output_shape
    ov::Shape compressed_shape = output_shape;
    compressed_shape.push_back(in_features);
    ov::Shape scale_shape = output_shape;
    scale_shape.push_back(num_groups);
    
    ov::Tensor compressed(ov::element::u8, compressed_shape);
    ov::Tensor scale(ov::element::f16, scale_shape);
    ov::Tensor zero_point(ov::element::u8, scale_shape);
    
    auto* compressed_data = static_cast<uint8_t*>(compressed.data());
    auto* scale_data = static_cast<uint16_t*>(scale.data());
    auto* zero_point_data = static_cast<uint8_t*>(zero_point.data());
    
    // Dispatch based on dtype - type check happens ONCE, not per element
    auto dtype = weight.get_element_type();
    if (dtype == ov::element::f32) {
        quantize_int8_asym_typed<float, InputType::F32, convert_f32>(
            static_cast<const float*>(weight.data()),
            compressed_data, scale_data, zero_point_data,
            out_features, in_features, num_groups, group_size);
    } else if (dtype == ov::element::f16) {
        quantize_int8_asym_typed<uint16_t, InputType::F16, convert_f16>(
            static_cast<const uint16_t*>(weight.data()),
            compressed_data, scale_data, zero_point_data,
            out_features, in_features, num_groups, group_size);
    } else if (dtype == ov::element::bf16) {
        quantize_int8_asym_typed<uint16_t, InputType::BF16, convert_bf16>(
            static_cast<const uint16_t*>(weight.data()),
            compressed_data, scale_data, zero_point_data,
            out_features, in_features, num_groups, group_size);
    } else {
        throw std::runtime_error("Unsupported tensor dtype for quantization");
    }
    
    QuantizedWeight result;
    result.compressed = compressed;
    result.scale = scale;
    result.zero_point = zero_point;
    result.compressed_type = ov::element::u8;
    result.has_zero_point = true;
    return result;
}

}  // namespace rtn
}  // namespace genai
}  // namespace ov

// Include FP8 support in a separate section to keep the file organized
#include "openvino/core/type/float8_e4m3.hpp"

namespace ov {
namespace genai {
namespace rtn {

/**
 * @brief Convert FP8 E4M3 value to float, applying scale
 * This is used for direct FP8 -> INT4/INT8 quantization
 */
inline float fp8_to_float_scaled(uint8_t fp8_val, float scale) {
    return static_cast<float>(ov::float8_e4m3::from_bits(fp8_val)) * scale;
}

/**
 * @brief Quantize FP8 weights directly to INT4 symmetric without intermediate F32 tensor
 * 
 * This optimized version processes FP8 weights in groups, converting to F32 only within
 * each group's scope to minimize memory usage. The scale from FP8 quantization is applied
 * during the conversion.
 * 
 * Memory optimization: Only allocates group_size floats at a time instead of full tensor
 * 
 * @param weight_fp8 Input FP8 weight tensor
 * @param scale_inv FP8 scale tensor (per-channel or block-wise)
 * @param group_size Group size for INT4 quantization
 * @return QuantizedWeight with INT4 weights packed as U8, FP16 scales
 */
inline QuantizedWeight quantize_fp8_to_int4_sym(
    const ov::Tensor& weight_fp8, 
    const ov::Tensor& scale_inv, 
    int group_size = 128
) {
    auto shape = weight_fp8.get_shape();
    auto scale_shape = scale_inv.get_shape();
    
    if (shape.size() != 2) {
        throw std::runtime_error("FP8 weight tensor must be 2D for direct quantization");
    }
    
    size_t out_features = shape[0];
    size_t in_features = shape[1];
    
    // FP8 scale can be [out_features, num_fp8_groups] or [out_features, 1]
    size_t fp8_num_groups = (scale_shape.size() >= 2) ? scale_shape[1] : 1;
    size_t fp8_group_size = in_features / fp8_num_groups;
    
    if (group_size <= 0) {
        group_size = static_cast<int>(in_features);
    }
    
    size_t num_groups = (in_features + group_size - 1) / group_size;
    size_t packed_in_features = (in_features + 1) / 2;
    
    ov::Tensor compressed(ov::element::u8, {out_features, packed_in_features});
    ov::Tensor scale(ov::element::f16, {out_features, num_groups});
    
    auto* compressed_data = static_cast<uint8_t*>(compressed.data());
    auto* scale_data = static_cast<uint16_t*>(scale.data());
    const uint8_t* fp8_ptr = static_cast<const uint8_t*>(weight_fp8.data());
    const float* fp8_scale_ptr = scale_inv.data<float>();
    
    std::memset(compressed_data, 0, out_features * packed_in_features);
    
    constexpr float factor = 8.0f;  // 2^(4-1) = 8
    constexpr int8_t level_low = -8;
    constexpr int8_t level_high = 7;
    constexpr float eps = 1.1920929e-7f;
    
    ov::parallel_for(out_features, [&](size_t row) {
        const uint8_t* row_fp8 = fp8_ptr + row * in_features;
        uint8_t* row_compressed = compressed_data + row * packed_in_features;
        uint16_t* row_scale = scale_data + row * num_groups;
        const float* row_fp8_scale = fp8_scale_ptr + row * fp8_num_groups;
        
        for (size_t g = 0; g < num_groups; g++) {
            size_t group_start = g * group_size;
            size_t group_end = std::min(group_start + static_cast<size_t>(group_size), in_features);
            size_t actual_group_size = group_end - group_start;
            
            // Find min/max by converting FP8 to F32 on-the-fly
            float w_min = std::numeric_limits<float>::max();
            float w_max = std::numeric_limits<float>::lowest();
            
            for (size_t i = group_start; i < group_end; ++i) {
                size_t fp8_group_idx = i / fp8_group_size;
                float fp8_scale = row_fp8_scale[fp8_group_idx];
                float val = fp8_to_float_scaled(row_fp8[i], fp8_scale);
                w_min = std::min(w_min, val);
                w_max = std::max(w_max, val);
            }
            
            // Compute scale for INT4
            float max_abs = std::max(std::abs(w_min), std::abs(w_max));
            float scale_val = max_abs / factor;
            if (scale_val < eps) scale_val = eps;
            float inv_scale = 1.0f / scale_val;
            
            row_scale[g] = float_to_fp16(scale_val);
            
            // Quantize
            for (size_t i = group_start; i < group_end; ++i) {
                size_t fp8_group_idx = i / fp8_group_size;
                float fp8_scale = row_fp8_scale[fp8_group_idx];
                float val = fp8_to_float_scaled(row_fp8[i], fp8_scale);
                
                int8_t q = static_cast<int8_t>(std::round(val * inv_scale));
                q = std::max(level_low, std::min(level_high, q));
                
                size_t byte_idx = i / 2;
                if (i % 2 == 0) {
                    row_compressed[byte_idx] = static_cast<uint8_t>(q) & 0x0F;
                } else {
                    row_compressed[byte_idx] |= (static_cast<uint8_t>(q) & 0x0F) << 4;
                }
            }
        }
    });
    
    QuantizedWeight result;
    result.compressed = compressed;
    result.scale = scale;
    result.has_zero_point = false;
    result.compressed_type = ov::element::i4;
    return result;
}

/**
 * @brief Quantize FP8 weights directly to INT4 asymmetric without intermediate F32 tensor
 */
inline QuantizedWeight quantize_fp8_to_int4_asym(
    const ov::Tensor& weight_fp8, 
    const ov::Tensor& scale_inv, 
    int group_size = 128
) {
    auto shape = weight_fp8.get_shape();
    auto scale_shape = scale_inv.get_shape();
    
    if (shape.size() != 2) {
        throw std::runtime_error("FP8 weight tensor must be 2D for direct quantization");
    }
    
    size_t out_features = shape[0];
    size_t in_features = shape[1];
    
    size_t fp8_num_groups = (scale_shape.size() >= 2) ? scale_shape[1] : 1;
    size_t fp8_group_size = in_features / fp8_num_groups;
    
    if (group_size <= 0) {
        group_size = static_cast<int>(in_features);
    }
    
    size_t num_groups = (in_features + group_size - 1) / group_size;
    size_t packed_in_features = (in_features + 1) / 2;
    
    ov::Tensor compressed(ov::element::u8, {out_features, packed_in_features});
    ov::Tensor scale(ov::element::f16, {out_features, num_groups});
    ov::Tensor zero_point(ov::element::u8, {out_features, num_groups});
    
    auto* compressed_data = static_cast<uint8_t*>(compressed.data());
    auto* scale_data = static_cast<uint16_t*>(scale.data());
    auto* zp_data = static_cast<uint8_t*>(zero_point.data());
    const uint8_t* fp8_ptr = static_cast<const uint8_t*>(weight_fp8.data());
    const float* fp8_scale_ptr = scale_inv.data<float>();
    
    std::memset(compressed_data, 0, out_features * packed_in_features);
    
    constexpr float levels = 15.0f;
    constexpr int32_t level_low = 0;
    constexpr int32_t level_high = 15;
    constexpr float eps = 1.1920929e-7f;
    
    ov::parallel_for(out_features, [&](size_t row) {
        const uint8_t* row_fp8 = fp8_ptr + row * in_features;
        uint8_t* row_compressed = compressed_data + row * packed_in_features;
        uint16_t* row_scale = scale_data + row * num_groups;
        uint8_t* row_zp = zp_data + row * num_groups;
        const float* row_fp8_scale = fp8_scale_ptr + row * fp8_num_groups;
        
        for (size_t g = 0; g < num_groups; g++) {
            size_t group_start = g * group_size;
            size_t group_end = std::min(group_start + static_cast<size_t>(group_size), in_features);
            
            float w_min = std::numeric_limits<float>::max();
            float w_max = std::numeric_limits<float>::lowest();
            
            for (size_t i = group_start; i < group_end; ++i) {
                size_t fp8_group_idx = i / fp8_group_size;
                float fp8_scale = row_fp8_scale[fp8_group_idx];
                float val = fp8_to_float_scaled(row_fp8[i], fp8_scale);
                w_min = std::min(w_min, val);
                w_max = std::max(w_max, val);
            }
            
            float scale_val = (w_max - w_min) / levels;
            if (scale_val < eps) scale_val = eps;
            float inv_scale = 1.0f / scale_val;
            
            int32_t zp = static_cast<int32_t>(std::round(-w_min * inv_scale));
            zp = std::max(level_low, std::min(level_high, zp));
            
            row_scale[g] = float_to_fp16(scale_val);
            row_zp[g] = static_cast<uint8_t>(zp);
            
            for (size_t i = group_start; i < group_end; ++i) {
                size_t fp8_group_idx = i / fp8_group_size;
                float fp8_scale = row_fp8_scale[fp8_group_idx];
                float val = fp8_to_float_scaled(row_fp8[i], fp8_scale);
                
                int32_t q = static_cast<int32_t>(std::round(val * inv_scale)) + zp;
                q = std::max(level_low, std::min(level_high, q));
                
                size_t byte_idx = i / 2;
                if (i % 2 == 0) {
                    row_compressed[byte_idx] = static_cast<uint8_t>(q) & 0x0F;
                } else {
                    row_compressed[byte_idx] |= (static_cast<uint8_t>(q) & 0x0F) << 4;
                }
            }
        }
    });
    
    QuantizedWeight result;
    result.compressed = compressed;
    result.scale = scale;
    result.zero_point = zero_point;
    result.has_zero_point = true;
    result.compressed_type = ov::element::u4;
    return result;
}

/**
 * @brief Quantize FP8 weights directly to INT8 symmetric without intermediate F32 tensor
 */
inline QuantizedWeight quantize_fp8_to_int8_sym(
    const ov::Tensor& weight_fp8, 
    const ov::Tensor& scale_inv, 
    int group_size = 128
) {
    auto shape = weight_fp8.get_shape();
    auto scale_shape = scale_inv.get_shape();
    
    if (shape.size() != 2) {
        throw std::runtime_error("FP8 weight tensor must be 2D for direct quantization");
    }
    
    size_t out_features = shape[0];
    size_t in_features = shape[1];
    
    size_t fp8_num_groups = (scale_shape.size() >= 2) ? scale_shape[1] : 1;
    size_t fp8_group_size = in_features / fp8_num_groups;
    
    if (group_size <= 0) {
        group_size = static_cast<int>(in_features);
    }
    
    size_t num_groups = (in_features + group_size - 1) / group_size;
    
    ov::Tensor compressed(ov::element::i8, {out_features, in_features});
    ov::Tensor scale(ov::element::f16, {out_features, num_groups});
    
    auto* compressed_data = static_cast<int8_t*>(compressed.data());
    auto* scale_data = static_cast<uint16_t*>(scale.data());
    const uint8_t* fp8_ptr = static_cast<const uint8_t*>(weight_fp8.data());
    const float* fp8_scale_ptr = scale_inv.data<float>();
    
    constexpr float factor = 128.0f;
    constexpr int8_t level_low = -128;
    constexpr int8_t level_high = 127;
    constexpr float eps = 1.1920929e-7f;
    
    ov::parallel_for(out_features, [&](size_t row) {
        const uint8_t* row_fp8 = fp8_ptr + row * in_features;
        int8_t* row_compressed = compressed_data + row * in_features;
        uint16_t* row_scale = scale_data + row * num_groups;
        const float* row_fp8_scale = fp8_scale_ptr + row * fp8_num_groups;
        
        for (size_t g = 0; g < num_groups; g++) {
            size_t group_start = g * group_size;
            size_t group_end = std::min(group_start + static_cast<size_t>(group_size), in_features);
            
            float w_min = std::numeric_limits<float>::max();
            float w_max = std::numeric_limits<float>::lowest();
            
            for (size_t i = group_start; i < group_end; ++i) {
                size_t fp8_group_idx = i / fp8_group_size;
                float fp8_scale = row_fp8_scale[fp8_group_idx];
                float val = fp8_to_float_scaled(row_fp8[i], fp8_scale);
                w_min = std::min(w_min, val);
                w_max = std::max(w_max, val);
            }
            
            float max_abs = std::max(std::abs(w_min), std::abs(w_max));
            float scale_val = max_abs / factor;
            if (scale_val < eps) scale_val = eps;
            float inv_scale = 1.0f / scale_val;
            
            row_scale[g] = float_to_fp16(scale_val);
            
            for (size_t i = group_start; i < group_end; ++i) {
                size_t fp8_group_idx = i / fp8_group_size;
                float fp8_scale = row_fp8_scale[fp8_group_idx];
                float val = fp8_to_float_scaled(row_fp8[i], fp8_scale);
                
                int32_t q = static_cast<int32_t>(std::round(val * inv_scale));
                q = std::max(static_cast<int32_t>(level_low), std::min(static_cast<int32_t>(level_high), q));
                row_compressed[i] = static_cast<int8_t>(q);
            }
        }
    });
    
    QuantizedWeight result;
    result.compressed = compressed;
    result.scale = scale;
    result.has_zero_point = false;
    result.compressed_type = ov::element::i8;
    return result;
}

/**
 * @brief Quantize FP8 weights directly to INT8 asymmetric without intermediate F32 tensor
 */
inline QuantizedWeight quantize_fp8_to_int8_asym(
    const ov::Tensor& weight_fp8, 
    const ov::Tensor& scale_inv, 
    int group_size = 128
) {
    auto shape = weight_fp8.get_shape();
    auto scale_shape = scale_inv.get_shape();
    
    if (shape.size() != 2) {
        throw std::runtime_error("FP8 weight tensor must be 2D for direct quantization");
    }
    
    size_t out_features = shape[0];
    size_t in_features = shape[1];
    
    size_t fp8_num_groups = (scale_shape.size() >= 2) ? scale_shape[1] : 1;
    size_t fp8_group_size = in_features / fp8_num_groups;
    
    if (group_size <= 0) {
        group_size = static_cast<int>(in_features);
    }
    
    size_t num_groups = (in_features + group_size - 1) / group_size;
    
    ov::Tensor compressed(ov::element::u8, {out_features, in_features});
    ov::Tensor scale(ov::element::f16, {out_features, num_groups});
    ov::Tensor zero_point(ov::element::u8, {out_features, num_groups});
    
    auto* compressed_data = static_cast<uint8_t*>(compressed.data());
    auto* scale_data = static_cast<uint16_t*>(scale.data());
    auto* zp_data = static_cast<uint8_t*>(zero_point.data());
    const uint8_t* fp8_ptr = static_cast<const uint8_t*>(weight_fp8.data());
    const float* fp8_scale_ptr = scale_inv.data<float>();
    
    constexpr float levels = 255.0f;
    constexpr int32_t level_low = 0;
    constexpr int32_t level_high = 255;
    constexpr float eps = 1.1920929e-7f;
    
    ov::parallel_for(out_features, [&](size_t row) {
        const uint8_t* row_fp8 = fp8_ptr + row * in_features;
        uint8_t* row_compressed = compressed_data + row * in_features;
        uint16_t* row_scale = scale_data + row * num_groups;
        uint8_t* row_zp = zp_data + row * num_groups;
        const float* row_fp8_scale = fp8_scale_ptr + row * fp8_num_groups;
        
        for (size_t g = 0; g < num_groups; g++) {
            size_t group_start = g * group_size;
            size_t group_end = std::min(group_start + static_cast<size_t>(group_size), in_features);
            
            float w_min = std::numeric_limits<float>::max();
            float w_max = std::numeric_limits<float>::lowest();
            
            for (size_t i = group_start; i < group_end; ++i) {
                size_t fp8_group_idx = i / fp8_group_size;
                float fp8_scale = row_fp8_scale[fp8_group_idx];
                float val = fp8_to_float_scaled(row_fp8[i], fp8_scale);
                w_min = std::min(w_min, val);
                w_max = std::max(w_max, val);
            }
            
            float scale_val = (w_max - w_min) / levels;
            if (scale_val < eps) scale_val = eps;
            float inv_scale = 1.0f / scale_val;
            
            int32_t zp = static_cast<int32_t>(std::round(-w_min * inv_scale));
            zp = std::max(level_low, std::min(level_high, zp));
            
            row_scale[g] = float_to_fp16(scale_val);
            row_zp[g] = static_cast<uint8_t>(zp);
            
            for (size_t i = group_start; i < group_end; ++i) {
                size_t fp8_group_idx = i / fp8_group_size;
                float fp8_scale = row_fp8_scale[fp8_group_idx];
                float val = fp8_to_float_scaled(row_fp8[i], fp8_scale);
                
                int32_t q = static_cast<int32_t>(std::round(val * inv_scale)) + zp;
                q = std::max(level_low, std::min(level_high, q));
                row_compressed[i] = static_cast<uint8_t>(q);
            }
        }
    });
    
    QuantizedWeight result;
    result.compressed = compressed;
    result.scale = scale;
    result.zero_point = zero_point;
    result.has_zero_point = true;
    result.compressed_type = ov::element::u8;
    return result;
}

}  // namespace rtn
}  // namespace genai
}  // namespace ov
