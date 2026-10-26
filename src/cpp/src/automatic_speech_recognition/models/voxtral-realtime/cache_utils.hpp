// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "openvino/openvino.hpp"

namespace ov::genai::voxtral_realtime {
namespace detail {

template <typename T>
void rephase_key_token(const T* source,
                       T* destination,
                       const size_t head_dim,
                       const std::vector<double>& cosines,
                       const std::vector<double>& sines) {
    const size_t head_dim_half = head_dim / 2;
    for (size_t index = 0; index < head_dim_half; ++index) {
        const double first = static_cast<double>(source[index]);
        const double second = static_cast<double>(source[index + head_dim_half]);
        destination[index] = static_cast<T>(first * cosines[index] - second * sines[index]);
        destination[index + head_dim_half] =
            static_cast<T>(second * cosines[index] + first * sines[index]);
    }
}

template <typename T>
std::pair<ov::Tensor, ov::Tensor> crop_and_rephase_cache(const ov::Tensor& key,
                                                          const ov::Tensor& value,
                                                          const size_t max_context,
                                                          const float rope_theta) {
    const ov::Shape& shape = key.get_shape();
    const size_t batch_size = shape[0];
    const size_t num_heads = shape[1];
    const size_t sequence_length = shape[2];
    const size_t head_dim = shape[3];
    const size_t evicted_tokens = sequence_length - max_context;
    const size_t tokens_per_head = max_context;
    const size_t head_dim_half = head_dim / 2;
    const ov::Shape retained_shape{batch_size, num_heads, tokens_per_head, head_dim};
    ov::Tensor retained_key(key.get_element_type(), retained_shape);
    ov::Tensor retained_value(value.get_element_type(), retained_shape);
    const T* key_data = key.data<const T>();
    const T* value_data = value.data<const T>();
    T* retained_key_data = retained_key.data<T>();
    T* retained_value_data = retained_value.data<T>();

    std::vector<double> cosines(head_dim_half);
    std::vector<double> sines(head_dim_half);
    for (size_t index = 0; index < head_dim_half; ++index) {
        const double frequency = std::pow(static_cast<double>(rope_theta),
                                          -2.0 * static_cast<double>(index) / static_cast<double>(head_dim));
        // Rebase retained keys with cache positions so their RoPE-relative distances remain unchanged.
        const double angle = -static_cast<double>(evicted_tokens) * frequency;
        cosines[index] = std::cos(angle);
        sines[index] = std::sin(angle);
    }

    const size_t retained_token_count = batch_size * num_heads * tokens_per_head;
    for (size_t output_token = 0; output_token < retained_token_count; ++output_token) {
        const size_t head_index = output_token / tokens_per_head;
        const size_t token_index = output_token % tokens_per_head + evicted_tokens;
        const size_t source_offset = (head_index * sequence_length + token_index) * head_dim;
        const size_t retained_offset = output_token * head_dim;
        rephase_key_token(key_data + source_offset,
                          retained_key_data + retained_offset,
                          head_dim,
                          cosines,
                          sines);
        std::copy_n(value_data + source_offset, head_dim, retained_value_data + retained_offset);
    }
    return {std::move(retained_key), std::move(retained_value)};
}

}  // namespace detail

inline std::pair<ov::Tensor, ov::Tensor> crop_and_rephase_cache(const ov::Tensor& key,
                                                                 const ov::Tensor& value,
                                                                 const size_t max_context,
                                                                 const float rope_theta) {
    OPENVINO_ASSERT(key.get_shape().size() == 4 && key.get_shape() == value.get_shape(),
                    "Voxtral key/value caches must have matching [batch, heads, sequence, head_dim] shapes.");
    OPENVINO_ASSERT(max_context > 0, "Voxtral sliding-window context must be positive.");
    OPENVINO_ASSERT(key.get_element_type() == value.get_element_type(),
                    "Voxtral key/value cache element types must match.");
    if (key.get_shape()[2] <= max_context) {
        ov::Tensor key_copy(key.get_element_type(), key.get_shape());
        ov::Tensor value_copy(value.get_element_type(), value.get_shape());
        key.copy_to(key_copy);
        value.copy_to(value_copy);
        return {std::move(key_copy), std::move(value_copy)};
    }

    OPENVINO_ASSERT(key.get_shape()[3] > 0 && key.get_shape()[3] % 2 == 0,
                    "Voxtral RoPE head dimension must be positive and even.");
    OPENVINO_ASSERT(rope_theta > 0.0f, "Voxtral RoPE theta must be positive.");
    if (key.get_element_type() == ov::element::f32)
        return detail::crop_and_rephase_cache<float>(key, value, max_context, rope_theta);
    if (key.get_element_type() == ov::element::f16)
        return detail::crop_and_rephase_cache<ov::float16>(key, value, max_context, rope_theta);
    if (key.get_element_type() == ov::element::bf16)
        return detail::crop_and_rephase_cache<ov::bfloat16>(key, value, max_context, rope_theta);
    OPENVINO_THROW("Unsupported Voxtral KV-cache element type: ", key.get_element_type());
}
}  // namespace ov::genai::voxtral_realtime
