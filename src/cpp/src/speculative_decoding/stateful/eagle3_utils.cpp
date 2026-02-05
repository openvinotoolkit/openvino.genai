// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "eagle3_utils.hpp"

#include <cstring>
#include "openvino/core/except.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {
namespace utils {
namespace eagle3 {

ov::Tensor slice_hidden_state_at_position(const ov::Tensor& hidden_features, size_t position) {
    OPENVINO_ASSERT(hidden_features.get_size() > 0, "Hidden features tensor is empty");

    const auto shape = hidden_features.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1 && shape[1] > 0, 
                    "Expected shape [1, seq_len, hidden_size]");
    OPENVINO_ASSERT(position < shape[1], 
                    "Position ", position, " out of bounds for seq_len ", shape[1]);

    auto [start_coord, end_coord] = ov::genai::utils::make_roi(shape, 1, position, position + 1);
    return ov::Tensor(hidden_features, start_coord, end_coord);
}

ov::Tensor concatenate_hidden_states(const std::vector<ov::Tensor>& hidden_states) {
    OPENVINO_ASSERT(!hidden_states.empty(), "Cannot concatenate empty vector of hidden states");

    if (hidden_states.size() == 1) {
        return hidden_states[0];  // No concatenation needed
    }

    // Validate all tensors and compute total sequence length
    const auto& first_shape = hidden_states[0].get_shape();
    const auto element_type = hidden_states[0].get_element_type();
    
    OPENVINO_ASSERT(first_shape.size() == 3 && first_shape[0] == 1, 
                    "Expected shape [1, seq_len, hidden_size]");
    
    const size_t hidden_size = first_shape[2];
    size_t total_seq_len = 0;

    for (const auto& tensor : hidden_states) {
        const auto& shape = tensor.get_shape();
        OPENVINO_ASSERT(shape.size() == 3 && shape[0] == 1 && shape[2] == hidden_size,
                        "All tensors must have compatible shapes [1, *, ", hidden_size, "]");
        OPENVINO_ASSERT(tensor.get_element_type() == element_type,
                        "All tensors must have the same element type");
        total_seq_len += shape[1];
    }

    // Create output tensor
    ov::Shape output_shape = {1, total_seq_len, hidden_size};
    ov::Tensor result(element_type, output_shape);

    // Copy data
    size_t current_offset = 0;
    for (const auto& tensor : hidden_states) {
        const auto& shape = tensor.get_shape();
        const size_t seq_len = shape[1];
        const size_t copy_size = seq_len * hidden_size * element_type.size();
        
        std::memcpy(
            static_cast<char*>(result.data()) + current_offset * hidden_size * element_type.size(),
            tensor.data(),
            copy_size
        );
        
        current_offset += seq_len;
    }

    return result;
}

}  // namespace eagle3
}  // namespace utils
}  // namespace genai
}  // namespace ov
