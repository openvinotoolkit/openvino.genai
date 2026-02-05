// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace genai {
namespace utils {
namespace eagle3 {

/**
 * @brief Extracts hidden state at a specific position from hidden features tensor.
 *
 * This function creates a view of the hidden state tensor containing only the features
 * corresponding to the specified position. Used in DRAFT_ITERATION to extract position-specific
 * hidden states for each sequence in multi-sequence tree search.
 *
 * @param hidden_features Hidden state tensor with shape [1, seq_len, hidden_size].
 * @param position Position index to extract (0-based).
 * @return Tensor view containing only the specified position: [1, 1, hidden_size].
 * @throws Exception if tensor is empty, shape is invalid, or position is out of bounds.
 */
ov::Tensor slice_hidden_state_at_position(const ov::Tensor& hidden_features, size_t position);

/**
 * @brief Concatenates multiple hidden state tensors along the sequence dimension.
 *
 * This function concatenates a vector of hidden state tensors along dimension 1 (seq_len).
 * All input tensors must have compatible shapes [1, seq_len_i, hidden_size] and the same
 * data type. The result has shape [1, sum(seq_len_i), hidden_size].
 *
 * Used in DRAFT_ITERATION to combine hidden states from multiple sequences in multi-sequence
 * tree search, typically in layer-first interleaved order.
 *
 * @param hidden_states Vector of hidden state tensors to concatenate.
 * @return Concatenated tensor with shape [1, total_seq_len, hidden_size].
 * @throws Exception if input is empty, shapes are incompatible, or data types differ.
 */
ov::Tensor concatenate_hidden_states(const std::vector<ov::Tensor>& hidden_states);

}  // namespace eagle3
}  // namespace utils
}  // namespace genai
}  // namespace ov
