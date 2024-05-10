// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>

namespace ov {
namespace generate_utils {

Tensor init_attention_mask(Tensor& position_ids);

void print_tensor(const ov::Tensor& tensor);

std::pair<int64_t, float> softmax(const ov::Tensor& logits, const size_t batch_idx);

bool is_xml(const std::string& path);

}  // namespace generate_utils
}  // namespace ov