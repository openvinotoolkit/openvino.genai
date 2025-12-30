// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "openvino/runtime/tensor.hpp"
#include <vector>

namespace ov::genai::module::tensor_utils {

ov::Tensor squeeze(const ov::Tensor& tensor, size_t dim);

ov::Tensor unsqueeze(const ov::Tensor& tensor, size_t dim);

std::vector<ov::Tensor> split(const ov::Tensor& tensor);

ov::Tensor stack(const std::vector<ov::Tensor>& tensors);

}