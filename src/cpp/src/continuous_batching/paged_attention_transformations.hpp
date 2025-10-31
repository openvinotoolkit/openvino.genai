// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace genai {

namespace utils {

void apply_gather_before_matmul_transformation(std::shared_ptr<ov::Model> model);

}  // namespace utils
}  // namespace genai
}  // namespace ov
