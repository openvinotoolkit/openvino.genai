// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "openvino/runtime/tensor.hpp"

using GGUFMetaData =
    std::variant<std::monostate, float, int, ov::Tensor, std::string, std::vector<std::string>, std::vector<int32_t>>;
