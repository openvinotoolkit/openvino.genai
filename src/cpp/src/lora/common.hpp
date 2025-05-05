// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <map>
#include <memory>
#include <string>
#include <optional>
#include <vector>

#include "openvino/op/constant.hpp"

namespace ov {
namespace genai {
namespace utils {

template <typename T>
struct LoRAParts {
    T alpha, A, B;

    LoRAParts() = default;
    LoRAParts(const T& alpha, const T& A, const T& B) : alpha(alpha), A(A), B(B) {}

    template <typename Other>
    LoRAParts(const LoRAParts<Other>& other) : alpha(other.alpha), A(other.A), B(other.B) {}
};


using LoRAWeight = LoRAParts<std::shared_ptr<ov::op::v0::Constant>>;
using LoRATensors = std::map<std::string, LoRAWeight>;

}
}
}
