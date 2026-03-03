// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>

#include "modeling/weights/weight_source.hpp"

namespace ov {
namespace genai {
namespace gguf {

class GGUFWeightSource : public ov::genai::modeling::weights::WeightSource {
public:
    explicit GGUFWeightSource(const std::unordered_map<std::string, ov::Tensor>& consts);

    std::vector<std::string> keys() const override;
    bool has(const std::string& name) const override;
    const ov::Tensor& get_tensor(const std::string& name) const override;

private:
    const std::unordered_map<std::string, ov::Tensor>& consts_;
    std::vector<std::string> keys_;  // Canonical names
    std::unordered_map<std::string, std::string> canonical_to_gguf_;  // canonical -> gguf mapping
};

}  // namespace gguf
}  // namespace genai
}  // namespace ov
