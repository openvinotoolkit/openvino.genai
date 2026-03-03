// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <openvino/openvino.hpp>

#include "modeling/weights/weight_source.hpp"

namespace ov {
namespace genai {
namespace modeling {
namespace weights {

struct SyntheticWeightSpec {
    std::string name;
    ov::Shape shape;
    ov::element::Type dtype = ov::element::f32;
};

class SyntheticWeightSource : public WeightSource {
public:
    SyntheticWeightSource(std::vector<SyntheticWeightSpec> specs,
                          uint32_t seed = 2026,
                          float low = -0.02f,
                          float high = 0.02f);

    std::vector<std::string> keys() const override;
    bool has(const std::string& name) const override;
    const ov::Tensor& get_tensor(const std::string& name) const override;
    void release_tensor(const std::string& name) override;
    void release_all_cached_tensors() override;

private:
    ov::Tensor make_tensor(const SyntheticWeightSpec& spec, uint32_t local_seed) const;

    std::unordered_map<std::string, SyntheticWeightSpec> specs_;
    std::vector<std::string> keys_;
    uint32_t seed_ = 2026;
    float low_ = -0.02f;
    float high_ = 0.02f;
    mutable std::unordered_map<std::string, ov::Tensor> cache_;
};

}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov

