// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include <openvino/openvino.hpp>

namespace ov {
namespace genai {
namespace modeling {
namespace weights {

class WeightSource {
public:
    virtual ~WeightSource() = default;

    virtual std::vector<std::string> keys() const = 0;
    virtual bool has(const std::string& name) const = 0;
    virtual const ov::Tensor& get_tensor(const std::string& name) const = 0;

    // Optional memory-management hooks for streaming/large-weight sources.
    // Default implementation is no-op.
    virtual void release_tensor(const std::string& name) {
        (void)name;
    }

    virtual void release_all_cached_tensors() {}
};

}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov
