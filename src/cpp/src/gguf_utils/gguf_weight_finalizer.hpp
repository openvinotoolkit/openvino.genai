// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>

#include <openvino/openvino.hpp>

#include "gguf_utils/gguf.hpp"
#include "modeling/ops/context.hpp"
#include "modeling/weights/weight_finalizer.hpp"

namespace ov {
namespace genai {
namespace gguf {

class GGUFWeightFinalizer : public ov::genai::modeling::weights::WeightFinalizer {
public:
    GGUFWeightFinalizer(const std::unordered_map<std::string, ov::Tensor>& consts,
                        const std::unordered_map<std::string, gguf_tensor_type>& qtypes);

    ov::genai::modeling::weights::FinalizedWeight finalize(const std::string& name,
                                                           ov::genai::modeling::weights::WeightSource& source,
                                                           ov::genai::modeling::OpContext& ctx) override;

private:
    gguf_tensor_type resolve_qtype(const std::string& base_key) const;
    std::string base_key_from_name(const std::string& name) const;

    const std::unordered_map<std::string, ov::Tensor>& consts_;
    const std::unordered_map<std::string, gguf_tensor_type>& qtypes_;
    std::unordered_map<std::string, ov::Output<ov::Node>> cache_;
};

}  // namespace gguf
}  // namespace genai
}  // namespace ov
