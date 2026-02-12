// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace ov {
namespace genai {

class ASRPipelineImplBase {
public:
    ASRPipelineImplBase(const std::filesystem::path& model_dir,
                        const std::string& device,
                        const ov::AnyMap& properties = {}) {};
    virtual ~ASRPipelineImplBase() = default;
};

} // namespace genai
} // namespace ov