// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/core/core.hpp"
#include "openvino/runtime/infer_request.hpp"

namespace ov::genai {

class FunASREncoder {
public:
    FunASREncoder(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties);
    ov::Tensor encode(const ov::Tensor& features);

private:
    ov::InferRequest m_request;
};

}  // namespace ov::genai
