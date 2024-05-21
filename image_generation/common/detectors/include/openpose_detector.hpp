// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tensor.hpp"

class OpenposeDetector {
public:
    OpenposeDetector() = default;
    void Load(const std::string& model_path);
    int foo();

private:
    ov::CompiledModel body_model;
};