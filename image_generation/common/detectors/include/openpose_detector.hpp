// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "base_detector.hpp"
#include "openvino/runtime/compiled_model.hpp"

class OpenposeDetector {
public:
    OpenposeDetector() = default;

    void load(const std::string&);
    ov::Tensor preprocess(ov::Tensor);
    std::pair<ov::Tensor, ov::Tensor> inference(ov::Tensor);
    void postprocess();

    // will be deleted
    void forward(const std::string&, unsigned long w, unsigned long h, unsigned long c);

private:
    ov::CompiledModel body_model;
};