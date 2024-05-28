// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "base_detector.hpp"
#include "openvino/runtime/compiled_model.hpp"

class OpenposeDetector {
public:
    OpenposeDetector() = default;

    void load(const std::string&);
    void preprocess();
    void inference(const std::string&);
    void postprocess();

    void load_bgr(const std::string&, unsigned long w, unsigned long h, unsigned long c);

    int foo();

private:
    ov::CompiledModel body_model;
};