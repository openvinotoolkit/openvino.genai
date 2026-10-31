// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "include/base/inference_thread.hpp"
#include "openvino/runtime/tensor.hpp"

std::function<bool(size_t, size_t, ov::Tensor&)> make_image_generation_step_callback(
    InferenceThreadContext* context,
    std::shared_ptr<PipelineState> state);
