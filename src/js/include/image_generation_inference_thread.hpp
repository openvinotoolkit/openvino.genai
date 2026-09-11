// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>

#include "include/base/inference_thread.hpp"
#include "openvino/runtime/tensor.hpp"

// Builds the per-step callback passed to image-generation generate(). It bridges each diffusion step to the JS
// streamer ThreadSafeFunction and blocks until JS reports whether generation should stop. is_busy is released while
// the JS callback runs so it may call decode() between steps. Shared by text2image, image2image and inpainting.
std::function<bool(size_t, size_t, ov::Tensor&)> make_image_generation_step_callback(
    InferenceThreadContext* context,
    std::shared_ptr<std::atomic<bool>> is_busy);
