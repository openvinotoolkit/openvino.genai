
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <openvino/runtime/tensor.hpp>

namespace utils {
ov::Tensor load_image(const std::filesystem::path& image_path);
std::vector<ov::Tensor> load_images(const std::filesystem::path& image_path);
}  // namespace utils
