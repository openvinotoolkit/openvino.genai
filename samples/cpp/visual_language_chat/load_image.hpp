
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/runtime/tensor.hpp>
#include <filesystem>

namespace utils {
ov::Tensor load_image(const std::filesystem::path& image_path);
std::vector<ov::Tensor> load_images(const std::filesystem::path& image_path);
}
