
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/runtime/tensor.hpp>
#include <cstdint>
#include <optional>
#include <filesystem>

namespace utils {
ov::Tensor load_image(const std::filesystem::path& image_path, std::optional<int32_t> target_height = std::nullopt, std::optional<int32_t> target_width = std::nullopt);
std::vector<ov::Tensor> load_images(const std::filesystem::path& image_path, std::optional<int32_t> target_height = std::nullopt, std::optional<int32_t> target_width = std::nullopt);
}
