// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/visibility.hpp"
#include <openvino/runtime/compiled_model.hpp>
#include <array>
#include <filesystem>

namespace ov::genai {
class OPENVINO_GENAI_EXPORTS ProcessorConfig {
public:
    size_t patch_size = 14;
    size_t scale_resolution = 448;
    /// @brief 0 disables slicing.
    size_t max_slice_nums = 0;
    std::array<float, 3> norm_mean{0.0f, 0.0f, 0.0f};
    std::array<float, 3> norm_std{1.0f, 1.0f, 1.0f};
    ProcessorConfig() = default;
    explicit ProcessorConfig(const std::filesystem::path& json_path);
    ProcessorConfig(const ProcessorConfig&) = default;
};

/*
 * Utils that allow to use encode(), generate() and operator()() in the following way:
 * pipe.generate(input_ids, ov::genai::scale_resolution(448), ...)
 * pipe(input_ids, ov::genai::scale_resolution(448), ...)
*/
static constexpr ov::Property<size_t> patch_size{"patch_size"};
static constexpr ov::Property<size_t> scale_resolution{"scale_resolution"};
static constexpr ov::Property<size_t> max_slice_nums{"max_slice_nums"};
static constexpr ov::Property<std::array<float, 3>> norm_mean{"norm_mean"};
static constexpr ov::Property<std::array<float, 3>> norm_std{"norm_std"};
}  // namespace ov::genai
