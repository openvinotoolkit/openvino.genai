// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/minicpmv4_x_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "openvino/core/except.hpp"

namespace ov::genai::minicpmv4_x_utils {

int ensure_divide(const double length, const int divisor) {
    return std::max(static_cast<int>(std::round(length / divisor)) * divisor, divisor);
}

ImageSize find_best_resize(
    const double height,
    const double width,
    const int scale_resolution,
    const int patch_size,
    const int merge_factor,
    const bool allow_upscale) {
    double new_height = height;
    double new_width = width;
    if (height * width > static_cast<double>(scale_resolution) * scale_resolution || allow_upscale) {
        const double aspect_ratio = width / height;
        new_height = scale_resolution / std::sqrt(aspect_ratio);
        new_width = new_height * aspect_ratio;
    }
    const int factor = patch_size * merge_factor;
    return {static_cast<size_t>(ensure_divide(new_height, factor)),
            static_cast<size_t>(ensure_divide(new_width, factor))};
}

ImageSize get_refine_size(
    const int height,
    const int width,
    const SlicesGrid& grid,
    const int scale_resolution,
    const int patch_size,
    const int merge_factor) {
    OPENVINO_ASSERT(grid.has_slices(), "Refining an image requires a non-empty slices grid");
    const int grid_rows = static_cast<int>(grid.rows);
    const int grid_cols = static_cast<int>(grid.cols);
    const int refine_height = ensure_divide(height, grid_rows);
    const int refine_width = ensure_divide(width, grid_cols);
    const ImageSize best = find_best_resize(
        static_cast<double>(refine_height) / grid_rows,
        static_cast<double>(refine_width) / grid_cols,
        scale_resolution,
        patch_size,
        merge_factor,
        /*allow_upscale=*/true);
    return {best.height * grid.rows, best.width * grid.cols};
}

SlicesGrid get_slices_grid(const int height, const int width, const int max_slice_nums, const int scale_resolution) {
    const double log_ratio = std::log(static_cast<double>(width) / height);
    const double ratio =
        static_cast<double>(width) * height / (static_cast<double>(scale_resolution) * scale_resolution);
    const int multiple = std::min(static_cast<int>(std::ceil(ratio)), max_slice_nums);
    if (multiple <= 1) {
        return {};
    }

    SlicesGrid best_grid{1, 1};
    double min_error = std::numeric_limits<double>::infinity();
    for (int num_slices : {multiple - 1, multiple, multiple + 1}) {
        if (num_slices == 1 || num_slices > max_slice_nums) {
            continue;
        }
        for (int num_rows = 1; num_rows <= num_slices; ++num_rows) {
            if (num_slices % num_rows != 0) {
                continue;
            }
            const int num_cols = num_slices / num_rows;
            const double error = std::abs(log_ratio - std::log(static_cast<double>(num_cols) / num_rows));
            if (error < min_error) {
                best_grid = {static_cast<size_t>(num_rows), static_cast<size_t>(num_cols)};
                min_error = error;
            } else if (error == min_error && static_cast<size_t>(num_rows) > best_grid.rows) {
                best_grid = {static_cast<size_t>(num_rows), static_cast<size_t>(num_cols)};
            }
        }
    }
    return best_grid;
}

} // namespace ov::genai::minicpmv4_x_utils
