// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace ov::genai::utils {

inline size_t get_spatio_temporal_compression_factor(const std::vector<bool>& spatio_temporal_scaling) {
    size_t compression_factor = 1;
    for (const bool use_scaling : spatio_temporal_scaling) {
        if (use_scaling) {
            compression_factor *= 2;
        }
    }
    return compression_factor;
}

inline std::pair<size_t, size_t> get_spatial_temporal_compression_ratios(
    size_t patch_size,
    size_t patch_size_t,
    const std::vector<bool>& spatio_temporal_scaling) {
    const size_t compression_factor = get_spatio_temporal_compression_factor(spatio_temporal_scaling);
    return {patch_size * compression_factor, patch_size_t * compression_factor};
}

inline size_t get_spatial_compression_ratio(
    size_t patch_size,
    const std::vector<bool>& spatio_temporal_scaling) {
    return patch_size * get_spatio_temporal_compression_factor(spatio_temporal_scaling);
}

} // namespace ov::genai::utils
