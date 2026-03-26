// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <sstream>
#include <string>

#include "openvino/core/except.hpp"

namespace ov::genai {

class TaylorSeerCacheConfig {
public:
    std::string to_string() const {
        std::ostringstream oss;
        oss << "TaylorSeerCacheConfig {\n"
            << "  cache_interval: " << cache_interval << "\n"
            << "  disable_cache_before_step: " << disable_cache_before_step << "\n"
            << "  disable_cache_after_step: " << disable_cache_after_step << "\n"
            << "}";
        return oss.str();
    }

    /** The interval between full computation steps. Must be at least 2. */
    std::size_t cache_interval = 3;

    /** The denoising step index before which caching is disabled */
    std::size_t disable_cache_before_step = 6;

    /** The denoising step index after which caching is disabled.
     *  If negative, calculated as num_inference_steps + disable_cache_after_step */
    int disable_cache_after_step = -2;
};

}  // namespace ov::genai
