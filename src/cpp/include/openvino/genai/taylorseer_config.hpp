// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace ov::genai {

class TaylorSeerCacheConfig {
public:
    /**
     * @brief Constructs a TaylorSeerCacheConfig with the specified parameters.
     * @param cache_interval_ The interval between full computation steps. After a full computation,
     *        cached (predicted) outputs are reused for this many subsequent denoising steps before
     *        refreshing with a new full forward pass.
     * @param disable_cache_before_step_ The denoising step index before which caching is disabled.
     *        Full computation is performed for initial steps (0 to disable_cache_before_step - 1)
     *        to gather data for Taylor series approximations. Caching begins at this step.
     * @param disable_cache_after_step_ The denoising step index after which caching is disabled.
     *        If negative, it is calculated as num_inference_steps + disable_cache_after_step.
     *        For steps >= this value, full computations are performed without predictions.
     */
    TaylorSeerCacheConfig(std::size_t cache_interval_ = 3,
                          std::size_t disable_cache_before_step_ = 6,
                          int disable_cache_after_step_ = -2)
        : cache_interval(cache_interval_),
          disable_cache_before_step(disable_cache_before_step_),
          disable_cache_after_step(disable_cache_after_step_) {}

    std::size_t get_cache_interval() const {
        return cache_interval;
    }

    std::size_t get_disable_cache_before_step() const {
        return disable_cache_before_step;
    }

    int get_disable_cache_after_step() const {
        return disable_cache_after_step;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "TaylorSeerCacheConfig {\n"
            << "  cache_interval: " << cache_interval << "\n"
            << "  disable_cache_before_step: " << disable_cache_before_step << "\n"
            << "  disable_cache_after_step: " << disable_cache_after_step << "\n"
            << "}";
        return oss.str();
    }

private:
    std::size_t cache_interval = 0;
    std::size_t disable_cache_before_step = 6;
    int disable_cache_after_step = -2;
};

} // namespace ov::genai