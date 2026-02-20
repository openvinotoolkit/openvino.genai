// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <sstream>

#include "openvino/core/except.hpp"

namespace ov::genai {

class TaylorSeerCacheConfig {
public:
    /**
     * @brief Constructs a TaylorSeerCacheConfig with the specified parameters.
     * @param cache_interval_ The interval between full computation steps. After a full computation,
     *        cached (predicted) outputs are reused for this many subsequent denoising steps before
     *        refreshing with a new full forward pass. Must be at least 2.
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
          disable_cache_after_step(disable_cache_after_step_) {
        if (cache_interval < 2) {
            OPENVINO_THROW("TaylorSeerCacheConfig: cache_interval must be at least 2, got ",
                           cache_interval);
        }
    }

    /**
     * @brief Validates that the configuration is compatible with the number of inference steps.
     * @param num_inference_steps The total number of inference steps.
     */
    void validate(std::size_t num_inference_steps) const {
        if (disable_cache_before_step > num_inference_steps) {
            OPENVINO_THROW("TaylorSeerCacheConfig: disable_cache_before_step (",
                           disable_cache_before_step,
                           ") cannot be greater than num_inference_steps (",
                           num_inference_steps, ")");
        }

        if (disable_cache_after_step < 0) {
            std::size_t abs_disable_after = static_cast<std::size_t>(-disable_cache_after_step);
            if (abs_disable_after > num_inference_steps) {
                OPENVINO_THROW("TaylorSeerCacheConfig: abs(disable_cache_after_step) (",
                               abs_disable_after,
                               ") cannot be greater than num_inference_steps (",
                               num_inference_steps, ")");
            }
        }
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

    /** The interval between full computation steps */
    std::size_t cache_interval = 3;

    /** The denoising step index before which caching is disabled */
    std::size_t disable_cache_before_step = 6;

    /** The denoising step index after which caching is disabled.
     *  If negative, calculated as num_inference_steps + disable_cache_after_step */
    int disable_cache_after_step = -2;
};

} // namespace ov::genai
