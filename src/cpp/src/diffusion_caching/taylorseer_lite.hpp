// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <optional>
#include <unordered_map>
#include <sstream>

#include <openvino/core/except.hpp>
#include <openvino/runtime/tensor.hpp>
#include <openvino/genai/taylorseer_config.hpp>

namespace ov::genai {

/**
 * @brief State management class for TaylorSeer cache mechanism.
 *
 * Maintains Taylor series factors and tracks the last update step to enable
 * prediction of transformer outputs during inference.
 */
class TaylorSeerState {
public:
    TaylorSeerState() = default;

    /**
     * @brief Gets the step index when the Taylor factors were last updated.
     * @return The last update step index, or std::nullopt if never updated.
     */
    std::optional<std::size_t> get_last_update_step() const {
        return m_last_update_step;
    }

    /**
     * @brief Gets the Taylor factor for a specific order.
     * @param order The order of the Taylor factor to retrieve (0 for features, 1 for first derivative, etc.)
     * @return The Taylor factor tensor for the specified order.
     */
    const ov::Tensor& get_taylor_factor(std::size_t order) const {
        return m_taylor_factors.at(order);
    }

    /**
     * @brief Determines if a full computation should be performed at the current step.
     * @param current_step The current denoising step index.
     * @param config The TaylorSeer cache configuration.
     * @param num_inference_steps Total number of inference steps in the generation process.
     * @return true if full computation is required, false if cached prediction can be used.
     */
    bool should_compute(std::size_t current_step,
                       const TaylorSeerCacheConfig& config,
                       std::size_t num_inference_steps) const {
        // Always compute during warm-up phase
        if (current_step < config.get_disable_cache_before_step()) {
            return true;
        }

        int disable_cache_after_step = config.get_disable_cache_after_step();
        if (disable_cache_after_step < 0) {
            disable_cache_after_step = static_cast<int>(num_inference_steps) + disable_cache_after_step;
        }
        if (disable_cache_after_step > 0 && current_step >= static_cast<std::size_t>(disable_cache_after_step)) {
            return true;
        }

        auto offset = current_step - config.get_disable_cache_before_step();
        auto first_compute_offset = config.get_cache_interval() - 1;

        if (offset < first_compute_offset) {
            return false;  // Predict using cached values
        } else {
            // Compute at first_compute_offset, then every cache_interval steps
            return ((offset - first_compute_offset) % config.get_cache_interval()) == 0;
        }

        // return (current_step - config.get_disable_cache_before_step()) % config.get_cache_interval() == 0;
    }

    /**
     * @brief Updates Taylor factors with a new output from a full computation.
     * @param current_step The current denoising step index.
     * @param output The output tensor from the full computation.
     * @throws ov::Exception if current_step is not greater than the last update step.
     */
    void update(std::size_t current_step, const ov::Tensor& output) {
        const bool is_first_update = !m_last_update_step.has_value();

        OPENVINO_ASSERT(is_first_update || current_step > *m_last_update_step,
                       "Current step (", current_step,
                       ") must be greater than the last update step (",
                       *m_last_update_step, ") for TaylorSeerState update.");

        std::unordered_map<std::size_t, ov::Tensor> new_factors = {{0, output}};

        if (!is_first_update) {
            const float divisor = 1.0f / static_cast<float>(current_step - *m_last_update_step);

            // Compute higher-order Taylor factors using finite differences
            for (std::size_t order = 1; order < m_max_order; ++order) {
                const auto& curr_factor = new_factors[order - 1];
                const float* curr_data = curr_factor.data<const float>();

                const auto& prev_factor = get_taylor_factor(order - 1);
                const float* prev_data = prev_factor.data<const float>();

                ov::Tensor new_factor(curr_factor.get_element_type(), curr_factor.get_shape());
                float* factor_data = new_factor.data<float>();

                for (size_t i = 0; i < curr_factor.get_size(); ++i) {
                    factor_data[i] = (curr_data[i] - prev_data[i]) * divisor;
                }
                new_factors[order] = new_factor;
            }
        }

        // Update stored factors
        for (const auto& [order, new_taylor_factor]: new_factors) {
            m_taylor_factors[order] = new_taylor_factor;
        }

        m_last_update_step = current_step;
    }

    /**
     * @brief Predicts the output at the current step using Taylor series approximation.
     * @param current_step The current denoising step index.
     * @return The predicted output tensor.
     * @throws ov::Exception if Taylor factors are not yet available for prediction.
     */
    ov::Tensor predict(std::size_t current_step) const {
        OPENVINO_ASSERT(m_taylor_factors.size() >= m_max_order,
                       "Insufficient Taylor factors available for prediction. Required: ",
                       m_max_order, ", Available: ", m_taylor_factors.size());

        OPENVINO_ASSERT(m_last_update_step.has_value(),
                       "Cannot predict before first update.");

        OPENVINO_ASSERT(current_step > *m_last_update_step,
                       "Cannot predict for step ", current_step,
                       " as it is not after the last update step ",
                       *m_last_update_step);

        const std::size_t step_offset = current_step - *m_last_update_step;

        // Apply Taylor series: f(x) ≈ Σ (factor_i * (step_offset)^i)
        const ov::Tensor& base_factor = get_taylor_factor(0);
        ov::Tensor output(base_factor.get_element_type(), base_factor.get_shape());
        base_factor.copy_to(output);
        float* output_data = output.data<float>();
        for (std::size_t order = 1; order < m_max_order; ++order) {
            const float coeff = static_cast<float>(std::pow(step_offset, order));
            const float* factor_data = get_taylor_factor(order).data<const float>();

            for (size_t i = 0; i < output.get_size(); ++i) {
                output_data[i] += factor_data[i] * coeff;
            }
        }

        return output;
    }

private:
    /**
     * @brief Maps order to corresponding Taylor factor tensor.
     * Order 0 stores the base output, order 1 stores the first derivative approximation, etc.
     */
    std::unordered_map<std::size_t, ov::Tensor> m_taylor_factors = {};

    /**
     * @brief The step index when Taylor factors were last updated.
     * Initialized to std::nullopt to indicate no updates yet.
     */
    std::optional<std::size_t> m_last_update_step = std::nullopt;

    /**
     * @brief Maximum order of Taylor series approximation to use.
     */
    std::size_t m_max_order = 2;
};

} // namespace ov::genai
