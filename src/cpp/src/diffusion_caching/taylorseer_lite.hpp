// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <openvino/core/except.hpp>
#include <openvino/genai/taylorseer_config.hpp>
#include <openvino/runtime/tensor.hpp>
#include <optional>
#include <sstream>
#include <vector>

namespace ov::genai {

/**
 * @brief State management class for TaylorSeer cache mechanism.
 *
 * Maintains Taylor series factors and tracks the last update step to enable
 * prediction of transformer outputs during inference.
 */
class TaylorSeerState {
public:
    /**
     * @brief Constructs TaylorSeerState with optional configuration.
     * @param config Optional TaylorSeer configuration.
     * @param num_inference_steps Total number of inference steps in the generation process.
     * @note If config is not provided or invalid, TaylorSeer remains inactive.
     */
    TaylorSeerState(const std::optional<TaylorSeerCacheConfig>& config, std::size_t num_inference_steps) {
        initialize(config, num_inference_steps);
    }

    /**
     * @brief Initializes TaylorSeer with the given configuration.
     * @param config Optional TaylorSeer configuration.
     * @param num_inference_steps Total number of inference steps.
     */
    void initialize(const std::optional<TaylorSeerCacheConfig>& config, std::size_t num_inference_steps) {
        // Reset all state to ensure clean initialization
        reset_state();

        if (!config) {
            return;
        }

        OPENVINO_ASSERT(config->cache_interval >= 2,
                        "TaylorSeerCacheConfig: cache_interval must be at least 2, got ",
                        config->cache_interval);

        // Check if TaylorSeer will be effective
        if (config->disable_cache_before_step >= num_inference_steps) {
            return;
        }

        int disable_cache_after_step = config->disable_cache_after_step;
        if (disable_cache_after_step < 0) {
            disable_cache_after_step = static_cast<int>(num_inference_steps) + disable_cache_after_step;
            if (disable_cache_after_step < 0) {  // If still negative, it means caching is disabled for all steps
                return;
            }
        }

        if (static_cast<std::size_t>(disable_cache_after_step) <= config->disable_cache_before_step) {
            return;
        }

        // Precompute schedule and check if schedule has any prediction steps
        bool has_predictions = false;
        m_schedule.resize(num_inference_steps);
        for (std::size_t step = 0; step < num_inference_steps; ++step) {
            m_schedule[step] = should_compute_at_step(step, *config, num_inference_steps);
            if (!m_schedule[step]) {
                has_predictions = true;
                m_last_prediction_step = step;  // Track the last prediction step
            }
        }
        m_is_active = has_predictions;

        if (!m_is_active) {
            m_schedule.clear();
        }
    }

    /**
     * @brief Checks if TaylorSeer is active and will perform caching.
     * @return true if TaylorSeer is active, false otherwise.
     */
    bool is_active() const {
        return m_is_active;
    }

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
        OPENVINO_ASSERT(order < m_max_order,
                        "Requested Taylor factor order ",
                        order,
                        " is out of bounds. Maximum supported order is ",
                        m_max_order - 1,
                        ".");
        return m_taylor_factors[order];
    }

    /**
     * @brief Determines if a full computation should be performed at the current step using precomputed schedule.
     * @param current_step The current denoising step index.
     * @return true if full computation is required, false if cached prediction can be used.
     */
    bool should_compute(std::size_t current_step) const {
        OPENVINO_ASSERT(current_step < m_schedule.size(),
                        "Step ",
                        current_step,
                        " is out of bounds for precomputed schedule of size ",
                        m_schedule.size());
        return m_schedule[current_step];
    }

    /**
     * @brief Updates Taylor factors with a new output from a full computation.
     * @param current_step The current denoising step index.
     * @param output The output tensor from the full computation.
     * @throws ov::Exception if current_step is not greater than the last update step.
     * @note Updates are skipped if there are no future prediction steps.
     */
    void update(std::size_t current_step, const ov::Tensor& output) {
        // Skip update if no future predictions will use these Taylor factors
        if (m_last_prediction_step.has_value() && current_step >= *m_last_prediction_step) {
            return;
        }

        const bool is_first_update = !m_last_update_step.has_value();

        OPENVINO_ASSERT(is_first_update || current_step > *m_last_update_step,
                        "Current step (",
                        current_step,
                        ") must be greater than the last update step (",
                        *m_last_update_step,
                        ") for TaylorSeerState update.");

        // Validate tensor shape consistency
        if (!is_first_update) {
            const auto& prev_factor = get_taylor_factor(0);
            OPENVINO_ASSERT(output.get_shape() == prev_factor.get_shape(),
                            "Output tensor shape ",
                            output.get_shape(),
                            " does not match previous tensor shape ",
                            prev_factor.get_shape());
        }

        std::array<ov::Tensor, m_max_order> new_factors;

        // Detach factor 0 from potential InferRequest-owned buffer by copying data.
        ov::Tensor detached_output(output.get_element_type(), output.get_shape());
        output.copy_to(detached_output);
        new_factors[0] = std::move(detached_output);

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

        // Update Taylor factors
        for (std::size_t order = 0; order < m_max_order; ++order) {
            m_taylor_factors[order] = std::move(new_factors[order]);
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
                        m_max_order,
                        ", Available: ",
                        m_taylor_factors.size());

        OPENVINO_ASSERT(m_last_update_step.has_value(), "Cannot predict before first update.");

        OPENVINO_ASSERT(current_step > *m_last_update_step,
                        "Cannot predict for step ",
                        current_step,
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
     * @brief Resets all internal state to initial values.
     *
     * Clears Taylor factors, schedule, and resets tracking state.
     */
    void reset_state() {
        m_is_active = false;
        m_last_update_step = std::nullopt;
        m_last_prediction_step = std::nullopt;
        m_schedule.clear();

        // Clear Taylor factor tensors
        for (auto& factor : m_taylor_factors) {
            factor = ov::Tensor();
        }
    }

    bool should_compute_at_step(std::size_t current_step,
                                const TaylorSeerCacheConfig& config,
                                std::size_t num_inference_steps) const {
        // Always compute during warm-up phase and guarantee enough steps to compute Taylor factors
        if (current_step < std::max(config.disable_cache_before_step, m_max_order)) {
            return true;
        }

        int disable_cache_after_step = config.disable_cache_after_step;
        if (disable_cache_after_step < 0) {
            disable_cache_after_step = static_cast<int>(num_inference_steps) + disable_cache_after_step;
        }
        if (disable_cache_after_step >= 0 && current_step >= static_cast<std::size_t>(disable_cache_after_step)) {
            return true;
        }

        auto offset = current_step - std::max(config.disable_cache_before_step, m_max_order);
        auto first_compute_offset = config.cache_interval - 1;

        if (offset < first_compute_offset) {
            return false;  // Predict using cached values
        } else {
            // Compute at first_compute_offset, then every cache_interval steps
            return ((offset - first_compute_offset) % config.cache_interval) == 0;
        }
    }

    /**
     * @brief Number of Taylor series terms to store (base output + derivatives).
     *
     * Set to 2 to store order-0 (base output) and order-1 (first derivative),
     * enabling first-order Taylor approximation: f(x) ≈ f(x₀) + f'(x₀)·Δx.
     * Based on TaylorSeer Lite findings that linear approximation provides
     * the best balance between accuracy, computational efficiency and memory footprint.
     * Higher orders may introduce numerical instability without significant quality gains.
     */
    static constexpr std::size_t m_max_order = 2;

    /**
     * @brief Array of Taylor factor tensors indexed by order.
     * Order 0 stores the base output, order 1 stores the first derivative approximation.
     */
    std::array<ov::Tensor, m_max_order> m_taylor_factors = {};

    /**
     * @brief The step index when Taylor factors were last updated.
     * Initialized to std::nullopt to indicate no updates yet.
     */
    std::optional<std::size_t> m_last_update_step = std::nullopt;

    /**
     * @brief Precomputed schedule of compute vs predict decisions for each step.
     * Empty if not using precomputed schedule.
     */
    std::vector<bool> m_schedule;

    /**
     * @brief Whether TaylorSeer is active for this generation.
     */
    bool m_is_active = false;

    /**
     * @brief The last step index where a prediction will be made.
     * Used to skip unnecessary updates after all predictions are complete.
     */
    std::optional<std::size_t> m_last_prediction_step = std::nullopt;
};

}  // namespace ov::genai
