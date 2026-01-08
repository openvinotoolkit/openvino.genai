// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "z_image_flow_match_euler_discrete.hpp"
#include "image_generation/numpy_utils.hpp"

namespace ov::genai {

void ZImageFlowMatchEulerDiscreteScheduler::set_timesteps(size_t image_seq_len, size_t num_inference_steps, float strength) {
    m_timesteps.clear();
    m_sigmas.clear();

    m_num_inference_steps = num_inference_steps;
    m_strength = strength;

    m_sigmas = numpy_utils::linspace<float>(m_sigma_max, m_sigma_min, m_num_inference_steps, true);

    float shift = m_config.shift;

    // fill sigma
    float mu = calculate_shift(image_seq_len);
    if (m_config.use_dynamic_shifting) {
        float exp_mu = std::exp(mu);
        for (size_t i = 0; i < m_sigmas.size(); ++i) {
            m_sigmas[i] = exp_mu / (exp_mu + (1 / m_sigmas[i] - 1));
        }
    } else {
        for (size_t i = 0; i < m_sigmas.size(); ++i) {
            m_sigmas[i] = shift * m_sigmas[i] / (1 + (shift - 1) * m_sigmas[i]);
        }
    }

    // fill timesteps
    for (size_t i = 0; i < m_sigmas.size(); ++i) {
        m_timesteps.push_back(m_sigmas[i] * m_config.num_train_timesteps);
    }
    m_sigmas.push_back(0);
    m_step_index = -1, m_begin_index = -1;
}

void ZImageFlowMatchEulerDiscreteScheduler::set_sigma_min(float sigma_min) {
    m_sigma_min = sigma_min;
}

}