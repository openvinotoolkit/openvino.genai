// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image_generation/schedulers/flow_match_euler_discrete.hpp"

#include <cassert>
#include <fstream>
#include <iterator>
#include <random>

#include "image_generation/numpy_utils.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

FlowMatchEulerDiscreteScheduler::Config::Config(const std::filesystem::path& scheduler_config_path) {
    std::ifstream file(scheduler_config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", scheduler_config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "num_train_timesteps", num_train_timesteps);
    read_json_param(data, "shift", shift);
    read_json_param(data, "use_dynamic_shifting", use_dynamic_shifting);
    read_json_param(data, "base_shift", base_shift);
    read_json_param(data, "max_shift", max_shift);
    read_json_param(data, "base_image_seq_len", base_image_seq_len);
    read_json_param(data, "max_image_seq_len", max_image_seq_len);
}

FlowMatchEulerDiscreteScheduler::FlowMatchEulerDiscreteScheduler(const std::filesystem::path& scheduler_config_path)
    : FlowMatchEulerDiscreteScheduler(Config(scheduler_config_path)) {}

FlowMatchEulerDiscreteScheduler::FlowMatchEulerDiscreteScheduler(const Config& scheduler_config)
    : m_config(scheduler_config) {
    using numpy_utils::linspace;

    int32_t num_train_timesteps = m_config.num_train_timesteps;
    float shift = m_config.shift;

    auto linspaced = linspace<float>(1.0f, static_cast<float>(num_train_timesteps), num_train_timesteps, true);
    for (auto it = linspaced.rbegin(); it != linspaced.rend(); ++it) {
        m_timesteps.push_back(*it);
    }

    std::transform(m_timesteps.begin(),
                   m_timesteps.end(),
                   std::back_inserter(m_sigmas),
                   [num_train_timesteps](float x) {
                       return x / num_train_timesteps;
                   });

    if (!m_config.use_dynamic_shifting) {
        std::transform(m_sigmas.begin(), m_sigmas.end(), m_sigmas.begin(), [shift](float x) {
            return shift * x / (1 + (shift - 1) * x);
        });
    }

    for (size_t i = 0; i < m_timesteps.size(); ++i) {
        m_timesteps[i] = m_sigmas[i] * num_train_timesteps;
    }

    m_step_index = -1, m_begin_index = -1;
    m_sigma_max = m_sigmas[0], m_sigma_min = m_sigmas.back();
}

float FlowMatchEulerDiscreteScheduler::sigma_to_t(float sigma) {
    return sigma * m_config.num_train_timesteps;
}

void FlowMatchEulerDiscreteScheduler::set_timesteps(size_t num_inference_steps, float strength) {
    m_timesteps.clear();
    m_sigmas.clear();

    m_num_inference_steps = num_inference_steps;
    int32_t num_train_timesteps = m_config.num_train_timesteps;
    float shift = m_config.shift;

    using numpy_utils::linspace;
    m_timesteps = linspace<float>(sigma_to_t(m_sigma_max), sigma_to_t(m_sigma_min), m_num_inference_steps, true);

    for (const float& i : m_timesteps) {
        m_sigmas.push_back(i / num_train_timesteps);
    }

    OPENVINO_ASSERT(!m_config.use_dynamic_shifting,
                    "Parameter 'use_dynamic_shifting' is not supported. Please, add support.");

    for (size_t i = 0; i < m_sigmas.size(); ++i) {
        m_sigmas[i] = shift * m_sigmas[i] / (1 + (shift - 1) * m_sigmas[i]);
        m_timesteps[i] = m_sigmas[i] * num_train_timesteps;
    }
    m_sigmas.push_back(0);

    m_step_index = -1, m_begin_index = -1;
}

std::map<std::string, ov::Tensor> FlowMatchEulerDiscreteScheduler::step(ov::Tensor noise_pred,
                                                                        ov::Tensor latents,
                                                                        size_t inference_step) {
    // noise_pred - model_output
    // latents - sample
    // inference_step

    float* model_output_data = noise_pred.data<float>();
    float* sample_data = latents.data<float>();

    if (m_step_index == -1)
        init_step_index();

    ov::Tensor prev_sample(latents.get_element_type(), latents.get_shape());
    float* prev_sample_data = prev_sample.data<float>();

    float sigma_diff = m_sigmas[m_step_index + 1] - m_sigmas[m_step_index];

    for (size_t i = 0; i < prev_sample.get_size(); ++i) {
        prev_sample_data[i] = sample_data[i] + sigma_diff * model_output_data[i];
    }

    m_step_index++;

    return {{"latent", prev_sample}};
}

std::vector<std::int64_t> FlowMatchEulerDiscreteScheduler::get_timesteps() const {
    OPENVINO_THROW("FlowMatchEulerDiscreteScheduler doesn't support int timesteps");
}

std::vector<float> FlowMatchEulerDiscreteScheduler::get_float_timesteps() const {
    return m_timesteps;
}

float FlowMatchEulerDiscreteScheduler::get_init_noise_sigma() const {
    return 1.0f;
}

void FlowMatchEulerDiscreteScheduler::scale_model_input(ov::Tensor sample, size_t inference_step) {
    return;
}

void FlowMatchEulerDiscreteScheduler::init_step_index() {
    // TODO: support index_for_timestep method
    m_step_index = (m_begin_index == -1) ? 0 : m_begin_index;
}

void FlowMatchEulerDiscreteScheduler::add_noise(ov::Tensor init_latent, std::shared_ptr<Generator> rng_generator) const {
    // use https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L117
    OPENVINO_THROW("Not implemented");
}

}  // namespace genai
}  // namespace ov
