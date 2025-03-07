// Copyright (C) 2023-2025 Intel Corporation
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

    m_timesteps = linspace<float>(1.0f, static_cast<float>(num_train_timesteps), num_train_timesteps, true);
    std::reverse(m_timesteps.begin(), m_timesteps.end());

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

    m_step_index = -1, m_begin_index = -1, m_strength = -1, m_num_inference_steps = -1;
    m_sigma_max = m_sigmas[0], m_sigma_min = m_sigmas.back();
}

double FlowMatchEulerDiscreteScheduler::sigma_to_t(double sigma) {
    return sigma * m_config.num_train_timesteps;
}

void FlowMatchEulerDiscreteScheduler::set_timesteps(size_t num_inference_steps, float strength) {
    m_timesteps.clear();
    m_sigmas.clear();

    m_num_inference_steps = num_inference_steps;
    m_strength = strength;

    int32_t num_train_timesteps = m_config.num_train_timesteps;
    float shift = m_config.shift;

    using numpy_utils::linspace;
    std::vector<double> timesteps = linspace<double>(sigma_to_t(m_sigma_max), sigma_to_t(m_sigma_min), m_num_inference_steps, true);

    std::vector<double> sigmas(timesteps.size());
    for (size_t i = 0; i < sigmas.size(); ++i) {
        sigmas[i] = timesteps[i] / num_train_timesteps;
    }

    OPENVINO_ASSERT(!m_config.use_dynamic_shifting,
                    "Parameter 'use_dynamic_shifting' is not supported. Please, add support.");

    m_sigmas.resize(sigmas.size());
    m_timesteps.resize(sigmas.size());

    for (size_t i = 0; i < sigmas.size(); ++i) {
        m_sigmas[i] = shift * sigmas[i] / (1.0 + (shift - 1.0) * sigmas[i]);
        m_timesteps[i] = m_sigmas[i] * num_train_timesteps;
    }
    m_sigmas.push_back(0.0f);

    m_step_index = -1, m_begin_index = -1;
}

std::map<std::string, ov::Tensor> FlowMatchEulerDiscreteScheduler::step(ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step, std::shared_ptr<Generator> generator) {
    // noise_pred - model_output
    // latents - sample
    // inference_step

    const float* model_output_data = noise_pred.data<const float>();
    const float* sample_data = latents.data<const float>();

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

std::vector<float> FlowMatchEulerDiscreteScheduler::get_float_timesteps() {
    OPENVINO_ASSERT(m_strength != -1,
                    "Parameter 'strength' was not yes passed to Scheduler.");
    OPENVINO_ASSERT(m_num_inference_steps != -1,
                    "Parameter 'num_inference_steps' was not yes passed to Scheduler.");
    OPENVINO_ASSERT(!m_timesteps.empty(), "'timesteps' have not yet been set.");
    // For Text2Image strength is always 1.0 (guaranteed by pipeline)
    float init_timestep = std::min(static_cast<float>(m_num_inference_steps) * m_strength, static_cast<float>(m_num_inference_steps));
    size_t t_start = static_cast<size_t>(std::max(static_cast<float>(m_num_inference_steps) - init_timestep, 0.0f));

    std::vector<float> timesteps;
    for (size_t i = t_start; i < m_timesteps.size(); ++i) {
        timesteps.push_back(m_timesteps[i]);
    }

    OPENVINO_ASSERT(!timesteps.empty(),
                    "After adjusting the num_inference_steps by strength parameter: ", m_strength,
                    " the number of pipeline steps is less then 1 and not appropriate for this pipeline. Please set a different strength value.");

    set_begin_index(t_start);
    return timesteps;
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

void FlowMatchEulerDiscreteScheduler::add_noise(ov::Tensor init_latent, ov::Tensor noise, int64_t latent_timestep) const {
    // use https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L117
    OPENVINO_THROW("Not implemented");
}

size_t FlowMatchEulerDiscreteScheduler::_index_for_timestep(float timestep) {
    if (m_schedule_timesteps.empty()) {
        m_schedule_timesteps = m_timesteps;
    }

    for (size_t i = 0; i < m_schedule_timesteps.size(); ++i) {
        if (timestep == m_schedule_timesteps[i]) {
            return i;
        }
    }

    OPENVINO_THROW("Failed to find index for timestep ", timestep);
}

void FlowMatchEulerDiscreteScheduler::scale_noise(ov::Tensor sample, float timestep, ov::Tensor noise) {
    OPENVINO_ASSERT(timestep > 0, "Timestep is not computed yet");
    
    size_t index_for_timestep;
    if (m_begin_index == -1) {
        index_for_timestep = _index_for_timestep(timestep);
    } else if (m_step_index != -1) {
        index_for_timestep = m_step_index;
    } else {
        index_for_timestep = m_begin_index;
    }

    const float sigma = m_sigmas[index_for_timestep];

    float * sample_data = sample.data<float>();
    const float * noise_data = noise.data<float>();

    for (size_t i = 0; i < sample.get_size(); ++i) {
        sample_data[i] = sigma * noise_data[i] + (1.0f - sigma) * sample_data[i];
    }
}

void FlowMatchEulerDiscreteScheduler::set_timesteps(size_t image_seq_len, size_t num_inference_steps, float strength) {
    m_timesteps.clear();
    m_sigmas.clear();

    m_num_inference_steps = num_inference_steps;
    m_strength = strength;

    float linspace_end = 1.0f / m_num_inference_steps;
    m_sigmas = numpy_utils::linspace<float>(1.0f, linspace_end, m_num_inference_steps, true);

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

float FlowMatchEulerDiscreteScheduler::calculate_shift(size_t image_seq_len) {
    size_t base_seq_len = m_config.base_image_seq_len;
    size_t max_seq_len = m_config.max_image_seq_len;
    float base_shift = m_config.base_shift;
    float max_shift = m_config.max_shift;

    float m = (max_shift - base_shift) / (max_seq_len - base_seq_len);
    float b = base_shift - m * base_seq_len;
    float mu = image_seq_len * m + b;
    return mu;
}

void FlowMatchEulerDiscreteScheduler::set_begin_index(size_t begin_index) {
    m_begin_index = begin_index;
}

}  // namespace genai
}  // namespace ov
