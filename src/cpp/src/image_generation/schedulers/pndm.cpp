// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <random>
#include <fstream>
#include <iterator>

#include "image_generation/schedulers/pndm.hpp"
#include "image_generation/numpy_utils.hpp"

namespace ov {
namespace genai {

PNDMScheduler::Config::Config(const std::filesystem::path& scheduler_config_path) {
    std::ifstream file(scheduler_config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", scheduler_config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "num_train_timesteps", num_train_timesteps);
    read_json_param(data, "beta_start", beta_start);
    read_json_param(data, "beta_end", beta_end);
    read_json_param(data, "beta_schedule", beta_schedule);
    read_json_param(data, "trained_betas", trained_betas);
    read_json_param(data, "set_alpha_to_one", set_alpha_to_one);
    read_json_param(data, "skip_prk_steps", skip_prk_steps);
    read_json_param(data, "steps_offset", steps_offset);
    read_json_param(data, "prediction_type", prediction_type);
    read_json_param(data, "timestep_spacing", timestep_spacing);
}

PNDMScheduler::PNDMScheduler(const std::filesystem::path& scheduler_config_path) 
    : PNDMScheduler(Config(scheduler_config_path)) {
}

PNDMScheduler::PNDMScheduler(const Config& scheduler_config): m_config(scheduler_config) {

    std::vector<float> alphas, betas;

    using numpy_utils::linspace;

    if (!m_config.trained_betas.empty()) {
        betas = m_config.trained_betas;
    } else if (m_config.beta_schedule == BetaSchedule::LINEAR) {
        betas = linspace<float>(m_config.beta_start, m_config.beta_end, m_config.num_train_timesteps);
    } else if (m_config.beta_schedule == BetaSchedule::SCALED_LINEAR) {
        float start = std::sqrt(m_config.beta_start);
        float end = std::sqrt(m_config.beta_end);
        betas = linspace<float>(start, end, m_config.num_train_timesteps);
        std::for_each(betas.begin(), betas.end(), [] (float & x) { x *= x; });
        // TODO: elif beta_schedule == "squaredcos_cap_v2":
    } else {
        OPENVINO_THROW("'beta_schedule' must be one of 'LINEAR' or 'SCALED_LINEAR'. Please, add support of other types");
    }

    std::transform(betas.begin(), betas.end(), std::back_inserter(alphas), [] (float b) { return 1.0f - b; });

    for (size_t i = 1; i <= alphas.size(); i++) {
        float alpha_cumprod =
            std::accumulate(std::begin(alphas), std::begin(alphas) + i, 1.0, std::multiplies<float>{});
        m_alphas_cumprod.push_back(alpha_cumprod);
    }

    m_final_alpha_cumprod = m_config.set_alpha_to_one ? 1 : m_alphas_cumprod[0];

    // running values
    m_ets = {};
    m_counter = 0;

    // setable values
    m_num_inference_steps = -1;
    m_prk_timesteps = {};
    m_plms_timesteps = {};
    m_timesteps = {};
}

void PNDMScheduler::set_timesteps(size_t num_inference_steps, float strength) {
    m_timesteps.clear(), m_prk_timesteps.clear(), m_plms_timesteps.clear();

    OPENVINO_ASSERT(num_inference_steps <= m_config.num_train_timesteps,
                    "`num_inference_steps` cannot be larger than `m_config.num_train_timesteps`");

    m_num_inference_steps = num_inference_steps;

    switch (m_config.timestep_spacing) {
        case TimestepSpacing::LINSPACE:
        {
            using numpy_utils::linspace;
            float end = static_cast<float>(m_config.num_train_timesteps - 1);
            auto linspaced = linspace<float>(0.0f, end, num_inference_steps, true);
            for (float val : linspaced) {
                m_timesteps.push_back(static_cast<int64_t>(std::round(val)));
            }
            break;
        }
        case TimestepSpacing::LEADING:
        {
            size_t step_ratio = m_config.num_train_timesteps / m_num_inference_steps;
            for (size_t i = 0; i < m_num_inference_steps; ++i) {
                m_timesteps.push_back(i * step_ratio + m_config.steps_offset);
            }
            break;
        }
        case TimestepSpacing::TRAILING:
        {
            float step_ratio = static_cast<float>(m_config.num_train_timesteps) / static_cast<float>(m_num_inference_steps);
            for (float i = m_config.num_train_timesteps; i > 0; i-=step_ratio){
                m_timesteps.push_back(static_cast<int64_t>(std::round(i)) - 1);
            }
            std::reverse(m_timesteps.begin(), m_timesteps.end());
            break;
        }
        default:
            OPENVINO_THROW("Unsupported value for 'timestep_spacing'. Please make sure to choose one of 'linspace', 'leading' or 'trailing'.");
    }

    if (m_config.skip_prk_steps) {
        m_prk_timesteps = {};
        std::copy(m_timesteps.begin(), m_timesteps.end() - 1, std::back_inserter(m_plms_timesteps));
        m_plms_timesteps.push_back(m_timesteps[m_timesteps.size() - 2]);
        m_plms_timesteps.push_back(m_timesteps[m_timesteps.size() - 1]);
        std::reverse(m_plms_timesteps.begin(), m_plms_timesteps.end());
    } else {
        OPENVINO_THROW("'skip_prk_steps=false' case isn't supported. Please, add support.");
    }

    m_timesteps = m_prk_timesteps;
    m_timesteps.insert(m_timesteps.end(), m_plms_timesteps.begin(), m_plms_timesteps.end());

    m_ets = {};
    m_counter = 0;
    m_cur_sample = ov::Tensor(ov::element::f32, {});

    // apply 'strength' used in image generation
    // in diffusers, it's https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py#L711
    {
        size_t init_timestep = std::min<size_t>(num_inference_steps * strength, num_inference_steps);
        size_t t_start = std::max<size_t>(num_inference_steps - init_timestep, 0);
        m_timesteps = std::vector<int64_t>(m_timesteps.begin() + t_start, m_timesteps.end());

        OPENVINO_ASSERT(!m_timesteps.empty(),
                        "After adjusting the num_inference_steps by strength parameter: ", strength,
                        " the number of pipeline steps is less then 1 and not appropriate for this pipeline. Please set a different strength value.");
    }
}

std::map<std::string, ov::Tensor> PNDMScheduler::step(ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step, std::shared_ptr<Generator> generator) {
    // noise_pred - model_output
    // latents - sample
    // inference_step

    if (m_counter < m_prk_timesteps.size() && !m_config.skip_prk_steps) {
        OPENVINO_THROW("'skip_prk_steps=false' case isn't supported. Please, add support.");
    } else {
        return step_plms(noise_pred, latents, m_timesteps[inference_step]);
    }
}

std::map<std::string, ov::Tensor> PNDMScheduler::step_plms(ov::Tensor model_output, ov::Tensor sample, size_t timestep) {
    OPENVINO_ASSERT(m_num_inference_steps != -1,
                    "Number of inference steps isn't set, you need to run `set_timesteps` after creating the scheduler");

    int prev_timestep = timestep - m_config.num_train_timesteps / m_num_inference_steps;

    if (m_counter != 1) {
        if (m_ets.size() > 3) {
            m_ets = std::vector<ov::Tensor>(m_ets.end() - 3, m_ets.end());
        }
        ov::Tensor ets_last(model_output.get_element_type(), model_output.get_shape());
        model_output.copy_to(ets_last);
        m_ets.push_back(ets_last);
    } else {
        prev_timestep = timestep;
        timestep = timestep + m_config.num_train_timesteps / m_num_inference_steps;
    }

    float* model_output_data = model_output.data<float>();

    size_t m_ets_size = m_ets.size();

    if (m_ets_size == 1 && m_counter == 0) {
        m_cur_sample = ov::Tensor(sample.get_element_type(), sample.get_shape());
        sample.copy_to(m_cur_sample);
    } else if (m_ets_size == 1 && m_counter == 1) {
        const float* ets_data = m_ets[0].data<float>();
        for (size_t i = 0; i < model_output.get_size(); ++i) {
            model_output_data[i] = (model_output_data[i] + ets_data[i]) / 2.0f;
        }
        sample = ov::Tensor(m_cur_sample.get_element_type(), m_cur_sample.get_shape());
        m_cur_sample.copy_to(sample);
        m_cur_sample = ov::Tensor(ov::element::f32, {});
    } else if (m_ets_size == 2) {
        const float* ets_data_1 = m_ets[1].data<float>();
        const float* ets_data_2 = m_ets[0].data<float>();
        for (size_t i = 0; i < model_output.get_size(); ++i) {
            model_output_data[i] = (3.0f * ets_data_1[i] - ets_data_2[i]) / 2.0f;
        }
    } else if (m_ets_size == 3) {
        const float* ets_data_1 = m_ets[2].data<float>();
        const float* ets_data_2 = m_ets[1].data<float>();
        const float* ets_data_3 = m_ets[0].data<float>();
        for (size_t i = 0; i < model_output.get_size(); ++i) {
            model_output_data[i] = (23.0f * ets_data_1[i] - 16.0f * ets_data_2[i] + 5.0f * ets_data_3[i]) / 12.0f;
        }
    } else if (m_ets_size == 4) {
        const float* ets_data_1 = m_ets[3].data<float>();
        const float* ets_data_2 = m_ets[2].data<float>();
        const float* ets_data_3 = m_ets[1].data<float>();
        const float* ets_data_4 = m_ets[0].data<float>();

        for (size_t i = 0; i < model_output.get_size(); ++i) {
            model_output_data[i] = (1.0f / 24.0f)
                                   * (55.0f * ets_data_1[i] - 59.0f * ets_data_2[i] + 37.0f * ets_data_3[i] - 9.0f * ets_data_4[i]);
        }
    } else {
        OPENVINO_THROW("PNDMScheduler: Unsupported step_plms case.");
    }

    ov::Tensor prev_sample = get_prev_sample(sample, timestep, prev_timestep, model_output);
    m_counter++;

    std::map<std::string, ov::Tensor> result{{"latent", prev_sample}};
    return result;
}

ov::Tensor PNDMScheduler::get_prev_sample(ov::Tensor sample, size_t timestep, int prev_timestep, ov::Tensor model_output) {
    float alpha_prod_t = m_alphas_cumprod[timestep];
    float alpha_prod_t_prev = (prev_timestep >= 0) ? m_alphas_cumprod[prev_timestep] : m_final_alpha_cumprod;
    float beta_prod_t = 1 - alpha_prod_t;
    float beta_prod_t_prev = 1 - alpha_prod_t_prev;

    float sample_coeff = std::sqrt((alpha_prod_t_prev / alpha_prod_t));
    float model_output_denom_coeff = alpha_prod_t * std::sqrt(beta_prod_t_prev) +
                                     std::sqrt((alpha_prod_t * beta_prod_t * alpha_prod_t_prev));

    float* model_output_data = model_output.data<float>();
    float* sample_data = sample.data<float>();

    switch (m_config.prediction_type) {
        case PredictionType::EPSILON:
            break;
        case PredictionType::V_PREDICTION:
            for (size_t i = 0; i < model_output.get_size(); ++i) {
                model_output_data[i] = std::sqrt(alpha_prod_t) * model_output_data[i] + std::sqrt(beta_prod_t) * sample_data[i];
            }
            break;
        default:
            OPENVINO_THROW("Unsupported value for 'PredictionType'");
    }

    ov::Tensor prev_sample = ov::Tensor(model_output.get_element_type(), model_output.get_shape());
    float* prev_sample_data = prev_sample.data<float>();

    for (size_t i = 0; i < prev_sample.get_size(); ++i) {
        prev_sample_data[i] = sample_coeff * sample_data[i] - (alpha_prod_t_prev - alpha_prod_t) * model_output_data[i] / model_output_denom_coeff;
    }

    return prev_sample;
}

void PNDMScheduler::add_noise(ov::Tensor init_latent, ov::Tensor noise, int64_t latent_timestep) const {
    float sqrt_alpha_prod = std::sqrt(m_alphas_cumprod[latent_timestep]);
    float sqrt_one_minus_alpha_prod = std::sqrt(1.0 - m_alphas_cumprod[latent_timestep]);

    float * init_latent_data = init_latent.data<float>();
    const float * noise_data = noise.data<float>();

    for (size_t i = 0; i < init_latent.get_size(); ++i) {
        init_latent_data[i] = sqrt_alpha_prod * init_latent_data[i] + sqrt_one_minus_alpha_prod * noise_data[i];
    }
}

std::vector<int64_t> PNDMScheduler::get_timesteps() const {
    OPENVINO_ASSERT(!m_timesteps.empty(), "'timesteps' have not yet been set.");

    return m_timesteps;
}

void PNDMScheduler::scale_model_input(ov::Tensor sample, size_t inference_step) {
    return;
}

float PNDMScheduler::get_init_noise_sigma() const {
    return 1.0f;
}


} // namespace genai
} // namespace ov
