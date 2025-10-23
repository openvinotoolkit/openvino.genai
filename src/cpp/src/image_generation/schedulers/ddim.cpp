// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <random>
#include <fstream>
#include <iterator>

#include "image_generation/schedulers/ddim.hpp"
#include "image_generation/numpy_utils.hpp"

namespace ov {
namespace genai {

DDIMScheduler::Config::Config(const std::filesystem::path& scheduler_config_path) {
    std::ifstream file(scheduler_config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", scheduler_config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "num_train_timesteps", num_train_timesteps);
    read_json_param(data, "beta_start", beta_start);
    read_json_param(data, "beta_end", beta_end);
    read_json_param(data, "beta_schedule", beta_schedule);
    read_json_param(data, "trained_betas", trained_betas);
    read_json_param(data, "clip_sample", clip_sample);
    read_json_param(data, "set_alpha_to_one", set_alpha_to_one);
    read_json_param(data, "steps_offset", steps_offset);
    read_json_param(data, "prediction_type", prediction_type);
    read_json_param(data, "thresholding", thresholding);
    read_json_param(data, "dynamic_thresholding_ratio", dynamic_thresholding_ratio);
    read_json_param(data, "clip_sample_range", clip_sample_range);
    read_json_param(data, "sample_max_value", sample_max_value);
    read_json_param(data, "timestep_spacing", timestep_spacing);
    read_json_param(data, "rescale_betas_zero_snr", rescale_betas_zero_snr);
}

DDIMScheduler::DDIMScheduler(const std::filesystem::path& scheduler_config_path) 
    : DDIMScheduler(Config(scheduler_config_path)) {
}

DDIMScheduler::DDIMScheduler(const Config& scheduler_config)
    : m_config(scheduler_config),
      m_final_alpha_cumprod(0.0f),
      m_num_inference_steps(0) {
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
    } else {
        OPENVINO_THROW("'beta_schedule' must be one of 'LINEAR' or 'SCALED_LINEAR'. Please, add support of other types");
    }

    if (m_config.rescale_betas_zero_snr) {
        using numpy_utils::rescale_zero_terminal_snr;
        rescale_zero_terminal_snr(betas);
    }

    std::transform(betas.begin(), betas.end(), std::back_inserter(alphas), [] (float b) { return 1.0f - b; });

    for (size_t i = 1; i <= alphas.size(); i++) {
        float alpha_cumprod =
            std::accumulate(std::begin(alphas), std::begin(alphas) + i, 1.0, std::multiplies<float>{});
        m_alphas_cumprod.push_back(alpha_cumprod);
    }

    m_final_alpha_cumprod = m_config.set_alpha_to_one ? 1 : m_alphas_cumprod[0];
}

void DDIMScheduler::set_timesteps(size_t num_inference_steps, float strength) {
    m_timesteps.clear();

    OPENVINO_ASSERT(num_inference_steps <= m_config.num_train_timesteps,
                    "`num_inference_steps` cannot be larger than `m_config.num_train_timesteps`");

    m_num_inference_steps = num_inference_steps;

    switch (m_config.timestep_spacing) {
        case TimestepSpacing::LINSPACE:
        {
            using numpy_utils::linspace;
            float end = static_cast<float>(m_config.num_train_timesteps - 1);
            auto linspaced = linspace<float>(0.0f, end, num_inference_steps, true);
            for (auto it = linspaced.rbegin(); it != linspaced.rend(); ++it) {
                m_timesteps.push_back(static_cast<int64_t>(std::round(*it)));
            }
            break;
        }
        case TimestepSpacing::LEADING:
        {
            size_t step_ratio = m_config.num_train_timesteps / m_num_inference_steps;
            for (size_t i = num_inference_steps - 1; i != -1; --i) {
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
            break;
        }
        default:
            OPENVINO_THROW("Unsupported value for 'timestep_spacing'");
    }

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

std::map<std::string, ov::Tensor> DDIMScheduler::step(ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step, std::shared_ptr<Generator> generator) {
    // noise_pred - model_output
    // latents - sample
    // inference_step

    size_t timestep = m_timesteps[inference_step];

    // get previous step value (=t-1)
    int prev_timestep = timestep - m_config.num_train_timesteps / m_num_inference_steps;

    // compute alphas, betas
    float alpha_prod_t = m_alphas_cumprod[timestep];
    float alpha_prod_t_prev = (prev_timestep >= 0) ? m_alphas_cumprod[prev_timestep] : m_final_alpha_cumprod;
    float beta_prod_t = 1 - alpha_prod_t;

    // compute predicted original sample from predicted noise also called
    // "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    std::vector<float> pred_original_sample, pred_epsilon;
    float pos_val, pe_val;
    for (size_t j = 0; j < noise_pred.get_size(); j++) {
        switch (m_config.prediction_type) {
            case PredictionType::EPSILON:
                pos_val = (latents.data<float>()[j] - std::sqrt(beta_prod_t) * noise_pred.data<float>()[j]) / std::sqrt(alpha_prod_t);
                pe_val = noise_pred.data<float>()[j];
                pred_original_sample.push_back(pos_val);
                pred_epsilon.push_back(pe_val);
                break;
            case PredictionType::SAMPLE:
                pos_val = noise_pred.data<float>()[j];
                pe_val = (latents.data<float>()[j] - std::sqrt(alpha_prod_t) * pos_val) / std::sqrt(beta_prod_t);
                pred_original_sample.push_back(pos_val);
                pred_epsilon.push_back(pe_val);
                break;
            case PredictionType::V_PREDICTION:
                pos_val = std::sqrt(alpha_prod_t) * latents.data<float>()[j] - std::sqrt(beta_prod_t) * noise_pred.data<float>()[j];
                pe_val = std::sqrt(alpha_prod_t) * noise_pred.data<float>()[j] + std::sqrt(beta_prod_t) * latents.data<float>()[j];
                pred_original_sample.push_back(pos_val);
                pred_epsilon.push_back(pe_val);
                break;
            default:
                OPENVINO_THROW("Unsupported value for 'PredictionType'");
        }
    }

    // TODO: support m_config.thresholding
    OPENVINO_ASSERT(!m_config.thresholding,
                    "Parameter 'thresholding' is not supported. Please, add support.");
    // TODO: support m_config.clip_sample
    OPENVINO_ASSERT(!m_config.clip_sample,
                    "Parameter 'clip_sample' is not supported. Please, add support.");

    // compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    std::vector<float> pred_sample_direction(pred_epsilon.size());
    std::transform(pred_epsilon.begin(), pred_epsilon.end(), pred_sample_direction.begin(), [alpha_prod_t_prev](auto x) {
        return std::sqrt(1 - alpha_prod_t_prev) * x;
    });

    // compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    ov::Tensor prev_sample(latents.get_element_type(), latents.get_shape());
    float* prev_sample_data = prev_sample.data<float>();
    for (size_t i = 0; i < prev_sample.get_size(); ++i) {
        prev_sample_data[i] = std::sqrt(alpha_prod_t_prev) * pred_original_sample[i] + pred_sample_direction[i];
    }

    std::map<std::string, ov::Tensor> result{{"latent", prev_sample}};

    return result;
}

std::vector<int64_t> DDIMScheduler::get_timesteps() const {
    OPENVINO_ASSERT(!m_timesteps.empty(), "'timesteps' have not yet been set.");

    return m_timesteps;
}

float DDIMScheduler::get_init_noise_sigma() const {
    return 1.0f;
}

void DDIMScheduler::scale_model_input(ov::Tensor sample, size_t inference_step) {
    return;
}

void DDIMScheduler::add_noise(ov::Tensor init_latent, ov::Tensor noise, int64_t latent_timestep) const {
    float sqrt_alpha_prod = std::sqrt(m_alphas_cumprod[latent_timestep]);
    float sqrt_one_minus_alpha_prod = std::sqrt(1.0 - m_alphas_cumprod[latent_timestep]);

    float * init_latent_data = init_latent.data<float>();
    const float * noise_data = noise.data<float>();

    for (size_t i = 0; i < init_latent.get_size(); ++i) {
        init_latent_data[i] = sqrt_alpha_prod * init_latent_data[i] + sqrt_one_minus_alpha_prod * noise_data[i];
    }
}


} // namespace genai
} // namespace ov
