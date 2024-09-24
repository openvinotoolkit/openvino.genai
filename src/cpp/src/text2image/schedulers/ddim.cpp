// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <random>
#include <fstream>
#include <iterator>

#include "text2image/schedulers/ddim.hpp"


namespace {

// https://gist.github.com/lorenzoriano/5414671
template <typename T, typename U>
std::vector<T> linspace(U start, U end, size_t num, bool endpoint = false) {
    std::vector<T> indices;
    if (num != 0) {
        if (num == 1)
            indices.push_back(static_cast<T>(start));
        else {
            if (endpoint)
                --num;

            U delta = (end - start) / static_cast<U>(num);
            for (size_t i = 0; i < num; i++)
                indices.push_back(static_cast<T>(start + delta * i));

            if (endpoint)
                indices.push_back(static_cast<T>(end));
        }
    }
    return indices;
}

} // namespace

namespace ov {
namespace genai {

DDIMScheduler::Config::Config(const std::string& scheduler_config_path) {
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

DDIMScheduler::DDIMScheduler(const std::string scheduler_config_path) 
    : DDIMScheduler(Config(scheduler_config_path)) {
}

DDIMScheduler::DDIMScheduler(const Config& scheduler_config)
    : m_config(scheduler_config) {

    std::vector<float> alphas, betas;

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

    // TODO: Rescale for zero SNR
    // if (m_config.rescale_betas_zero_snr) {betas = rescale_zero_terminal_snr(betas)}

    std::transform(betas.begin(), betas.end(), std::back_inserter(alphas), [] (float b) { return 1.0f - b; });

    for (size_t i = 1; i <= alphas.size(); i++) {
        float alpha_cumprod =
            std::accumulate(std::begin(alphas), std::begin(alphas) + i, 1.0, std::multiplies<float>{});
        m_alphas_cumprod.push_back(alpha_cumprod);
    }

    m_final_alpha_cumprod = m_config.set_alpha_to_one ? 1 : m_alphas_cumprod[0];
}

void DDIMScheduler::set_timesteps(size_t num_inference_steps) {
    m_timesteps.clear();

    if (num_inference_steps > m_config.num_train_timesteps) {
        OPENVINO_THROW("`num_inference_steps cannot be larger than m_config.num_train_timesteps");

    }

    m_num_inference_steps = num_inference_steps;

    // TODO: add linspace and trailing
    if (m_config.timestep_spacing == TimestepSpacing::LEADING) {
        // step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            // # creates integer timesteps by multiplying by ratio
            // # casting to int to avoid issues when num_inference_step is power of 3
        // timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        // timesteps += self.config.steps_offset

        size_t step_ratio = m_config.num_train_timesteps  / m_num_inference_steps;

        for (size_t i = num_inference_steps - 1; i != -1; --i) {
            m_timesteps.push_back(i * step_ratio + m_config.steps_offset);
        }

    } else {
        OPENVINO_THROW("'timestep_spacing' must be 'LEADING'. Please, add support of other types");
    }

}

std::map<std::string, ov::Tensor> DDIMScheduler::step(ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step) {
    // noise_pred - model_output
    // latents - sample
    // inference_step

    size_t timestep = get_timesteps()[inference_step];

    // 1. get previous step value (=t-1)
    size_t prev_timestep = timestep - m_config.num_train_timesteps / m_num_inference_steps;

    // 2. compute alphas, betas
    float alpha_prod_t = m_alphas_cumprod[timestep];
    float alpha_prod_t_prev = (prev_timestep >= 0) ? m_alphas_cumprod[prev_timestep] : m_final_alpha_cumprod;
    float beta_prod_t = 1 - alpha_prod_t;

    // 3. compute predicted original sample from predicted noise also called
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

    // TODO:
    // 4. Clip or threshold "predicted x_0"
    // if m_config.thresholding:
    //         pred_original_sample = _threshold_sample(pred_original_sample)
    // elif m_config.clip_sample:
    //         pred_original_sample = pred_original_sample.clamp(
    //             -self.config.clip_sample_range, self.config.clip_sample_range
    //         )

    // 5. compute variance: "sigma_t(η)" -> see formula (16)
    // σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)

    float eta = 0.0f;
    float variance = get_variance(timestep, prev_timestep);
    float std_dev_t = eta * std::sqrt(variance);

    std::cout << "inference_step: " << timestep << " prev_timestep " << prev_timestep << std::endl;
    std::cout << "variance: " << variance << " std_dev_t: " << std_dev_t << std::endl;

    // Remove if it's unnecessary:
    // TODO:
    // if use_clipped_model_output: ...

    // 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    
    // std::vector<float> pred_sample_direction(pred_epsilon.size());
    // std::transform(pred_epsilon.begin(), pred_epsilon.end(), pred_sample_direction.begin(), [alpha_prod_t_prev, std_dev_t](auto x) {
    //     return std::sqrt(1 - alpha_prod_t_prev - std::pow(std_dev_t, 2)) * x;
    // });
    //(1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

    std::vector<float> pred_sample_direction = pred_epsilon;
    for (size_t i = 0; i < pred_sample_direction.size(); ++i){
        pred_sample_direction[i] *= std::sqrt(1 - alpha_prod_t_prev - std::pow(std_dev_t, 2) );
     }

    std::cout << "pred_sample_direction" << std::endl;
    for (size_t i = 0; i < 10; ++i) {
        std::cout << pred_sample_direction[i] << " ";
    }
    std::cout <<  std::endl;

    // 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    ov::Tensor prev_sample(latents.get_element_type(), latents.get_shape());
    float* prev_sample_data = prev_sample.data<float>();
    for (size_t i = 0; i < prev_sample.get_size(); ++i) {
        prev_sample_data[i] = std::sqrt(alpha_prod_t_prev) * pred_original_sample[i] + pred_sample_direction[i];
    }
    // prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

    // TODO: if eta > 0:

    std::cout << "prev_sample" << prev_sample.get_element_type() << std::endl;
    for (size_t i = 0; i < 10; ++i) {
        std::cout << prev_sample_data[i] << " ";
    }
    std::cout << std::endl;

    std::map<std::string, ov::Tensor> result{{"latent", prev_sample}};

    return result;
}

std::vector<int64_t> DDIMScheduler::get_timesteps() const {
    return m_timesteps;
}

float DDIMScheduler::get_init_noise_sigma() const {
    return 1.0f;
}

void DDIMScheduler::scale_model_input(ov::Tensor sample, size_t inference_step) {
    return;
}

float DDIMScheduler::get_variance(size_t timestep, size_t prev_timestep) {
    float alpha_prod_t = m_alphas_cumprod[timestep];
    float alpha_prod_t_prev = (prev_timestep >= 0) ? m_alphas_cumprod[prev_timestep] : m_final_alpha_cumprod;
    float beta_prod_t = 1 - alpha_prod_t;
    float beta_prod_t_prev = 1 - alpha_prod_t_prev;

    float variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev);

    return variance;
}


} // namespace genai
} // namespace ov