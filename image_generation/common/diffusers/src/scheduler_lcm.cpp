// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <random>
#include <fstream>
#include <iterator>

#include "scheduler_lcm.hpp"

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

std::vector<float> read_vector_from_txt(std::string& file_name) {
    std::ifstream input_data(file_name, std::ifstream::in);
    std::istream_iterator<float> start(input_data), end;
    std::vector<float> res(start, end);
    return res;
}

LCMScheduler::LCMScheduler(size_t num_train_timesteps,
                           float beta_start,
                           float beta_end,
                           BetaSchedule beta_schedule,
                           PredictionType prediction_type,
                           const std::vector<float>& trained_betas,
                           size_t original_inference_steps,
                           bool set_alpha_to_one,
                           float timestep_scaling,
                           bool thresholding,
                           bool clip_sample,
                           float clip_sample_range,
                           float dynamic_thresholding_ratio,
                           float sample_max_value,
                           bool read_torch_noise,
                           uint32_t seed):
                           prediction_type_config(prediction_type),
                           num_train_timesteps_config(num_train_timesteps),
                           original_inference_steps_config(original_inference_steps),
                           timestep_scaling_config(timestep_scaling),
                           thresholding(thresholding),
                           clip_sample(clip_sample),
                           clip_sample_range(clip_sample_range),
                           dynamic_thresholding_ratio(dynamic_thresholding_ratio),
                           sample_max_value(sample_max_value),
                           read_torch_noise(read_torch_noise),
                           seed(seed),
                           gen(seed),
                           normal(0.0f, 1.0f) {

    sigma_data = 0.5f; // Default: 0.5

    std::vector<float> alphas, betas;

    if (!trained_betas.empty()) {
        auto betas = trained_betas;
    } else if (beta_schedule == BetaSchedule::LINEAR) {
        for (size_t i = 0; i < num_train_timesteps; i++) {
            betas.push_back(beta_start + (beta_end - beta_start) * i / (num_train_timesteps - 1));
        }
    } else if (beta_schedule == BetaSchedule::SCALED_LINEAR) {
        float start = sqrt(beta_start);
        float end = sqrt(beta_end);
        std::vector<float> temp = linspace<float, float>(start, end, num_train_timesteps, true);
        for (float b : temp) {
            betas.push_back(b * b);
        }
    } else {
        OPENVINO_THROW("'beta_schedule' must be one of 'EPSILON' or 'SCALED_LINEAR'");
    }
    for (float b : betas) {
        alphas.push_back(1.0f - b);
    }
    for (size_t i = 1; i <= alphas.size(); i++) {
        float alpha_cumprod =
            std::accumulate(std::begin(alphas), std::begin(alphas) + i, 1.0, std::multiplies<float>{});
        alphas_cumprod.push_back(alpha_cumprod);
    }

    final_alpha_cumprod = set_alpha_to_one ? 1 : alphas_cumprod[0];
}

void LCMScheduler::set_timesteps(size_t num_inference_steps) {
    float strength = 1.0f;
    // LCM Timesteps Setting
    size_t k = num_train_timesteps_config / original_inference_steps_config;

    size_t origin_timesteps_size = original_inference_steps_config * strength;
    std::vector<size_t> lcm_origin_timesteps(origin_timesteps_size);
    std::iota(lcm_origin_timesteps.begin(), lcm_origin_timesteps.end(), 1);
    std::transform(lcm_origin_timesteps.begin(), lcm_origin_timesteps.end(), lcm_origin_timesteps.begin(), [&k](auto& x) {
        return x * k - 1;
    });

    size_t skipping_step = origin_timesteps_size / num_inference_steps;
    assert(skipping_step >= 1 && "The combination of `original_steps x strength` is smaller than `num_inference_steps`");

    this->num_inference_steps = num_inference_steps;
    // LCM Inference Steps Schedule
    std::reverse(lcm_origin_timesteps.begin(),lcm_origin_timesteps.end());

    // v1. based on https://github.com/huggingface/diffusers/blame/2a7f43a73bda387385a47a15d7b6fe9be9c65eb2/src/diffusers/schedulers/scheduling_lcm.py#L387
    std::vector<size_t> inference_indices = linspace<size_t, float>(0, origin_timesteps_size, num_inference_steps);
    for (size_t i : inference_indices){
        m_timesteps.push_back(lcm_origin_timesteps[i]);
    }

    // // v2. based on diffusers==0.23.1
    // std::vector<float> temp;
    // for(size_t i = 0; i < lcm_origin_timesteps.size(); i+=skipping_step)
    //     temp.push_back(lcm_origin_timesteps[i]);
    // for(size_t i = 0; i < num_inference_steps; i++)
    //     m_timesteps.push_back(temp[i]);

}

std::map<std::string, ov::Tensor> LCMScheduler::step(ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step) {
    ov::Tensor timestep(ov::element::i64, {1}, &m_timesteps[inference_step]);

    float* noise_pred_data = noise_pred.data<float>();
    float* latents_data = latents.data<float>();

    // 1. get previous step value
    int64_t prev_step_index = inference_step + 1;
    int64_t curr_step = m_timesteps[inference_step];
    int64_t prev_timestep = prev_step_index < static_cast<int64_t>(m_timesteps.size()) ? m_timesteps[prev_step_index] : curr_step;

    // 2. compute alphas, betas
    float alpha_prod_t = alphas_cumprod[curr_step];
    float alpha_prod_t_prev = (prev_timestep >= 0) ? alphas_cumprod[prev_timestep] : final_alpha_cumprod;
    float alpha_prod_t_sqrt = std::sqrt(alpha_prod_t);
    float alpha_prod_t_prev_sqrt = std::sqrt(alpha_prod_t_prev);
    float beta_prod_t_sqrt = std::sqrt(1 - alpha_prod_t);
    float beta_prod_t_prev_sqrt = std::sqrt(1 - alpha_prod_t_prev);

    // 3. Get scalings for boundary conditions
    // get_scalings_for_boundary_condition_discrete(...)
    float scaled_timestep = curr_step * timestep_scaling_config;
    float c_skip = std::pow(sigma_data, 2) / (std::pow(scaled_timestep, 2) + std::pow(sigma_data, 2));
    float c_out = scaled_timestep / std::sqrt((std::pow(scaled_timestep, 2) + std::pow(sigma_data, 2)));

    // 4. Compute the predicted original sample x_0 based on the model parameterization
    std::vector<float> predicted_original_sample(latents.get_size());
    // "epsilon" by default
    if (prediction_type_config == PredictionType::EPSILON) {
        for (std::size_t i = 0; i < latents.get_size(); ++i) {
            predicted_original_sample[i] = (latents_data[i] - beta_prod_t_sqrt * noise_pred_data[i]) / alpha_prod_t_sqrt;
        }
    }

    // 5. Clip or threshold "predicted x_0"
    if (thresholding) {
        predicted_original_sample = threshold_sample(predicted_original_sample);
    } else if (clip_sample) {
        for (float& value : predicted_original_sample) {
            value = std::clamp(value, - clip_sample_range, clip_sample_range);
        }
    }

    // 6. Denoise model output using boundary conditions
    ov::Tensor denoised(latents.get_element_type(), latents.get_shape());
    float* denoised_data = denoised.data<float>();
    for (std::size_t i = 0; i < denoised.get_size(); ++i) {
        denoised_data[i] = c_out * predicted_original_sample[i] + c_skip * latents_data[i];
    }

    /// 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
    // Noise is not used on the final timestep of the timestep schedule.
    // This also means that noise is not used for one-step sampling.
    ov::Tensor prev_sample(latents.get_element_type(), latents.get_shape());
    float* prev_sample_data = prev_sample.data<float>();

    if (inference_step != num_inference_steps - 1) {
        std::vector<float> noise;
        if (read_torch_noise) {
            std::string noise_file = "./latents/torch_noise_step_" + std::to_string(inference_step) + ".txt";
            noise = read_vector_from_txt(noise_file);
        } else {
            noise = randn_function(noise_pred.get_size(), seed);
        }        

        for (std::size_t i = 0; i < noise_pred.get_size(); ++i) {
            prev_sample_data[i] = alpha_prod_t_prev_sqrt * denoised_data[i] + beta_prod_t_prev_sqrt * noise[i];
        }

    } else {
        std::copy_n(denoised_data, denoised.get_size(), prev_sample_data);
    }

    std::map<std::string, ov::Tensor> result{{"latent", prev_sample}, {"denoised", denoised}};
    return result;
}

std::vector<int64_t> LCMScheduler::get_timesteps() const {
    return m_timesteps;
}

float LCMScheduler::get_init_noise_sigma() const {
    return 1.0f;
}

void LCMScheduler::scale_model_input(ov::Tensor sample, size_t inference_step) {
    return;
}

// Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
std::vector<float> LCMScheduler::threshold_sample(const std::vector<float>& flat_sample) {
    /*
    "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
    prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
    s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
    pixels from saturation at each step. We find that dynamic thresholding results in significantly better
    photorealism as well as better image-text alignment, especially when using very large guidance weights."
    https://arxiv.org/abs/2205.11487
    */

    std::vector<float> thresholded_sample;
    // Calculate abs
    std::vector<float> abs_sample(flat_sample.size());
    std::transform(flat_sample.begin(), flat_sample.end(), abs_sample.begin(), [](float val) { return std::abs(val); });

    // Calculate s, the quantile threshold
    std::sort(abs_sample.begin(), abs_sample.end());
    const int s_index = std::min(static_cast<int>(std::round(dynamic_thresholding_ratio * flat_sample.size())),
                                static_cast<int>(flat_sample.size()) - 1);
    float s = abs_sample[s_index];
    s = std::clamp(s, 1.0f, sample_max_value);

    // Threshold and normalize the sample
    for (float& value : thresholded_sample) {
        value = std::clamp(value, -s, s) / s;
    }

    return thresholded_sample;
}

std::vector<float> LCMScheduler::randn_function(uint32_t size, uint32_t seed = 42) {
    std::vector<float> noise(size);
    {
        std::for_each(noise.begin(), noise.end(), [&](float& x) {
            x = normal(gen);
        });
    }
    return noise;
}
