// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "scheduler.hpp"

class LCMScheduler : public Scheduler {
public:
    LCMScheduler(size_t num_train_timesteps = 1000,
                 float beta_start = 0.00085f,
                 float beta_end = 0.012f,
                 BetaSchedule beta_schedule = BetaSchedule::SCALED_LINEAR,
                 PredictionType prediction_type = PredictionType::EPSILON,
                 const std::vector<float>& trained_betas = {},
                 size_t original_inference_steps = 50, 
                 bool set_alpha_to_one = true,
                 float timestep_scaling = 10.0f,
                 bool thresholding = false,
                 bool clip_sample = false,
                 float clip_sample_range = 1.0f,
                 float dynamic_thresholding_ratio = 0.995f,
                 float sample_max_value = 1.0f,
                 bool read_torch_noise = false);

    void set_timesteps(size_t num_inference_steps) override;

    std::vector<std::int64_t> get_timesteps() const override;

    float get_init_noise_sigma() const override;

    void scale_model_input(ov::Tensor sample, size_t inference_step) override;

    std::map<std::string, ov::Tensor> step(ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step) override;

private:
    std::vector<int64_t> m_timesteps;
    std::vector<float> alphas_cumprod;
    PredictionType prediction_type_config;
    float final_alpha_cumprod;
    size_t num_train_timesteps_config;
    size_t original_inference_steps_config;
    size_t num_inference_steps;
    float timestep_scaling_config;
    float sigma_data;
    bool thresholding;
    bool clip_sample;
    float clip_sample_range;
    float dynamic_thresholding_ratio;
    float sample_max_value;
    bool read_torch_noise;

    std::vector<float> threshold_sample(const std::vector<float>& flat_sample);
};
