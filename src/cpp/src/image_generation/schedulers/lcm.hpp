// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>
#include <random>
#include <vector>

#include "image_generation/schedulers/types.hpp"
#include "image_generation/schedulers/ischeduler.hpp"

namespace ov {
namespace genai {

class LCMScheduler : public IScheduler {
public:
    // values from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/schedulers/scheduling_lcm.py#L197-L214
    struct Config {
        size_t num_train_timesteps = 1000;
        float beta_start = 0.00085f, beta_end = 0.012f;
        BetaSchedule beta_schedule = BetaSchedule::SCALED_LINEAR;
        std::vector<float> trained_betas = {};
        size_t original_inference_steps = 50;
        bool clip_sample = false;
        float clip_sample_range = 1.0f;
        bool set_alpha_to_one = true;
        size_t steps_offset = 0;
        PredictionType prediction_type = PredictionType::EPSILON;
        bool thresholding = false;
        float dynamic_thresholding_ratio = 0.995f;
        float sample_max_value = 1.0f;
        TimestepSpacing timestep_spacing = TimestepSpacing::LEADING;
        float timestep_scaling = 10.0f;
        bool rescale_betas_zero_snr = false;

        Config() = default;
        explicit Config(const std::filesystem::path& scheduler_config_path);
    };

    explicit LCMScheduler(const std::filesystem::path& scheduler_config_path);
    explicit LCMScheduler(const Config& scheduler_config);

    void set_timesteps(size_t num_inference_steps, float strength) override;

    std::vector<std::int64_t> get_timesteps() const override;

    float get_init_noise_sigma() const override;

    void scale_model_input(ov::Tensor sample, size_t inference_step) override;

    std::map<std::string, ov::Tensor> step(ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step, std::shared_ptr<Generator> generator) override;

    void add_noise(ov::Tensor init_latent, ov::Tensor noise, int64_t latent_timestep) const override;

private:
    Config m_config;

    std::vector<float> m_alphas_cumprod;
    float m_final_alpha_cumprod;
    size_t m_num_inference_steps;
    float m_sigma_data;

    std::vector<int64_t> m_timesteps;

    std::vector<float> threshold_sample(const std::vector<float>& flat_sample);
};

} // namespace genai
} // namespace ov
