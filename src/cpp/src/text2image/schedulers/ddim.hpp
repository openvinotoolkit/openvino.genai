// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <list>
#include <string>

#include "text2image/schedulers/types.hpp"
#include "text2image/schedulers/ischeduler.hpp"

namespace ov {
namespace genai {

class DDIMScheduler : public IScheduler {
public:
    struct Config {
        int32_t num_train_timesteps = 1000;
        float beta_start = 0.0001f, beta_end = 0.02f;
        BetaSchedule beta_schedule = BetaSchedule::SCALED_LINEAR;
        std::vector<float> trained_betas = {};
        bool clip_sample = true, set_alpha_to_one = true;
        size_t steps_offset = 0;
        PredictionType prediction_type = PredictionType::EPSILON;
        bool thresholding = false;
        float dynamic_thresholding_ratio = 0.995f, clip_sample_range = 1.0f, sample_max_value = 1.0f;
        TimestepSpacing timestep_spacing = TimestepSpacing::LEADING;
        bool rescale_betas_zero_snr = false;

        Config() = default;
        explicit Config(const std::string& scheduler_config_path);
    };

    explicit DDIMScheduler(const std::string scheduler_config_path);
    explicit DDIMScheduler(const Config& scheduler_config);

    void set_timesteps(size_t num_inference_steps) override;

    std::vector<std::int64_t> get_timesteps() const override;

    float get_init_noise_sigma() const override;

    void scale_model_input(ov::Tensor sample, size_t inference_step) override;

    std::map<std::string, ov::Tensor> step(ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step) override;

private:
    Config m_config;

    std::vector<float> m_alphas_cumprod;
    float m_final_alpha_cumprod;

    size_t m_num_inference_steps;
    std::vector<int64_t> m_timesteps;
};

} // namespace genai
} // namespace ov
