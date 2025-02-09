// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <list>
#include <string>

#include "image_generation/schedulers/types.hpp"
#include "image_generation/schedulers/ischeduler.hpp"

namespace ov {
namespace genai {

class PNDMScheduler : public IScheduler {
public:
    struct Config {
        int32_t num_train_timesteps = 1000;
        float beta_start = 0.0001f, beta_end = 0.02f;
        BetaSchedule beta_schedule = BetaSchedule::LINEAR;
        std::vector<float> trained_betas = {};
        bool set_alpha_to_one = false, skip_prk_steps = false;
        PredictionType prediction_type = PredictionType::EPSILON;
        TimestepSpacing timestep_spacing = TimestepSpacing::LEADING;
        size_t steps_offset = 0;

        Config() = default;
        explicit Config(const std::filesystem::path& scheduler_config_path);
    };

    explicit PNDMScheduler(const std::filesystem::path& scheduler_config_path);
    explicit PNDMScheduler(const Config& scheduler_config);

    void set_timesteps(size_t num_inference_steps, float strength) override;

    std::vector<std::int64_t> get_timesteps() const override;

    float get_init_noise_sigma() const override;

    void scale_model_input(ov::Tensor sample, size_t inference_step) override;

    std::map<std::string, ov::Tensor> step(ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step, std::shared_ptr<Generator> generator) override;

    void add_noise(ov::Tensor init_latent, ov::Tensor noise, int64_t timestep) const override;

private:
    Config m_config;

    float m_final_alpha_cumprod;
    size_t m_num_inference_steps;
    size_t m_counter;

    std::vector<float> m_alphas_cumprod;
    std::vector<int64_t> m_timesteps;
    std::vector<int64_t> m_prk_timesteps;
    std::vector<int64_t> m_plms_timesteps;
    std::vector<ov::Tensor> m_ets;

    ov::Tensor m_cur_sample;

    std::map<std::string, ov::Tensor> step_plms(ov::Tensor model_output, ov::Tensor sample, size_t timestep);
    ov::Tensor get_prev_sample(ov::Tensor sample, size_t timestep, int prev_timestep, ov::Tensor model_output);
};

} // namespace genai
} // namespace ov
