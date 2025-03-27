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

class EulerAncestralDiscreteScheduler : public IScheduler {
public:
    struct Config {
        int32_t num_train_timesteps = 1000;
        float beta_start = 0.0001f, beta_end = 0.02f;
        BetaSchedule beta_schedule = BetaSchedule::LINEAR;
        std::vector<float> trained_betas = {};
        size_t steps_offset = 0;
        PredictionType prediction_type = PredictionType::EPSILON;
        TimestepSpacing timestep_spacing = TimestepSpacing::LEADING;
        bool rescale_betas_zero_snr = false;

        Config() = default;
        explicit Config(const std::filesystem::path& scheduler_config_path);
    };

    explicit EulerAncestralDiscreteScheduler(const std::filesystem::path& scheduler_config_path);
    explicit EulerAncestralDiscreteScheduler(const Config& scheduler_config);

    void set_timesteps(size_t num_inference_steps, float strength) override;

    std::vector<std::int64_t> get_timesteps() const override;

    float get_init_noise_sigma() const override;

    void scale_model_input(ov::Tensor sample, size_t inference_step) override;

    std::map<std::string, ov::Tensor> step(ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step, std::shared_ptr<Generator> generator) override;

    void add_noise(ov::Tensor init_latent, ov::Tensor noise, int64_t latent_timestep) const override;

private:
    Config m_config;

    std::vector<float> m_alphas_cumprod, m_sigmas;
    std::vector<int64_t> m_timesteps, m_schedule_timesteps;
    size_t m_num_inference_steps;

    int m_step_index, m_begin_index;
    bool m_is_scale_input_called;

    size_t _index_for_timestep(int64_t timestep) const;
};

} // namespace genai
} // namespace ov
