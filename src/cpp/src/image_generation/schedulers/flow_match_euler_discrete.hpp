// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <list>
#include <string>

#include "image_generation/schedulers/types.hpp"
#include "image_generation/schedulers/ischeduler.hpp"

namespace ov {
namespace genai {

class FlowMatchEulerDiscreteScheduler : public IScheduler {
public:
    struct Config {
        int32_t num_train_timesteps = 1000;
        float shift = 1.0f;
        bool use_dynamic_shifting = false;
        float base_shift = 0.5f, max_shift = 1.15f;
        int32_t base_image_seq_len = 256, max_image_seq_len = 4096;

        Config() = default;
        explicit Config(const std::filesystem::path& scheduler_config_path);
    };

    explicit FlowMatchEulerDiscreteScheduler(const std::filesystem::path& scheduler_config_path);
    explicit FlowMatchEulerDiscreteScheduler(const Config& scheduler_config);

    void set_timesteps(size_t num_inference_steps, float strength) override;

    void set_timesteps(size_t image_seq_len, size_t num_inference_steps, float strength) override;

    std::vector<float> get_float_timesteps() override;

    float get_init_noise_sigma() const override;

    void scale_model_input(ov::Tensor sample, size_t inference_step) override;

    std::map<std::string, ov::Tensor> step(ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step, std::shared_ptr<Generator> generator) override;

    void add_noise(ov::Tensor init_latent, ov::Tensor noise, int64_t latent_timestep) const override;

    void scale_noise(ov::Tensor sample, float timestep, ov::Tensor noise) override;

    void set_begin_index(size_t begin_index) override;

private:
    Config m_config;

    std::vector<float> m_sigmas;
    std::vector<float> m_timesteps, m_schedule_timesteps;

    float m_sigma_min, m_sigma_max;
    float m_strength;
    size_t m_step_index, m_begin_index;
    size_t m_num_inference_steps;

    void init_step_index();
    double sigma_to_t(double simga);
    size_t _index_for_timestep(float timestep);
    float calculate_shift(size_t image_seq_len);
};

} // namespace genai
} // namespace ov
