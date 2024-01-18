// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <list>

#include "scheduler.hpp"

class LMSDiscreteScheduler : public Scheduler {
public:
    LMSDiscreteScheduler(int32_t num_train_timesteps = 1000,
                         float beta_start = 0.00085f,
                         float beta_end = 0.012f,
                         BetaSchedule beta_schedule = BetaSchedule::SCALED_LINEAR,
                         PredictionType prediction_type = PredictionType::EPSILON,
                         const std::vector<float>& trained_betas = {});

    void set_timesteps(size_t num_inference_steps) override;

    std::vector<std::int64_t> get_timesteps() const override;

    float get_init_noise_sigma() const override;

    void scale_model_input(ov::Tensor sample, size_t inference_step) override;

    std::map<std::string, ov::Tensor> step(ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step) override;

private:
    std::vector<float> m_log_sigmas;
    std::vector<float> m_sigmas;
    std::vector<int64_t> m_timesteps;
    std::list<std::vector<float>> m_derivative_list;

    int64_t _sigma_to_t(float sigma) const;
};
