// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "flow_match_euler_discrete.hpp"

namespace ov::genai {

class ZImageFlowMatchEulerDiscreteScheduler : public FlowMatchEulerDiscreteScheduler {
public:
    explicit ZImageFlowMatchEulerDiscreteScheduler(const std::filesystem::path& scheduler_config_path, const std::string &device)
        : FlowMatchEulerDiscreteScheduler(scheduler_config_path) {
            init_step_process(device);
        }
    explicit ZImageFlowMatchEulerDiscreteScheduler(const Config& scheduler_config, const std::string &device)
        : FlowMatchEulerDiscreteScheduler(scheduler_config) {
            init_step_process(device);
        }

    void set_timesteps(size_t image_seq_len, size_t num_inference_steps, float strength) override;
    void set_sigma_min(float sigma_min);
    std::map<std::string, ov::Tensor> step(ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step, std::shared_ptr<Generator> generator) override;

private:
    void init_step_process(const std::string &device);

    InferRequest m_step_infer_request;
};

}
