// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "flow_match_euler_discrete.hpp"

namespace ov::genai {

class ZImageFlowMatchEulerDiscreteScheduler : public FlowMatchEulerDiscreteScheduler {
public:
    explicit ZImageFlowMatchEulerDiscreteScheduler(const std::filesystem::path& scheduler_config_path)
        : FlowMatchEulerDiscreteScheduler(scheduler_config_path) {}
    explicit ZImageFlowMatchEulerDiscreteScheduler(const Config& scheduler_config)
        : FlowMatchEulerDiscreteScheduler(scheduler_config) {}

    void set_timesteps(size_t image_seq_len, size_t num_inference_steps, float strength) override;
    void set_sigma_min(float sigma_min);

private:

};

}
