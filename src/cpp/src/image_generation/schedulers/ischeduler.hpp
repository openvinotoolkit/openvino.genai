// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>
#include <map>

#include "openvino/genai/image_generation/scheduler.hpp"
#include "openvino/genai/image_generation/generation_config.hpp"

#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace genai {

class IScheduler : public Scheduler {
public:
    virtual void set_timesteps(size_t num_inference_steps, float strength) = 0;

    virtual void set_timesteps_with_sigma(std::vector<float> sigma, float mu) = 0;

    virtual std::vector<std::int64_t> get_timesteps() const = 0;

    virtual std::vector<float> get_float_timesteps() const = 0;

    virtual float get_init_noise_sigma() const = 0;

    virtual void scale_model_input(ov::Tensor sample, size_t inference_step) = 0;

    virtual std::map<std::string, ov::Tensor> step(
        ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step, std::shared_ptr<Generator> generator) = 0;

    virtual void add_noise(ov::Tensor init_latent, std::shared_ptr<Generator> generator) const = 0;
};

} // namespace genai
} // namespace ov
