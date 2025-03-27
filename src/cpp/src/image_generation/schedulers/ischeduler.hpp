// Copyright (C) 2023-2025 Intel Corporation
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

    virtual float get_init_noise_sigma() const = 0;

    virtual void scale_model_input(ov::Tensor sample, size_t inference_step) = 0;

    virtual std::map<std::string, ov::Tensor> step(
        ov::Tensor noise_pred, ov::Tensor latents, size_t inference_step, std::shared_ptr<Generator> generator) = 0;

    virtual void add_noise(ov::Tensor init_latent, ov::Tensor noise, int64_t latent_timestep) const = 0;

    virtual void set_timesteps(size_t image_seq_len, size_t num_inference_steps, float strength) {
        OPENVINO_THROW("Scheduler doesn't support `set_timesteps(size_t image_seq_len, size_t num_inference_steps, float strength)` method");
    }

    virtual std::vector<std::int64_t> get_timesteps() const {
         OPENVINO_THROW("Scheduler doesn't support int timesteps");
    }

    virtual std::vector<float> get_float_timesteps() {
        OPENVINO_THROW("Scheduler doesn't support float timesteps");
    }

    virtual void scale_noise(ov::Tensor sample, float timestep, ov::Tensor noise) {
        OPENVINO_THROW("Scheduler doesn't support `scale_noise` method");
    }

    virtual void set_begin_index(size_t begin_index) {};

};

} // namespace genai
} // namespace ov
