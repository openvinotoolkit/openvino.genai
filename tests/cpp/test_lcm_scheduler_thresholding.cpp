// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "image_generation/schedulers/lcm.hpp"
#include "openvino/genai/image_generation/generation_config.hpp"

namespace {

using ov::genai::LCMScheduler;

// Epsilon prediction followed by dynamic thresholding, computed independently of the scheduler.
std::vector<double> reference_denoised(const LCMScheduler::Config& config,
                                       int64_t timestep,
                                       const std::vector<float>& sample,
                                       const std::vector<float>& model_output) {
    double beta_start_sqrt = std::sqrt(static_cast<double>(config.beta_start));
    double beta_end_sqrt = std::sqrt(static_cast<double>(config.beta_end));
    double alpha_cumprod = 1.0;
    for (int64_t i = 0; i <= timestep; ++i) {
        double beta = beta_start_sqrt + (beta_end_sqrt - beta_start_sqrt) * i / (config.num_train_timesteps - 1);
        alpha_cumprod *= 1.0 - beta * beta;
    }
    const double alpha_sqrt = std::sqrt(alpha_cumprod);
    const double beta_sqrt = std::sqrt(1.0 - alpha_cumprod);

    const size_t size = sample.size();
    std::vector<double> x0(size);
    for (size_t i = 0; i < size; ++i) {
        x0[i] = (sample[i] - beta_sqrt * model_output[i]) / alpha_sqrt;
    }

    std::vector<double> abs_x0(size);
    for (size_t i = 0; i < size; ++i) {
        abs_x0[i] = std::abs(x0[i]);
    }
    std::sort(abs_x0.begin(), abs_x0.end());
    const size_t quantile_index =
        std::min(static_cast<size_t>(std::lround(config.dynamic_thresholding_ratio * size)), size - 1);
    const double scale = std::clamp(abs_x0[quantile_index], 1.0, static_cast<double>(config.sample_max_value));
    for (auto& value : x0) {
        value = std::clamp(value, -scale, scale) / scale;
    }

    const double scaled_timestep = timestep * static_cast<double>(config.timestep_scaling);
    const double sigma_data_squared = 0.25;
    const double denominator = scaled_timestep * scaled_timestep + sigma_data_squared;
    const double c_skip = sigma_data_squared / denominator;
    const double c_out = scaled_timestep / std::sqrt(denominator);

    std::vector<double> denoised(size);
    for (size_t i = 0; i < size; ++i) {
        denoised[i] = c_out * x0[i] + c_skip * sample[i];
    }
    return denoised;
}

}  // namespace

TEST(LCMSchedulerThresholding, StepAppliesDynamicThresholding) {
    LCMScheduler::Config config;
    config.thresholding = true;
    config.sample_max_value = 1.0f;

    const ov::Shape shape{1, 4, 8, 8};
    const size_t size = ov::shape_size(shape);
    std::mt19937 rng(1234);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    ov::Tensor latent(ov::element::f32, shape);
    ov::Tensor model_output(ov::element::f32, shape);
    for (size_t i = 0; i < size; ++i) {
        latent.data<float>()[i] = 2.0f * normal(rng);  // wide enough for the threshold to clamp
        model_output.data<float>()[i] = normal(rng);
    }

    LCMScheduler scheduler(config);
    scheduler.set_timesteps(4, 1.0f);
    const int64_t timestep = scheduler.get_timesteps()[0];

    const std::vector<float> latent_values(latent.data<float>(), latent.data<float>() + size);
    const std::vector<float> model_output_values(model_output.data<float>(), model_output.data<float>() + size);
    const auto expected = reference_denoised(config, timestep, latent_values, model_output_values);

    auto generator = std::make_shared<ov::genai::CppStdGenerator>(42);
    const auto result = scheduler.step(model_output, latent, 0, generator);
    const float* denoised = result.at("denoised").data<float>();

    ASSERT_EQ(result.at("denoised").get_size(), size);
    for (size_t i = 0; i < size; ++i) {
        EXPECT_NEAR(denoised[i], expected[i], 1e-3 * (1.0 + std::abs(expected[i]))) << "index " << i;
    }
}
