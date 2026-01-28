// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "../utils/utils.hpp"
#include "module_genai/modules/unipc_multistep_scheduler.hpp"

using namespace ov::genai::module;

struct UniPCMultistepSchedulerTestData {
    UniPCMultistepScheduler::Config config;
    std::vector<float> expected_sigmas;
    std::vector<int64_t> expected_timesteps;
    std::vector<float> expected_latents;
};

namespace TEST_DATA {

std::vector<UniPCMultistepSchedulerTestData> unipc_multistep_scheduler_test_data() {
    std::vector<UniPCMultistepSchedulerTestData> datas;

    UniPCMultistepSchedulerTestData flow_sigma_data {};
    flow_sigma_data.config.use_flow_sigmas = true;
    flow_sigma_data.config.flow_shift = 3.0f;
    flow_sigma_data.config.prediction_type = UniPCMultistepScheduler::PredictionType::FLOW_PREDICTION;
    flow_sigma_data.expected_sigmas = {
        0.99966645, 0.96394110, 0.92272168, 0.87463522, 0.81780970, 0.74962479,
        0.66629612, 0.56214833, 0.42826521, 0.24979164, 0.00000000
    };
    flow_sigma_data.expected_timesteps = {
        999, 963, 922, 874, 817, 749, 666, 562, 428, 249
    };
    flow_sigma_data.expected_latents = {
        0.00000000, 0.00570375, 0.01140750, 0.01711125, 0.02281500, 0.02851875,
        0.03422250, 0.03992625, 0.04563000, 0.05133375, 0.05703750, 0.06274125,
        0.06844500, 0.07414875, 0.07985250, 0.08555625, 0.09126000, 0.09696375,
        0.10266750, 0.10837125
    };
    datas.push_back(flow_sigma_data);

    UniPCMultistepSchedulerTestData karras_sigma_data {};
    karras_sigma_data.config.use_karras_sigmas = true;
    karras_sigma_data.config.flow_shift = 3.0f;
    karras_sigma_data.config.prediction_type = UniPCMultistepScheduler::PredictionType::FLOW_PREDICTION;
    karras_sigma_data.expected_sigmas = {
        1.57407272e+02, 8.57107315e+01, 4.40476494e+01, 2.11056042e+01,
        9.27474022e+00, 3.65264034e+00, 1.24636030e+00, 3.49706143e-01,
        7.39302859e-02, 1.00013297e-02, 0.00000000e+00
    };
    karras_sigma_data.expected_timesteps = {
        999, 937, 864, 775, 663, 511, 301, 102,  18,   0
    };
    karras_sigma_data.expected_latents = {
        0.00000000, -0.03659932, -0.07319865, -0.10979797, -0.14639730,
        -0.18299662, -0.21959594, -0.25619527, -0.29279459, -0.32939392,
        -0.36599324, -0.40259256, -0.43919189, -0.47579121, -0.51239054,
        -0.54898986, -0.58558919, -0.62218851, -0.65878783, -0.69538716
    };
    datas.push_back(karras_sigma_data);

    UniPCMultistepSchedulerTestData karras_flow_sigma_data {};
    karras_flow_sigma_data.config.use_karras_sigmas = true;
    karras_flow_sigma_data.config.use_flow_sigmas = true;
    karras_flow_sigma_data.config.flow_shift = 3.0f;
    karras_flow_sigma_data.config.prediction_type = UniPCMultistepScheduler::PredictionType::FLOW_PREDICTION;
    karras_flow_sigma_data.expected_sigmas = {
        0.99368715, 0.98846740, 0.97780126, 0.95476258, 0.90267396, 0.78506827,
        0.55483544, 0.25909799, 0.06884086, 0.00990229, 0.00000000
    };
    karras_flow_sigma_data.expected_timesteps = {
        993, 988, 977, 954, 902, 785, 554, 259,  68,   9
    };
    karras_flow_sigma_data.expected_latents = {
        0.00000000, 0.00538023, 0.01076047, 0.01614070, 0.02152094, 0.02690117,
        0.03228141, 0.03766164, 0.04304188, 0.04842211, 0.05380235, 0.05918258,
        0.06456282, 0.06994305, 0.07532329, 0.08070352, 0.08608376, 0.09146399,
        0.09684423, 0.10222446
    };
    datas.push_back(karras_flow_sigma_data);

    return datas;
}

}

class UniPCMultistepSchedulerTest : public ::testing::TestWithParam<UniPCMultistepSchedulerTestData> {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<UniPCMultistepSchedulerTestData> & obj) {
        std::string test_type;
        if (obj.param.config.use_karras_sigmas && obj.param.config.use_flow_sigmas) {
            test_type = "karras_flow_sigmas";
        } else if (obj.param.config.use_karras_sigmas) {
            test_type = "karras_sigmas";
        } else if (obj.param.config.use_flow_sigmas) {
            test_type = "flow_sigmas";
        } else {
            test_type = "default_sigmas";
        }

        return test_type;
    }

    void run() {
        const UniPCMultistepSchedulerTestData &data = GetParam();
        const auto &config = data.config;
        UniPCMultistepScheduler scheduler(config);
        scheduler.set_timesteps(10);
        const auto &sigmas = scheduler.get_sigmas();
        const auto &timesteps = scheduler.get_timesteps();
        auto sigmas_data = sigmas.data<const float>();

        for (size_t i = 0; i < sigmas.get_size(); ++i) {
            EXPECT_NEAR(sigmas_data[i], data.expected_sigmas[i], 1e-4);
        }

        for (size_t i = 0; i < timesteps.size(); ++i) {
            EXPECT_NEAR(timesteps[i], data.expected_timesteps[i], 0);
        }

        ov::Tensor latents(ov::element::f32, {1, 16, 5, 64, 64});
        ov::Tensor noise_pred(ov::element::f32, {1, 16, 5, 64, 64});
        auto latents_data = latents.data<float>();
        auto noise_pred_data = noise_pred.data<float>();

        for (size_t i = 0; i < latents.get_size(); i++) {
            latents_data[i] = static_cast<float>(i % 100) * 0.01f;
        }

        size_t index = 0;
        for (const auto &t : timesteps) {
            for (size_t i = 0; i < noise_pred.get_size(); i++) {
                noise_pred_data[i] = static_cast<float>(i % 100) * 0.001f * (11.0f - static_cast<float>(index));
            }
            index++;
            latents = scheduler.step(
            noise_pred, t, latents)["prev_sample"];
        }

        latents_data = latents.data<float>();
        for (size_t i = 0; i < data.expected_latents.size(); i++) {
            EXPECT_NEAR(latents_data[i], data.expected_latents[i], 1e-4);
        }
    }
};

TEST_P(UniPCMultistepSchedulerTest, Test) {
    run();
}

namespace unipc_multistep_scheduler_test {

auto test_data = TEST_DATA::unipc_multistep_scheduler_test_data();

}

INSTANTIATE_TEST_SUITE_P(SchedulerTestSuite,
                         UniPCMultistepSchedulerTest,
                         ::testing::ValuesIn(unipc_multistep_scheduler_test::test_data),
                         UniPCMultistepSchedulerTest::get_test_case_name);

