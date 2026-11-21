// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>

#include "image_generation/schedulers/ddim.hpp"
#include "image_generation/schedulers/euler_ancestral_discrete.hpp"
#include "image_generation/schedulers/euler_discrete.hpp"
#include "image_generation/schedulers/flow_match_euler_discrete.hpp"
#include "image_generation/schedulers/lcm.hpp"
#include "image_generation/schedulers/pndm.hpp"

using namespace ov::genai;

template <typename SchedulerT>
std::shared_ptr<IScheduler> make_scheduler() {
    return std::make_shared<SchedulerT>(typename SchedulerT::Config{});
}

class ImageGenerationSchedulerZeroStepsTest
    : public ::testing::TestWithParam<std::shared_ptr<IScheduler> (*)()> {};

TEST_P(ImageGenerationSchedulerZeroStepsTest, SetTimestepsRejectsZeroInferenceSteps) {
    auto scheduler = GetParam()();
    EXPECT_THROW(scheduler->set_timesteps(0, 1.0f), ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(AllSchedulers,
                         ImageGenerationSchedulerZeroStepsTest,
                         ::testing::Values(&make_scheduler<DDIMScheduler>,
                                           &make_scheduler<EulerAncestralDiscreteScheduler>,
                                           &make_scheduler<EulerDiscreteScheduler>,
                                           &make_scheduler<FlowMatchEulerDiscreteScheduler>,
                                           &make_scheduler<LCMScheduler>,
                                           &make_scheduler<PNDMScheduler>));

TEST(ImageGenerationSchedulerValidation, FlowMatchSetTimestepsOverloadsRejectZeroInferenceSteps) {
    FlowMatchEulerDiscreteScheduler scheduler{FlowMatchEulerDiscreteScheduler::Config{}};
    EXPECT_THROW(scheduler.set_timesteps(4096, 0, 1.0f), ov::Exception);
    EXPECT_THROW(scheduler.set_timesteps_with_mu(0.0, 0, 1.0f), ov::Exception);
}

TEST(ImageGenerationSchedulerValidation, LCMStepRejectsZeroImagesPerPrompt) {
    LCMScheduler scheduler{LCMScheduler::Config{}};
    scheduler.set_timesteps(4, 1.0f);

    ov::Tensor latents(ov::element::f32, ov::Shape{0, 4, 8, 8});
    ov::Tensor noise_pred(ov::element::f32, ov::Shape{0, 4, 8, 8});
    EXPECT_THROW(scheduler.step(noise_pred, latents, 0, nullptr), ov::Exception);
}
