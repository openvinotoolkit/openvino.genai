// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <vector>
#include <optional>
#include <cstring>

#include "openvino/genai/taylorseer_config.hpp"
#include "diffusion_caching/taylorseer_lite.hpp"

using ov::genai::TaylorSeerCacheConfig;
using ov::genai::TaylorSeerState;


class TaylorSeerCacheConfigTest : public ::testing::Test {
protected:
    void AssertConfigEquals(const TaylorSeerCacheConfig& config,
                           size_t expected_interval,
                           size_t expected_before,
                           int expected_after) {
        EXPECT_EQ(config.cache_interval, expected_interval);
        EXPECT_EQ(config.disable_cache_before_step, expected_before);
        EXPECT_EQ(config.disable_cache_after_step, expected_after);
    }
};

TEST_F(TaylorSeerCacheConfigTest, DefaultConstructor) {
    TaylorSeerCacheConfig config;
    // Default constructor uses member initializers: cache_interval=3, disable_before=6, disable_after=-2
    AssertConfigEquals(config, 3, 6, -2);
}

TEST_F(TaylorSeerCacheConfigTest, InvalidCacheIntervalZero) {
    TaylorSeerCacheConfig config{0, 0, -1};
    EXPECT_THROW(
        {
            TaylorSeerState state(config, 10);
            (void)state;
        },
        ov::Exception);
}

TEST_F(TaylorSeerCacheConfigTest, InvalidCacheIntervalOne) {
    TaylorSeerCacheConfig config{1, 0, -1};
    EXPECT_THROW(
        {
            TaylorSeerState state(config, 10);
            (void)state;
        },
        ov::Exception);
}

struct FullConstructorParams {
    size_t cache_interval;
    size_t disable_cache_before_step;
    int disable_cache_after_step;
};

class TSFullConstructorTest : public TaylorSeerCacheConfigTest,
                           public ::testing::WithParamInterface<FullConstructorParams> {};

TEST_P(TSFullConstructorTest, AllParameters) {
    auto params = GetParam();
    TaylorSeerCacheConfig config{params.cache_interval,
                                  params.disable_cache_before_step,
                                  params.disable_cache_after_step};
    AssertConfigEquals(config, params.cache_interval,
                      params.disable_cache_before_step,
                      params.disable_cache_after_step);
}

INSTANTIATE_TEST_SUITE_P(
    CustomValues,
    TSFullConstructorTest,
    ::testing::Values(
        FullConstructorParams{3, 6, -2},
        FullConstructorParams{3, 1, 24}
    )
);


struct PartialConstructorParams {
    size_t cache_interval;
    std::optional<size_t> disable_cache_before_step;
    size_t expected_before;
};

class TSPartialConstructorTest : public TaylorSeerCacheConfigTest,
                                 public ::testing::WithParamInterface<PartialConstructorParams> {};

TEST_P(TSPartialConstructorTest, PartialParameters) {
    auto params = GetParam();

    if (params.disable_cache_before_step.has_value()) {
        TaylorSeerCacheConfig config{params.cache_interval, *params.disable_cache_before_step};
        AssertConfigEquals(config, params.cache_interval, params.expected_before, -2);
    } else {
        TaylorSeerCacheConfig config{params.cache_interval};
        AssertConfigEquals(config, params.cache_interval, params.expected_before, -2);
    }
}

INSTANTIATE_TEST_SUITE_P(
    PartialValues,
    TSPartialConstructorTest,
    ::testing::Values(
        // Only cache_interval (should use default disable_cache_before_step = 6)
        PartialConstructorParams{7, std::nullopt, 6},
        // Both parameters
        PartialConstructorParams{3, 0, 0}
    )
);

// Parameterized test for to_string with various values
struct ToStringParams {
    size_t cache_interval;
    size_t disable_cache_before_step;
    int disable_cache_after_step;
};

class TSToStringTest : public TaylorSeerCacheConfigTest,
                       public ::testing::WithParamInterface<ToStringParams> {};

TEST_P(TSToStringTest, FormatsCorrectly) {
    auto params = GetParam();
    TaylorSeerCacheConfig config{params.cache_interval,
                                  params.disable_cache_before_step,
                                  params.disable_cache_after_step};
    std::string result = config.to_string();

    EXPECT_NE(result.find("TaylorSeerCacheConfig"), std::string::npos);
    EXPECT_NE(result.find("cache_interval: " + std::to_string(params.cache_interval)), std::string::npos);
    EXPECT_NE(result.find("disable_cache_before_step: " + std::to_string(params.disable_cache_before_step)), std::string::npos);
    EXPECT_NE(result.find("disable_cache_after_step: " + std::to_string(params.disable_cache_after_step)), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(
    VariousConfigurations,
    TSToStringTest,
    ::testing::Values(
        ToStringParams{3, 4, -2},
        ToStringParams{5, 2, 26}
    )
);



class TaylorSeerStateTest : public ::testing::Test {
protected:
    ov::Tensor CreateTestTensor(const std::vector<float>& data) {
        ov::Tensor tensor(ov::element::f32, {data.size()});
        std::memcpy(tensor.data<float>(), data.data(), data.size() * sizeof(float));
        return tensor;
    }

    // Helper to compare tensors
    void AssertTensorsEqual(const ov::Tensor& actual, const ov::Tensor& expected) {
        ASSERT_EQ(actual.get_shape(), expected.get_shape());
        ASSERT_EQ(actual.get_element_type(), expected.get_element_type());

        auto actual_data = actual.data<float>();
        auto expected_data = expected.data<float>();
        size_t size = actual.get_size();

        for (size_t i = 0; i < size; ++i) {
            EXPECT_FLOAT_EQ(actual_data[i], expected_data[i]) << "Mismatch at index " << i;
        }
    }
};

TEST_F(TaylorSeerStateTest, SecondUpdateComputesDerivative) {
    TaylorSeerCacheConfig config{3, 2, -2};
    TaylorSeerState state(config, 10);
    auto tensor1 = CreateTestTensor({1.0f, 4.0f});
    auto tensor2 = CreateTestTensor({3.0f, 6.0f});

    state.update(0, tensor1);
    state.update(1, tensor2);

    EXPECT_EQ(state.get_last_update_step().value(), 1);

    AssertTensorsEqual(state.get_taylor_factor(0), tensor2);
    // First derivative: (tensor2 - tensor1) / step_offset = (3-1, 6-4) / 1 = (2, 2)
    auto expected_derivative = CreateTestTensor({2.0f, 2.0f});
    AssertTensorsEqual(state.get_taylor_factor(1), expected_derivative);
}


struct ShouldComputeParams {
    size_t current_step;
    size_t cache_interval;
    size_t disable_before;
    int disable_after;
    size_t num_inference_steps;
    bool expected_result;
    std::string description;
};

class TSShouldComputeTest : public TaylorSeerStateTest,
                            public ::testing::WithParamInterface<ShouldComputeParams> {};

TEST_P(TSShouldComputeTest, ComputeDecisions) {
    auto params = GetParam();
    TaylorSeerCacheConfig config{params.cache_interval,
                                  params.disable_before,
                                  params.disable_after};
    TaylorSeerState state(config, params.num_inference_steps);
    bool result = state.should_compute(params.current_step);
    EXPECT_EQ(result, params.expected_result) << params.description;
}

INSTANTIATE_TEST_SUITE_P(
    VariousScenarios,
    TSShouldComputeTest,
    ::testing::Values(
        // Warm-up phase
        ShouldComputeParams{0, 3, 6, -2, 50, true, "Step 0 in warm-up"},
        ShouldComputeParams{5, 3, 6, -2, 50, true, "Step 5 in warm-up"},

        // Cache interval logic
        ShouldComputeParams{6, 3, 6, -2, 50, false, "Step 6 - predict"},
        ShouldComputeParams{8, 3, 6, -2, 50, true, "Step 8 - compute"},
        ShouldComputeParams{12, 5, 6, -2, 50, false, "Step 12 - predict"},

        // Negative disable_after_step
        ShouldComputeParams{46, 3, 6, -2, 50, false, "Step 46 < 50 - 2"},
        ShouldComputeParams{48, 3, 6, -2, 50, true, "Step 48 >= 50 - 2"},
        ShouldComputeParams{49, 3, 6, -2, 50, true, "Step 49 >= 50 - 2"},

        // Positive disable_after_step
        ShouldComputeParams{25, 3, 6, 48, 50, false, "Step 25 < 48"},
        ShouldComputeParams{48, 3, 6, 48, 50, true, "Step 48 >= 48"},
        ShouldComputeParams{49, 3, 6, 48, 50, true, "Step 49 >= 48"}
    )
);

TEST_F(TaylorSeerStateTest, DisableCacheBeforeStepGreaterThanNumSteps) {
    TaylorSeerCacheConfig config{3, 100, -2};
    TaylorSeerState state(config, 50);
    EXPECT_THROW(state.should_compute(0), ov::Exception);
}

TEST_F(TaylorSeerStateTest, InactiveWhenDisableBeforeEqualsSteps) {
    TaylorSeerCacheConfig config{3, 50, -2};
    TaylorSeerState state(config, 50);
    // TaylorSeer is inactive when disable_before >= num_steps
    EXPECT_FALSE(state.is_active());
    EXPECT_THROW(state.should_compute(0), ov::Exception);
}

TEST_F(TaylorSeerStateTest, ActiveWhenDisableBeforeLessThanNumSteps) {
    TaylorSeerCacheConfig config{3, 50, -2};
    TaylorSeerState state(config, 60);
    // TaylorSeer is active when disable_before < num_steps
    EXPECT_TRUE(state.is_active());
    EXPECT_NO_THROW(state.should_compute(0));
}

TEST_F(TaylorSeerStateTest, InactiveWhenNegativeDisableAfterExceedsSteps) {
    TaylorSeerCacheConfig config{3, 6, -100};
    TaylorSeerState state(config, 50);
    // -100 + 50 = -50 (still negative), so inactive
    EXPECT_FALSE(state.is_active());
    EXPECT_THROW(state.should_compute(0), ov::Exception);
}

TEST_F(TaylorSeerStateTest, InactiveWhenNegativeDisableAfterEqualsSteps) {
    // -50 + 50 = 0, which is <= disable_before (6), so no caching window
    TaylorSeerCacheConfig config{3, 6, -50};
    TaylorSeerState state(config, 50);
    EXPECT_FALSE(state.is_active());
    EXPECT_THROW(state.should_compute(0), ov::Exception);
}

TEST_F(TaylorSeerStateTest, InactiveWhenDisableBeforeExceedsSteps) {
    TaylorSeerCacheConfig config{3, 50, -2};
    TaylorSeerState state(config, 50);

    // TaylorSeer is inactive, schedule is empty
    EXPECT_FALSE(state.is_active());
    EXPECT_THROW(state.should_compute(0), ov::Exception);
}

TEST_F(TaylorSeerStateTest, ShouldComputeWithZeroDisableBefore) {
    TaylorSeerCacheConfig config{3, 0, -2};
    TaylorSeerState state(config, 10);
    EXPECT_TRUE(state.is_active());
    // Warmup is max(0, 2) = 2, so steps 0-1 compute
    EXPECT_TRUE(state.should_compute(0));
    EXPECT_TRUE(state.should_compute(1));
    // After warmup, caching starts
    EXPECT_FALSE(state.should_compute(2));
    EXPECT_FALSE(state.should_compute(3));
    EXPECT_TRUE(state.should_compute(4));
}

TEST_F(TaylorSeerStateTest, InactiveWhenDisableAfterBeforeDisableBefore) {
    // disable_after=0 <= disable_before=2, so no caching window
    TaylorSeerCacheConfig config{3, 2, 0};
    TaylorSeerState state(config, 50);

    // TaylorSeer is inactive
    EXPECT_FALSE(state.is_active());
    EXPECT_THROW(state.should_compute(0), ov::Exception);
}

TEST_F(TaylorSeerStateTest, InactiveWhenDisableAfterEqualsDisableBefore) {
    TaylorSeerCacheConfig config{3, 10, 10};
    TaylorSeerState state(config, 50);

    // disable_after=10 <= disable_before=10, no caching window
    EXPECT_FALSE(state.is_active());
    EXPECT_THROW(state.should_compute(0), ov::Exception);
}

TEST_F(TaylorSeerStateTest, PredictWithTaylorSeries) {
    TaylorSeerCacheConfig config{3, 2, -2};
    TaylorSeerState state(config, 10);

    auto tensor1 = CreateTestTensor({1.0f});
    auto tensor2 = CreateTestTensor({5.0f});

    // Set up Taylor factors: f(0) = 1
    state.update(0, tensor1);
    // Set up Taylor factors: f(0) = 5, f(1) = (5-1)/1 = 4
    state.update(1, tensor2);

    // Predict at step 3 (offset=2 from step 1):
    // f(3) = 5.0 * 1 + 4.0 * 2 = 5 + 8 = 13.0
    auto predicted = state.predict(3);
    auto expected = CreateTestTensor({13.0f});
    AssertTensorsEqual(predicted, expected);
}

TEST_F(TaylorSeerStateTest, PredictRequiresSecondUpdate) {
    TaylorSeerCacheConfig config{3, 2, -2};
    TaylorSeerState state(config, 10);
    auto tensor = CreateTestTensor({1.0f});

    state.update(0, tensor);

    // Order 0 exists but order 1 is not yet computed (needs second update)
    EXPECT_NO_THROW(state.get_taylor_factor(0));
    // Can't predict before second update
    EXPECT_THROW(state.predict(5), ov::Exception);
}

TEST_F(TaylorSeerStateTest, FullWorkflow) {
    TaylorSeerCacheConfig config{3, 2, -2};
    TaylorSeerState state(config, 10);

    // Steps 0-1: warm-up (should compute)
    EXPECT_TRUE(state.should_compute(0));
    EXPECT_TRUE(state.should_compute(1));

    // Simulate updates during warm-up
    auto tensor1 = CreateTestTensor({1.0f});
    auto tensor2 = CreateTestTensor({3.0f});
    state.update(0, tensor1);
    state.update(1, tensor2);

    // After warm-up use cache
    EXPECT_FALSE(state.should_compute(2));
    EXPECT_FALSE(state.should_compute(5));
    // Can predict during cache steps
    EXPECT_NO_THROW(state.predict(4));

    // Step 7: compute
    EXPECT_TRUE(state.should_compute(7));

    // Steps 8-9: compute after disable_after_step
    EXPECT_TRUE(state.should_compute(8));
    EXPECT_TRUE(state.should_compute(9));
}

TEST_F(TaylorSeerStateTest, ReinitializeResetsState) {
    // Create an active TaylorSeer state
    TaylorSeerCacheConfig config{3, 2, -2};
    TaylorSeerState state(config, 10);

    EXPECT_TRUE(state.is_active());

    // Populate state with data
    auto tensor1 = CreateTestTensor({1.0f, 2.0f});
    auto tensor2 = CreateTestTensor({3.0f, 4.0f});
    state.update(0, tensor1);
    state.update(1, tensor2);

    EXPECT_TRUE(state.get_last_update_step().has_value());
    EXPECT_EQ(state.get_last_update_step().value(), 1);
    EXPECT_NO_THROW(state.get_taylor_factor(0));
    EXPECT_NO_THROW(state.get_taylor_factor(1));

    // Re-initialize with invalid config (should clear all state)
    state.initialize(std::nullopt, 10);

    EXPECT_FALSE(state.is_active());
    EXPECT_FALSE(state.get_last_update_step().has_value());
    EXPECT_THROW(state.should_compute(0), ov::Exception);  // Schedule should be empty

    // Re-initialize with valid config (should reset and work properly)
    state.initialize(config, 10);

    EXPECT_TRUE(state.is_active());
    EXPECT_FALSE(state.get_last_update_step().has_value());  // Should be reset
    EXPECT_NO_THROW(state.should_compute(0));

    // Should be able to use normally after reinitialization
    auto tensor3 = CreateTestTensor({5.0f, 6.0f});
    auto tensor4 = CreateTestTensor({7.0f, 8.0f});
    EXPECT_NO_THROW(state.update(0, tensor3));
    EXPECT_NO_THROW(state.update(1, tensor4));
    EXPECT_NO_THROW(state.predict(2));
}

TEST_F(TaylorSeerStateTest, ReinitializeWithInactiveConfig) {
    // Start with active config
    TaylorSeerCacheConfig active_config{3, 2, -2};
    TaylorSeerState state(active_config, 10);

    EXPECT_TRUE(state.is_active());

    auto tensor = CreateTestTensor({1.0f});
    state.update(0, tensor);
    EXPECT_TRUE(state.get_last_update_step().has_value());

    // Re-initialize with config that makes TaylorSeer inactive
    TaylorSeerCacheConfig inactive_config{3, 100, -2};
    state.initialize(inactive_config, 10);

    EXPECT_FALSE(state.is_active());
    EXPECT_FALSE(state.get_last_update_step().has_value());
    EXPECT_THROW(state.should_compute(0), ov::Exception);

    // Re-initialize back to active config
    state.initialize(active_config, 10);

    EXPECT_TRUE(state.is_active());
    EXPECT_FALSE(state.get_last_update_step().has_value());
    EXPECT_NO_THROW(state.should_compute(0));
}

TEST_F(TaylorSeerStateTest, ReinitializeDoesNotLeaveStaleSchedule) {
    // Start with config having specific schedule
    TaylorSeerCacheConfig config1{3, 2, -2};
    TaylorSeerState state(config1, 10);

    EXPECT_TRUE(state.is_active());
    EXPECT_TRUE(state.should_compute(0));
    EXPECT_FALSE(state.should_compute(2));

    // Re-initialize with different config
    TaylorSeerCacheConfig config2{5, 2, -2};
    state.initialize(config2, 10);

    EXPECT_TRUE(state.is_active());
    EXPECT_TRUE(state.should_compute(0));
    EXPECT_FALSE(state.should_compute(2));
    EXPECT_FALSE(state.should_compute(5));
    EXPECT_TRUE(state.should_compute(6));
}

TEST_F(TaylorSeerStateTest, NoUpdateInCoolDownPhase) {
    TaylorSeerCacheConfig config{3, 2, -2};
    TaylorSeerState state(config, 10);

    EXPECT_TRUE(state.is_active());

    // Warm-up phase: steps 0-1
    auto tensor1 = CreateTestTensor({1.0f, 2.0f, 3.0f});
    auto tensor2 = CreateTestTensor({2.0f, 4.0f, 6.0f});
    state.update(0, tensor1);
    state.update(1, tensor2);
    EXPECT_EQ(state.get_last_update_step().value(), 1);

    auto tensor3 = CreateTestTensor({3.0f, 6.0f, 9.0f});
    state.update(4, tensor3);
    EXPECT_EQ(state.get_last_update_step().value(), 4);

    // Find last prediction step
    size_t last_prediction_step = 0;
    for (size_t step = 0; step < 10; ++step) {
        if (!state.should_compute(step)) {
            last_prediction_step = step;
        }
    }
    EXPECT_EQ(last_prediction_step, 6);

    // After last prediction step, all updates should be skipped
    // Steps 7-9 are compute steps but update should be no-op
    EXPECT_TRUE(state.should_compute(7));
    EXPECT_TRUE(state.should_compute(8));
    EXPECT_TRUE(state.should_compute(9));

    // Attempt updates for steps 7-9 - all should be skipped
    auto tensor4 = CreateTestTensor({4.0f, 8.0f, 12.0f});
    auto tensor5 = CreateTestTensor({6.0f, 12.0f, 18.0f});

    state.update(7, tensor4);
    // Last update step should remain 4 (update at step 7 was skipped)
    EXPECT_EQ(state.get_last_update_step().value(), 4);

    state.update(9, tensor5);
    // Last update step should still be 4
    EXPECT_EQ(state.get_last_update_step().value(), 4);

    // Taylor factors should still reflect the last real update at step 4
    auto expected_factor_0 = CreateTestTensor({3.0f, 6.0f, 9.0f});
    AssertTensorsEqual(state.get_taylor_factor(0), expected_factor_0);
}
