// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
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
        EXPECT_EQ(config.get_cache_interval(), expected_interval);
        EXPECT_EQ(config.get_disable_cache_before_step(), expected_before);
        EXPECT_EQ(config.get_disable_cache_after_step(), expected_after);
    }
};

TEST_F(TaylorSeerCacheConfigTest, DefaultConstructor) {
    TaylorSeerCacheConfig config;
    // Default constructor uses member initializers: cache_interval=3, disable_before=6, disable_after=-2
    AssertConfigEquals(config, 3, 6, -2);
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
    TaylorSeerCacheConfig config(params.cache_interval,
                                  params.disable_cache_before_step,
                                  params.disable_cache_after_step);
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
        TaylorSeerCacheConfig config(params.cache_interval, *params.disable_cache_before_step);
        AssertConfigEquals(config, params.cache_interval, params.expected_before, -2);
    } else {
        TaylorSeerCacheConfig config(params.cache_interval);
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
    TaylorSeerCacheConfig config(params.cache_interval,
                                  params.disable_cache_before_step,
                                  params.disable_cache_after_step);
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
    TaylorSeerState state;
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
    TaylorSeerState state;
    TaylorSeerCacheConfig config(params.cache_interval,
                                  params.disable_before,
                                  params.disable_after);

    bool result = state.should_compute(params.current_step, config, params.num_inference_steps);
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

TEST_F(TaylorSeerStateTest, PredictWithTaylorSeries) {
    TaylorSeerState state;

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

TEST_F(TaylorSeerStateTest, GetTaylorFactorThrowsForInvalidOrder) {
    TaylorSeerState state;
    auto tensor = CreateTestTensor({1.0f});

    state.update(0, tensor);

    // Only order 0 exists
    EXPECT_NO_THROW(state.get_taylor_factor(0));
    EXPECT_THROW(state.get_taylor_factor(1), std::out_of_range);
    EXPECT_THROW(state.predict(5), ov::Exception);
}

TEST_F(TaylorSeerStateTest, FullWorkflow) {
    TaylorSeerState state;
    TaylorSeerCacheConfig config(3, 2, -2);
    size_t num_steps = 10;

    // Steps 0-1: warm-up (should compute)
    EXPECT_TRUE(state.should_compute(0, config, num_steps));
    EXPECT_TRUE(state.should_compute(1, config, num_steps));

    // Simulate updates during warm-up
    auto tensor1 = CreateTestTensor({1.0f});
    auto tensor2 = CreateTestTensor({3.0f});
    state.update(0, tensor1);
    state.update(1, tensor2);

    // After warm-up use cache
    EXPECT_FALSE(state.should_compute(2, config, num_steps));
    EXPECT_FALSE(state.should_compute(5, config, num_steps));

    // Can predict during cache steps
    EXPECT_NO_THROW(state.predict(4));

    // Step 7: compute
    EXPECT_TRUE(state.should_compute(7, config, num_steps));

    // Steps 8-9: compute after disable_after_step
    EXPECT_TRUE(state.should_compute(8, config, num_steps));
    EXPECT_TRUE(state.should_compute(9, config, num_steps));
}
