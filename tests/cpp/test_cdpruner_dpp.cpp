// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <openvino/openvino.hpp>
#include <tuple>
#include <vector>

#include "visual_language/cdpruner/cdpruner.hpp"
#include "visual_language/cdpruner/cdpruner_config.hpp"
#include "visual_language/cdpruner/fast_dpp.hpp"

using namespace ov::genai::cdpruner;

// =============================================================================
// Test Parameters Structure
// =============================================================================
enum class Backend { CPU, OpenCL };
enum class BatchMode { SingleBatch, MultiBatch };
enum class MatrixMode { Normal, Split };
enum class SplitMode { NonSplit, Split };

struct DPPTestParams {
    Backend backend;
    BatchMode batch_mode;
    MatrixMode matrix_mode;
    size_t num_tokens_to_select;

    // Helper method to get description for test naming
    std::string toString() const {
        std::string result;
        result += (backend == Backend::CPU) ? "CPU" : "OpenCL";
        result += "_";
        result += (batch_mode == BatchMode::SingleBatch) ? "Single" : "Multi";
        result += "_";
        result += (matrix_mode == MatrixMode::Normal) ? "Normal" : "Split";
        result += "_";
        result += std::to_string(num_tokens_to_select) + "tokens";
        return result;
    }
};

struct CDPrunerIntegrationTestParams {
    Backend backend;
    SplitMode split_mode;

    // Helper method to get description for test naming
    std::string toString() const {
        std::string result;
        result += (backend == Backend::CPU) ? "CPU" : "OpenCL";
        result += "_";
        result += (split_mode == SplitMode::NonSplit) ? "NonSplit" : "Split";
        return result;
    }
};

// =============================================================================
// Base Test Fixture
// =============================================================================
class DPPTestBase : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize common config for testing
        base_config.pruning_ratio = 50;  // Will prune 50% tokens
        base_config.relevance_weight = 0.5f;
        base_config.pruning_debug_mode = false;
        base_config.use_negative_relevance = false;
        base_config.numerical_threshold = 1e-6f;
        base_config.device = "CPU";
    }

    // Helper function to create test kernel matrix
    ov::Tensor createTestKernel(const std::vector<float>& data, const ov::Shape& shape) {
        ov::Tensor kernel(ov::element::f32, shape);
        std::memcpy(kernel.data<float>(), data.data(), data.size() * sizeof(float));
        return kernel;
    }

    // Standard 4x4 test kernel with known properties
    ov::Tensor createStandardTestKernel() {
        std::vector<float> kernel_data = {
            // 4x4 kernel matrix with known diagonal dominance
            0.8f,
            0.3f,
            0.1f,
            0.2f,  // token 0 row
            0.3f,
            0.9f,
            0.4f,
            0.1f,  // token 1 row (highest diagonal)
            0.1f,
            0.4f,
            0.7f,
            0.5f,  // token 2 row
            0.2f,
            0.1f,
            0.5f,
            0.6f  // token 3 row
        };
        return createTestKernel(kernel_data, {1, 4, 4});
    }

    // Create multi-batch standard test kernel
    ov::Tensor createMultiBatchStandardTestKernel(size_t batch_size = 2) {
        std::vector<float> single_kernel =
            {0.8f, 0.3f, 0.1f, 0.2f, 0.3f, 0.9f, 0.4f, 0.1f, 0.1f, 0.4f, 0.7f, 0.5f, 0.2f, 0.1f, 0.5f, 0.6f};

        std::vector<float> multi_kernel;
        for (size_t b = 0; b < batch_size; ++b) {
            multi_kernel.insert(multi_kernel.end(), single_kernel.begin(), single_kernel.end());
        }

        return createTestKernel(multi_kernel, {batch_size, 4, 4});
    }

    Config base_config;
};

// =============================================================================
// Parameterized Test Fixture
// =============================================================================
class DPPParameterizedTest : public DPPTestBase, public ::testing::WithParamInterface<DPPTestParams> {
protected:
    void SetUp() override {
        DPPTestBase::SetUp();

        auto params = GetParam();

        // Configure backend
        test_config = base_config;
        test_config.use_cl_kernel = (params.backend == Backend::OpenCL);

        // Create DPP selector
        dpp_selector = std::make_unique<FastGreedyDPP>(test_config);
    }

    // Execute DPP selection based on parameters
    std::vector<std::vector<size_t>> executeSelection(const DPPTestParams& params) {
        if (params.matrix_mode == MatrixMode::Split) {
            return executeSplitSelection(params);
        } else {
            return executeNormalSelection(params);
        }
    }

private:
    std::vector<std::vector<size_t>> executeNormalSelection(const DPPTestParams& params) {
        ov::Tensor kernel;
        if (params.batch_mode == BatchMode::SingleBatch) {
            kernel = createStandardTestKernel();
        } else {
            kernel = createMultiBatchStandardTestKernel();
        }

        return dpp_selector->select(kernel, params.num_tokens_to_select);
    }

    std::vector<std::vector<size_t>> executeSplitSelection(const DPPTestParams& params) {
        std::vector<float> standard_kernel_data =
            {0.8f, 0.3f, 0.1f, 0.2f, 0.3f, 0.9f, 0.4f, 0.1f, 0.1f, 0.4f, 0.7f, 0.5f, 0.2f, 0.1f, 0.5f, 0.6f};

        ov::Tensor first_kernel, second_kernel;
        if (params.batch_mode == BatchMode::SingleBatch) {
            first_kernel = createTestKernel(standard_kernel_data, {1, 4, 4});
            second_kernel = createTestKernel(standard_kernel_data, {1, 4, 4});
        } else {
            // Create multi-batch data
            std::vector<float> multi_data;
            for (int b = 0; b < 2; ++b) {
                multi_data.insert(multi_data.end(), standard_kernel_data.begin(), standard_kernel_data.end());
            }
            first_kernel = createTestKernel(multi_data, {2, 4, 4});
            second_kernel = createTestKernel(multi_data, {2, 4, 4});
        }

        size_t split_point = 4;
        return dpp_selector->select(first_kernel, second_kernel, params.num_tokens_to_select, split_point);
    }

protected:
    Config test_config;
    std::unique_ptr<FastGreedyDPP> dpp_selector;
};

// =============================================================================
// Test Cases
// =============================================================================

TEST_P(DPPParameterizedTest, BasicTokenSelection) {
    auto params = GetParam();
    auto selected_tokens = executeSelection(params);

    // Basic validations
    size_t expected_batch_count = (params.batch_mode == BatchMode::SingleBatch) ? 1 : 2;
    ASSERT_EQ(selected_tokens.size(), expected_batch_count);

    for (size_t batch = 0; batch < selected_tokens.size(); ++batch) {
        // Verify correct number of tokens selected
        if (params.matrix_mode == MatrixMode::Split && params.num_tokens_to_select == 3) {
            // Special case: split mode with 3 tokens may behave differently for OpenCL
            if (params.backend == Backend::OpenCL) {
                // OpenCL split mode might return different count due to even number requirement
                EXPECT_GT(selected_tokens[batch].size(), 0) << "Batch " << batch << " should select some tokens";
            } else {
                EXPECT_EQ(selected_tokens[batch].size(), params.num_tokens_to_select)
                    << "Batch " << batch << " should select " << params.num_tokens_to_select << " tokens";
            }
        } else {
            EXPECT_EQ(selected_tokens[batch].size(), params.num_tokens_to_select)
                << "Batch " << batch << " should select " << params.num_tokens_to_select << " tokens";
        }

        // Verify token indices are valid
        for (size_t token : selected_tokens[batch]) {
            if (params.matrix_mode == MatrixMode::Split) {
                EXPECT_LT(token, 8) << "Split mode token index should be < 8 (4 + 4)";
            } else {
                EXPECT_LT(token, 4) << "Normal mode token index should be < 4";
            }
        }

        // Verify no duplicate tokens
        std::vector<size_t> sorted_tokens = selected_tokens[batch];
        std::sort(sorted_tokens.begin(), sorted_tokens.end());
        auto it = std::unique(sorted_tokens.begin(), sorted_tokens.end());
        EXPECT_EQ(it, sorted_tokens.end()) << "Batch " << batch << " should not have duplicate tokens";
    }
}

TEST_P(DPPParameterizedTest, DeterministicResults) {
    auto params = GetParam();

    // Run selection multiple times to verify deterministic behavior
    auto result1 = executeSelection(params);
    auto result2 = executeSelection(params);

    ASSERT_EQ(result1.size(), result2.size());

    for (size_t batch = 0; batch < result1.size(); ++batch) {
        // Sort both results for comparison
        std::vector<size_t> sorted1 = result1[batch];
        std::vector<size_t> sorted2 = result2[batch];
        std::sort(sorted1.begin(), sorted1.end());
        std::sort(sorted2.begin(), sorted2.end());

        EXPECT_EQ(sorted1, sorted2) << "Batch " << batch << " should have deterministic results";
    }
}

TEST_P(DPPParameterizedTest, KnownResultValidation) {
    auto params = GetParam();
    auto selected_tokens = executeSelection(params);

    // Validate known results for specific configurations
    if (params.num_tokens_to_select == 1 && params.matrix_mode == MatrixMode::Normal) {
        // Single token selection should pick token 1 (highest diagonal value)
        for (const auto& batch_tokens : selected_tokens) {
            EXPECT_EQ(batch_tokens[0], 1) << "Single token selection should pick token 1";
        }
    }

    if (params.num_tokens_to_select == 3 && params.matrix_mode == MatrixMode::Normal) {
        // 3 token selection should pick tokens [1, 0, 3] (known DPP result)
        for (const auto& batch_tokens : selected_tokens) {
            std::vector<size_t> sorted_tokens = batch_tokens;
            std::sort(sorted_tokens.begin(), sorted_tokens.end());
            std::vector<size_t> expected = {0, 1, 3};
            EXPECT_EQ(sorted_tokens, expected) << "3 token selection should pick [0, 1, 3]";
        }
    }
}

// =============================================================================
// Test Parameter Generation
// =============================================================================

// Generate all combinations of test parameters
std::vector<DPPTestParams> generateTestParams() {
    std::vector<DPPTestParams> params;

    std::vector<Backend> backends = {Backend::CPU, Backend::OpenCL};
    std::vector<BatchMode> batch_modes = {BatchMode::SingleBatch, BatchMode::MultiBatch};
    std::vector<MatrixMode> matrix_modes = {MatrixMode::Normal, MatrixMode::Split};
    std::vector<size_t> token_counts = {1, 3, 4};  // Different token selection counts

    for (auto backend : backends) {
        for (auto batch_mode : batch_modes) {
            for (auto matrix_mode : matrix_modes) {
                for (auto token_count : token_counts) {
                    params.push_back({backend, batch_mode, matrix_mode, token_count});
                }
            }
        }
    }

    return params;
}

// Test naming function
std::string paramToString(const ::testing::TestParamInfo<DPPTestParams>& info) {
    return info.param.toString();
}

// Instantiate parameterized tests
INSTANTIATE_TEST_SUITE_P(CDPrunerTest, DPPParameterizedTest, ::testing::ValuesIn(generateTestParams()), paramToString);

// =============================================================================
// Integration Tests with CDPruner
// =============================================================================
class CDPrunerIntegrationTest : public DPPTestBase,
                                public ::testing::WithParamInterface<CDPrunerIntegrationTestParams> {
protected:
    void SetUp() override {
        DPPTestBase::SetUp();
        cdp_config = base_config;
        cdp_config.pruning_ratio = 50;

        // Configure based on test parameters
        auto params = GetParam();
        cdp_config.use_cl_kernel = (params.backend == Backend::OpenCL);

        // Set split threshold based on split mode
        if (params.split_mode == SplitMode::Split) {
            cdp_config.split_threshold = 50;  // Force splitting for testing
        } else {
            cdp_config.split_threshold = 2000;  // Default, no splitting
        }
    }

    Config cdp_config;
};

TEST_P(CDPrunerIntegrationTest, LargeSequenceSplitting) {
    // Test CDPruner with sequences large enough to trigger splitting
    CDPruner cdpruner(cdp_config);

    size_t batch_size = 1;
    size_t sequence_length = 600;
    size_t hidden_dim = 1024;

    ov::Tensor visual_features(ov::element::f32, {batch_size, sequence_length, hidden_dim});
    ov::Tensor text_features(ov::element::f32, {batch_size, hidden_dim});

    // Initialize with simple test patterns
    float* visual_data = visual_features.data<float>();
    float* text_data = text_features.data<float>();

    // Create simple patterns
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < sequence_length; ++s) {
            for (size_t h = 0; h < hidden_dim; ++h) {
                size_t idx = b * sequence_length * hidden_dim + s * hidden_dim + h;
                visual_data[idx] = 0.5f + 0.2f * idx;
            }
        }

        for (size_t h = 0; h < hidden_dim; ++h) {
            text_data[b * hidden_dim + h] = 0.3f + 0.1f * h;
        }
    }

    // Test pruning application
    auto pruned_features = cdpruner.apply_pruning(visual_features, text_features);
    auto pruned_shape = pruned_features.get_shape();

    EXPECT_EQ(pruned_shape[0], batch_size);       // Batch size unchanged
    EXPECT_LT(pruned_shape[1], sequence_length);  // Sequence length reduced
    EXPECT_EQ(pruned_shape[2], hidden_dim);       // Hidden dim unchanged
}

// =============================================================================
// CDPruner Integration Test Parameter Generation
// =============================================================================
std::vector<CDPrunerIntegrationTestParams> generateCDPrunerTestParams() {
    std::vector<CDPrunerIntegrationTestParams> params;

    for (auto backend : {Backend::CPU, Backend::OpenCL}) {
        for (auto split_mode : {SplitMode::NonSplit, SplitMode::Split}) {
            params.push_back({backend, split_mode});
        }
    }

    return params;
}

std::string cdprunerParamToString(const ::testing::TestParamInfo<CDPrunerIntegrationTestParams>& info) {
    return info.param.toString();
}

// Instantiate the parameterized CDPruner integration test
INSTANTIATE_TEST_SUITE_P(CDPrunerTest,
                         CDPrunerIntegrationTest,
                         ::testing::ValuesIn(generateCDPrunerTestParams()),
                         cdprunerParamToString);