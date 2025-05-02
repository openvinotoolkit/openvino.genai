// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache_eviction.hpp"

#include <algorithm>
#include <fstream>
#include <filesystem>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

const ov::genai::CacheEvictionConfig DEFAULT_CACHE_EVICTION_CONFIG = {32, 32, 192, ov::genai::AggregationMode::NORM_SUM};
const ov::genai::CacheEvictionConfig SHORT_RECENT_EVICTION_CONFIG = {32, 32, 72, ov::genai::AggregationMode::NORM_SUM};
constexpr size_t DEFAULT_BLOCK_SIZE = 4;
constexpr size_t DEFAULT_NUM_DECODER_LAYERS = 2;

class DefaultCacheEvictionAlgoTest : public testing::Test {
protected:
    DefaultCacheEvictionAlgoTest() {
        algo = ov::genai::CacheEvictionAlgorithm(eviction_config, block_size, num_decoder_layers);
    }
    size_t block_size = DEFAULT_BLOCK_SIZE;
    size_t num_decoder_layers = DEFAULT_NUM_DECODER_LAYERS;
    ov::genai::CacheEvictionConfig eviction_config = DEFAULT_CACHE_EVICTION_CONFIG;
    ov::genai::CacheEvictionAlgorithm algo;

    void evict_twice_and_expect_no_eviction() {
        auto blocks_to_evict = algo.evict_logical_blocks();
        for (const auto& evicted_blocks_for_this_layer : blocks_to_evict) {
            EXPECT_TRUE(evicted_blocks_for_this_layer.empty());
        }

        // again
        blocks_to_evict = algo.evict_logical_blocks();
        for (const auto& evicted_blocks_for_this_layer : blocks_to_evict) {
            EXPECT_TRUE(evicted_blocks_for_this_layer.empty());
        }
    }
};

AttentionScoresForEachDecoderLayer get_mock_scores(size_t num_layers, size_t num_tokens) {
    AttentionScoresForEachDecoderLayer retval;
    retval.reserve(num_layers);
    for (size_t i = 0; i < num_layers; i++) {
        auto tensor = ov::Tensor(ov::element::f32, ov::Shape{num_tokens});
        retval.push_back(tensor);
    }
    return retval;
}

TEST_F(DefaultCacheEvictionAlgoTest, NothingToEvictInitially) {
    evict_twice_and_expect_no_eviction();
}

class CacheEvictionAlgoTokenCountParameterizedTest : public DefaultCacheEvictionAlgoTest, public ::testing::WithParamInterface<size_t> {};


TEST_P(CacheEvictionAlgoTokenCountParameterizedTest, DoesntEvictIfTotalSizeNotReached) {
    const size_t num_tokens_to_register = GetParam();
    ASSERT_LT(num_tokens_to_register, eviction_config.get_max_cache_size());

    algo.register_new_token_scores(get_mock_scores(num_decoder_layers, num_tokens_to_register));

    evict_twice_and_expect_no_eviction();
}

INSTANTIATE_TEST_SUITE_P(VariousTokenCountsLessThanTotalSize, CacheEvictionAlgoTokenCountParameterizedTest,
                         ::testing::Values(8, 49, 190));


struct RangeCalculationTestStruct {
    size_t num_tokens;
    ov::genai::CacheEvictionAlgorithm::CacheEvictionRange expected_range;
};

class CacheEvictionRangeCalculationParameterizedTest : public DefaultCacheEvictionAlgoTest, public ::testing::WithParamInterface<RangeCalculationTestStruct> {};
const std::vector<RangeCalculationTestStruct> RANGE_CALCULATION_TEST_CASES = {
        {192, ov::genai::CacheEvictionAlgorithm::CacheEvictionRange(0, 0)},
        {192 + 1, ov::genai::CacheEvictionAlgorithm::CacheEvictionRange(0, 0)},
        {192 + 4, ov::genai::CacheEvictionAlgorithm::CacheEvictionRange(8, 41)},
        {192 + 4 + 1, ov::genai::CacheEvictionAlgorithm::CacheEvictionRange(8, 41)},
        {192 + 2 * 4, ov::genai::CacheEvictionAlgorithm::CacheEvictionRange(8, 42)},
        {192 + 8 * 4 + 3, ov::genai::CacheEvictionAlgorithm::CacheEvictionRange(8, 48)},
};
TEST_P(CacheEvictionRangeCalculationParameterizedTest, EvictableRangeCalculatedCorrectly) {
    const size_t num_tokens_to_register = GetParam().num_tokens;

    algo.register_new_token_scores(get_mock_scores(num_decoder_layers, num_tokens_to_register));
    auto range = algo.get_evictable_block_range();
    EXPECT_EQ(range.first, GetParam().expected_range.first);
    EXPECT_EQ(range.second, GetParam().expected_range.second);
}
INSTANTIATE_TEST_SUITE_P(VariousTokenCounts, CacheEvictionRangeCalculationParameterizedTest, ::testing::ValuesIn(RANGE_CALCULATION_TEST_CASES));


TEST_F(DefaultCacheEvictionAlgoTest, StartsEvictingOnceMaxSizeExceeded) {
    // all eviction areas filled, but no overflow yet
    algo.register_new_token_scores(get_mock_scores(num_decoder_layers, eviction_config.get_max_cache_size()));
    evict_twice_and_expect_no_eviction();

    // some tokens overflow the combined eviction area size, but the overflow size is less than 1 block
    algo.register_new_token_scores(get_mock_scores(num_decoder_layers, eviction_config.get_max_cache_size() + 1));
    evict_twice_and_expect_no_eviction();

    // same
    algo.register_new_token_scores(get_mock_scores(num_decoder_layers, eviction_config.get_max_cache_size() + DEFAULT_BLOCK_SIZE - 1));
    evict_twice_and_expect_no_eviction();

    // overflowing tokens now fill 1 extra block, all layers should evict 1 block
    algo.register_new_token_scores(get_mock_scores(num_decoder_layers, eviction_config.get_max_cache_size() + DEFAULT_BLOCK_SIZE));
    auto evictable_range = algo.get_evictable_block_range();
    EXPECT_EQ(evictable_range.second - evictable_range.first, eviction_config.get_evictable_size() / block_size + 1);

    auto evicted_blocks = algo.evict_logical_blocks();
    EXPECT_TRUE(std::all_of(evicted_blocks.begin(), evicted_blocks.end(), [](const std::set<size_t>& v) { return (v.size() == 1); }));
    EXPECT_TRUE(std::all_of(evicted_blocks.begin(), evicted_blocks.end(), [evictable_range](const std::set<size_t>& v) {
        size_t evicted_block_idx = *(v.begin());
        return (evicted_block_idx >= evictable_range.first) && (evicted_block_idx < evictable_range.second) ; }));
}

using CacheEvictionAlgoConfigurationTest = ::testing::TestWithParam<size_t>;

TEST_P(CacheEvictionAlgoConfigurationTest, EvictedBlocksAreLayeredAsConfigured) {
    size_t ref_num_layers = GetParam();
    auto algo = ov::genai::CacheEvictionAlgorithm(DEFAULT_CACHE_EVICTION_CONFIG, DEFAULT_BLOCK_SIZE, ref_num_layers);
    auto blocks_to_evict = algo.evict_logical_blocks();
    ASSERT_EQ(blocks_to_evict.size(), ref_num_layers);
}

INSTANTIATE_TEST_SUITE_P(VariousLayerCounts, CacheEvictionAlgoConfigurationTest, ::testing::Values(1, 4, 13, 23, 42));


void fill_scores(ov::Tensor& scores, size_t start_pos, size_t end_pos, float value) {
    ASSERT_LE(start_pos, end_pos);
    ASSERT_LE(end_pos, scores.get_size());

    for (size_t i = start_pos; i < end_pos; i++) {
        scores.data<float>()[i] = value;
    }
}

struct LowScoreBlocksTestStruct {
    std::string test_id;
    size_t tokens_over_max_cache_size;
    ov::genai::CacheEvictionConfig eviction_config;
    std::vector<std::set<size_t>> zero_filled_blocks;
    std::vector<std::set<size_t>> ref_evicted_blocks;
};

using CacheEvictionLowScoreBlocksParameterizedTest = ::testing::TestWithParam<LowScoreBlocksTestStruct>;

// clang-format off
const std::vector<LowScoreBlocksTestStruct> LOW_SCORE_BLOCK_EVICTION_TEST_CASES = {
        // low-scored blocks in evictable area
        {
                "one_block",
                1,  // one overflowing token amounting to one extra block to be evicted
                DEFAULT_CACHE_EVICTION_CONFIG,
                {{17}, {9}},
                {{17}, {9}}
        },

        // same, but with multiple blocks in evictable area
        {
                "three_blocks",
                2 * 4 + 2,  // 2 blocks worth of overflow + 2 tokens, amounting to 3 blocks to be evicted
                DEFAULT_CACHE_EVICTION_CONFIG,
                {{28, 10, 11}, {18, 8, 31}},
                {{28, 10, 11}, {18, 8, 31}}
        },
        // if there are more blocks with same low score than should be evicted, the lower-indexed ones should take precedence
        {
                "four_zeroed_two_to_evict",
                1 * 4 + 2,  // 2 blocks to be evicted
                DEFAULT_CACHE_EVICTION_CONFIG,
                {{15, 36, 13, 10}, {9, 39, 31, 11}},  // 4 zeroed blocks
                {{10, 13}, {9, 11}}
        },
        // will prefer to evict lower-indexed blocks if there are multiple same-scored blocks
        {
                "less_zeroed_than_to_evict",
                5 * 4 + 2,  // 6 blocks to be evicted
                DEFAULT_CACHE_EVICTION_CONFIG,
                {{}, {30, 22}},  // 1st layer has no zeroed blocks, 2nd has only 2 zeroed blocks
                {{8, 9, 10, 11, 12, 13}, {8, 9, 10, 11, 22, 30}}  // non-zeroed blocks to evict are taken from the beginning of evictable range
        },

        // low-scored blocks in non-evictable range do not lead to eviction
        {
                "zeros_also_in_non_evictable_areas",
                5 * 4 + 2,  // 6 blocks to be evicted
                DEFAULT_CACHE_EVICTION_CONFIG,
                {{0, 2, 7, 24, 31, 49}, {5, 19, 27, 39, 50, 52}},  // 1st layer has 0, 2, 7 in start_area, 49 in recent_area; 2nd has 5 in start_area, 50, 54 in recent_area
                {{8, 9, 10, 11, 24, 31}, {8, 9, 10, 19, 27, 39}}   // eviction padded up to 6 blocks by blocks in the beginning of the evictable_area
        },
        // more overflowing blocks than evictable area, recent area shifts accordingly to the end of the overflow
        {
                "more_overflow_than_eviction_blocks",
                4 * 4 + 1,  // 5 blocks to be evicted
                SHORT_RECENT_EVICTION_CONFIG,
                {{0, 9, 10, 11, 13}, {12, 11, 8, 9, 17}},
                {{8, 9, 10, 11, 13}, {8, 9, 10, 11, 12}}
        },
};
// clang-format on

TEST_P(CacheEvictionLowScoreBlocksParameterizedTest, EvictsLowestScoredBlocks) {
    auto test_struct = GetParam();
    size_t num_decoder_layers = DEFAULT_NUM_DECODER_LAYERS;
    auto algo = ov::genai::CacheEvictionAlgorithm(test_struct.eviction_config, DEFAULT_BLOCK_SIZE, num_decoder_layers);
    std::vector<std::set<size_t>> ref_lowest_scored_block_indices = test_struct.zero_filled_blocks;
    ASSERT_EQ(ref_lowest_scored_block_indices.size(), num_decoder_layers);

    auto scores = get_mock_scores(num_decoder_layers, algo.get_max_cache_size_after_eviction() + test_struct.tokens_over_max_cache_size);
    for (size_t layer_idx = 0; layer_idx < num_decoder_layers; layer_idx++) {
        auto& scores_per_layer = scores[layer_idx];
        // Fill scores of target blocks with 0, the rest with 1
        fill_scores(scores_per_layer, 0, scores_per_layer.get_size(), 1.0);
        for (size_t target_block_idx : test_struct.zero_filled_blocks[layer_idx]) {
            fill_scores(scores_per_layer, DEFAULT_BLOCK_SIZE * target_block_idx,
                        DEFAULT_BLOCK_SIZE * (target_block_idx + 1), 0.0);
        }
    }
    algo.register_new_token_scores(scores);

    auto test_evicted_blocks = algo.evict_logical_blocks();
    auto ref_evicted_blocks = test_struct.ref_evicted_blocks;
    for (size_t layer_idx = 0; layer_idx < num_decoder_layers; layer_idx++) {
        EXPECT_EQ(test_evicted_blocks[layer_idx], ref_evicted_blocks[layer_idx]);
    }
}


INSTANTIATE_TEST_SUITE_P(VariousSetsOfLowScoreBlocks, CacheEvictionLowScoreBlocksParameterizedTest,
                         ::testing::ValuesIn(LOW_SCORE_BLOCK_EVICTION_TEST_CASES),
                         [](const testing::TestParamInfo<CacheEvictionLowScoreBlocksParameterizedTest::ParamType>& info) {
                             return info.param.test_id;
                         });


static constexpr size_t BLOCKS_TO_EVICT = 3;  // 3 blocks to evict
struct NormalizationSettingTestStruct {
    ov::genai::AggregationMode normalization_mode;
    double token_score_power;
    bool newer_tokens_with_larger_score;
    std::array<size_t, BLOCKS_TO_EVICT> ref_evicted_blocks; // will be cast to std::set so order is irrelevant
};

using CacheEvictionNormalizationSettingTest = ::testing::TestWithParam<NormalizationSettingTestStruct>;

// clang-format off
const std::vector<NormalizationSettingTestStruct> NORMALIZATION_SETTING_TEST_CASES = {
    // power of 1.1 beats the 1 / N in the normalization, low-score blocks are in the end of the evictable area
    { ov::genai::AggregationMode::NORM_SUM, 1.1, false, { 40, 41, 42} },

    // newer tokens have larger score, low-score blocks are now in the beginning of the evictable area
    { ov::genai::AggregationMode::NORM_SUM, 1.1, true, { 8, 9, 10} },

    // power of 0.9 does not beat the 1 / N in the normalization, low-score blocks are in the beginning of the evictable area
    { ov::genai::AggregationMode::NORM_SUM, 0.9, false, { 8, 9, 10} },

    // newer tokens have larger score, low-score blocks are now in the beginning of the evictable area
    { ov::genai::AggregationMode::NORM_SUM, 0.9, true, { 8, 9, 10} },

    // for the SUM aggregation mode, only the score curve determines the evicted blocks
    { ov::genai::AggregationMode::SUM, 0.9, false, { 40, 41, 42} },
    { ov::genai::AggregationMode::SUM, 0.9, true, { 8, 9, 10} },
    { ov::genai::AggregationMode::SUM, 1.1, false, { 40, 41, 42} },
    { ov::genai::AggregationMode::SUM, 1.1, true, { 8, 9, 10} },
};
// clang-format on

TEST_P(CacheEvictionNormalizationSettingTest, TokenLifetimeNormalizationHasEffect) {
    const auto& test_struct = GetParam();
    auto config = DEFAULT_CACHE_EVICTION_CONFIG;
    config.aggregation_mode = test_struct.normalization_mode;

    const size_t NUM_DECODER_LAYERS = 1;
    auto algo = ov::genai::CacheEvictionAlgorithm(config, DEFAULT_BLOCK_SIZE, NUM_DECODER_LAYERS);
    auto scores = get_mock_scores(NUM_DECODER_LAYERS, algo.get_max_cache_size_after_eviction() + BLOCKS_TO_EVICT * DEFAULT_BLOCK_SIZE);
    for (auto& scores_per_layer : scores) {
        const size_t SCORES_SIZE = scores_per_layer.get_size();
        for (size_t i = 0; i < SCORES_SIZE; i++) {
            if (test_struct.newer_tokens_with_larger_score) {
               fill_scores(scores_per_layer, i, i + 1, std::pow(i, test_struct.token_score_power));
            } else {
               fill_scores(scores_per_layer, SCORES_SIZE - i - 1, SCORES_SIZE - i, std::pow(i, test_struct.token_score_power));
            }
        }
    }

    algo.register_new_token_scores(scores);
    auto blocks_to_evict = algo.evict_logical_blocks();
    std::set<size_t> ref_evicted_blocks;
    for (auto val : test_struct.ref_evicted_blocks) {
        ref_evicted_blocks.insert(val); // same for all decoder layers
    }

    for (const auto& test_evicted_blocks : blocks_to_evict) {
        EXPECT_EQ(test_evicted_blocks, ref_evicted_blocks);
    }

}

INSTANTIATE_TEST_SUITE_P(VariousAggregationModesAndScoreDistributions, CacheEvictionNormalizationSettingTest,
                         ::testing::ValuesIn(NORMALIZATION_SETTING_TEST_CASES),
                         [](const testing::TestParamInfo<CacheEvictionNormalizationSettingTest::ParamType>& info) {
                            std::stringstream ss;
                            if (info.param.normalization_mode == ov::genai::AggregationMode::NORM_SUM) {
                                ss << "norm_sum";
                            }
                            else {
                                ss << "sum";
                            }
                            ss << "_" << info.param.token_score_power;

                            if (info.param.newer_tokens_with_larger_score) {
                                ss << "_rising";
                            }
                            else {
                                ss << "_falling";
                            }

                            std::string retval = ss.str();
                            std::replace(retval.begin(), retval.end(), '.', '_');
                            return retval;
                         });



using CacheEvictionConfigModeCommonBehaviour = ::testing::TestWithParam<ov::genai::AggregationMode>;
const std::vector<ov::genai::AggregationMode> SCORE_ACCUMULATION_TEST_CASES = {ov::genai::AggregationMode::NORM_SUM,
                                                                               ov::genai::AggregationMode::SUM};

TEST_P(CacheEvictionConfigModeCommonBehaviour, ScoresAreAccumulated) {
    const auto& aggregation_mode = GetParam();

    auto config = DEFAULT_CACHE_EVICTION_CONFIG;
    config.aggregation_mode = aggregation_mode;
    const size_t NUM_DECODER_LAYERS = 1;
    auto algo = ov::genai::CacheEvictionAlgorithm(config, DEFAULT_BLOCK_SIZE, NUM_DECODER_LAYERS);

    auto scores_phase_1 = get_mock_scores(NUM_DECODER_LAYERS, algo.get_max_cache_size_after_eviction() + BLOCKS_TO_EVICT * DEFAULT_BLOCK_SIZE);
    for (auto& scores_per_layer : scores_phase_1) {
        // ones
        fill_scores(scores_per_layer, 0, scores_per_layer.get_size(), 1.0);
    }

    algo.register_new_token_scores(scores_phase_1);
    auto blocks_to_evict_phase_1 = algo.evict_logical_blocks();
    ASSERT_GT(blocks_to_evict_phase_1.size(), 0);
    ASSERT_EQ(blocks_to_evict_phase_1[0].size(), BLOCKS_TO_EVICT);

    const std::set<size_t> zeroed_blocks_in_phase_2{14, 3, 17, 21};  // only 14, 17 and 21 are in evictable range

    auto scores_phase_2 = get_mock_scores(NUM_DECODER_LAYERS, algo.get_max_cache_size_after_eviction() + BLOCKS_TO_EVICT * DEFAULT_BLOCK_SIZE);
    for (auto& scores_per_layer : scores_phase_2) {
        // zeroes for tokens that are expected to be evicted and large background score for the rest
        fill_scores(scores_per_layer, 0, scores_per_layer.get_size(), 1000.0);
        for (size_t target_block_idx : zeroed_blocks_in_phase_2) {
            fill_scores(scores_per_layer, DEFAULT_BLOCK_SIZE * target_block_idx,
                        DEFAULT_BLOCK_SIZE * (target_block_idx + 1), 0.0);
        }
    }

    algo.register_new_token_scores(scores_phase_2);

    const std::set<size_t> ref_evicted_blocks = {14, 17, 21};

    auto blocks_to_evict_phase_2 = algo.evict_logical_blocks();

    for (const auto& test_evicted_blocks : blocks_to_evict_phase_2) {
        EXPECT_EQ(test_evicted_blocks, ref_evicted_blocks);
    }

}


INSTANTIATE_TEST_SUITE_P(VariousAggregationModes, CacheEvictionConfigModeCommonBehaviour,
                         ::testing::ValuesIn(SCORE_ACCUMULATION_TEST_CASES));

struct CacheEvictionConfigInitParamsForTest {
    size_t start_size;
    size_t recent_size;
    size_t max_cache_size;
};

using CacheEvictionConfigInitializationTest = ::testing::TestWithParam<CacheEvictionConfigInitParamsForTest>;

const std::vector<CacheEvictionConfigInitParamsForTest> INVALID_CONFIG_INIT_PARAMS_CASES = {
        // zero area sizes
        {32, 32, 64},
        {0, 13, 39},
        {128, 0, 384},

        // max_cache_size less than start_size + recent_size
        {32, 64, 32},
};

TEST_P(CacheEvictionConfigInitializationTest, ThrowsForInvalidConfigParams) {
    auto params = GetParam();
    EXPECT_THROW(ov::genai::CacheEvictionConfig(params.start_size, params.recent_size, params.max_cache_size, ov::genai::AggregationMode::NORM_SUM), ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(VariousInvalidInitParams, CacheEvictionConfigInitializationTest,
                         ::testing::ValuesIn(INVALID_CONFIG_INIT_PARAMS_CASES));

struct CacheEvictionAlgoInitParamsForTest {
    ov::genai::CacheEvictionConfig config;
    size_t block_size;
    size_t num_decoder_layers;
};

using CacheEvictionAlgoInitializationTest = ::testing::TestWithParam<CacheEvictionAlgoInitParamsForTest>;

const std::vector<CacheEvictionAlgoInitParamsForTest> INVALID_ALGO_INIT_PARAMS_CASES = {
        // area sizes not multiple of block size
        { {32, 32, 97, ov::genai::AggregationMode::SUM}, 16, 8},
        { {11, 13, 50, ov::genai::AggregationMode::NORM_SUM}, 13, 1},
        { {128, 200, 584, ov::genai::AggregationMode::NORM_SUM}, 128, 19},

        // zero decoder layers
        { {32, 64, 192, ov::genai::AggregationMode::SUM}, 32, 0},
};
TEST_P(CacheEvictionAlgoInitializationTest, ThrowsForInvalidConfigs) {
    auto params = GetParam();
    EXPECT_THROW(ov::genai::CacheEvictionAlgorithm(params.config, params.block_size, params.num_decoder_layers), ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(VariousInvalidInitParams, CacheEvictionAlgoInitializationTest,
                         ::testing::ValuesIn(INVALID_ALGO_INIT_PARAMS_CASES));

TEST(CacheRotationCalculatorTest, CanInitializeWithBasicParams) {
    EXPECT_NO_THROW(ov::genai::CacheRotationCalculator(32, 128, 64));
}

TEST(CacheRotationCalculatorTest, ThrowsForNonPositiveTheta) {
    EXPECT_THROW(ov::genai::CacheRotationCalculator(32, 128, 64, -1.0), ov::Exception);
    EXPECT_THROW(ov::genai::CacheRotationCalculator(32, 128, 64, 0.0), ov::Exception);
}

struct CacheRotationCalculatorInitParams {
    size_t block_size;
    size_t max_context_length;
    size_t kv_head_size;
    double rope_theta;
};

struct CacheRotationCalculatorInputTestStruct {
    CacheRotationCalculatorInitParams init_params;
    std::set<size_t> evicted_block_logical_indices;
    size_t num_logical_blocks_before_eviction;
};

using CacheRotationCalculatorInvalidInputParameterizedTest =
    ::testing::TestWithParam<CacheRotationCalculatorInputTestStruct>;

// clang-format off
const std::vector<CacheRotationCalculatorInputTestStruct> CACHE_ROTATION_CALCULATOR_INVALID_INPUT_TEST_CASES = {
        {   // more num_logical_blocks_before_eviction than possible by max_context_length
            {8, 16, 4, 1337.0},
            {1, 2, 6, 39},
            32
        },
        {   // evicted block index out of bounds
            {16, 256, 32, 665.0},
            {8, 0, 5,50},
            9
        },
        {   // more blocks attempted to evict than num_logical_blocks_before_eviction
            {16, 256, 32, 665.0},
            {0, 1, 2},
            2
        }
};
// clang-format on

TEST_P(CacheRotationCalculatorInvalidInputParameterizedTest, ThrowsForInvalidEvictedBlocksInput) {
    const auto& test_struct = GetParam();
    const auto& init_params = test_struct.init_params;

    auto calc = ov::genai::CacheRotationCalculator(init_params.block_size,
                                                   init_params.max_context_length,
                                                   init_params.kv_head_size,
                                                   init_params.rope_theta);
    EXPECT_THROW(calc.get_rotation_data(test_struct.evicted_block_logical_indices,
                                                test_struct.num_logical_blocks_before_eviction),
                 ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(VariousInputsAndInitParams,
                         CacheRotationCalculatorInvalidInputParameterizedTest,
                         testing::ValuesIn(CACHE_ROTATION_CALCULATOR_INVALID_INPUT_TEST_CASES));

struct CacheRotationCalculatorNumCoefficientsTestStruct {
    CacheRotationCalculatorInitParams init_params;
    std::set<size_t> evicted_block_logical_indices;
    size_t num_logical_blocks_before_eviction;
    size_t expected_num_rotated_blocks;
};

// clang-format off
const std::vector<CacheRotationCalculatorNumCoefficientsTestStruct> CACHE_ROTATION_CALCULATOR_VALID_INPUT_TEST_CASES = {
        {
                {8, 512, 4, 1337.0},
                {1, 2},
                7,
                4
        },
        {
                {16, 256, 32, 665.0},
                {8, 0, 5, 3},
                9,
                5
        },
        {   // more blocks attempted to evict than num_logical_blocks_before_eviction
                {16, 1024, 32, 665.0},
                {24, 25, 26, 27, 28},
                30,
                1
        }
};
// clang-format on

using CacheRotationCalculatorNumCoefficientsParameterizedTest =
    ::testing::TestWithParam<CacheRotationCalculatorNumCoefficientsTestStruct>;

TEST_P(CacheRotationCalculatorNumCoefficientsParameterizedTest, GivesCorrectNumberOfRotationMultipliers) {
    const auto& test_struct = GetParam();
    const auto& init_params = test_struct.init_params;

    auto calc = ov::genai::CacheRotationCalculator(init_params.block_size,
                                                   init_params.max_context_length,
                                                   init_params.kv_head_size,
                                                   init_params.rope_theta);

    const auto rotation_multipliers = calc.get_rotation_data(test_struct.evicted_block_logical_indices,
                                                             test_struct.num_logical_blocks_before_eviction,
                                                             /* deltas_only = */ false);

    ASSERT_EQ(rotation_multipliers.size(), test_struct.expected_num_rotated_blocks);
    for (const auto& block_rotation_data : rotation_multipliers) {
        EXPECT_EQ(block_rotation_data.cosines.size(), block_rotation_data.sines.size());
        EXPECT_EQ(block_rotation_data.cosines.size(), init_params.block_size);
        for (const auto& token_coefficients : block_rotation_data.cosines) {
            EXPECT_EQ(token_coefficients.size(), init_params.kv_head_size / 2);
        }
        for (const auto& token_coefficients : block_rotation_data.sines) {
            EXPECT_EQ(token_coefficients.size(), init_params.kv_head_size / 2);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(VariousInputsAndInitParams,
                         CacheRotationCalculatorNumCoefficientsParameterizedTest,
                         testing::ValuesIn(CACHE_ROTATION_CALCULATOR_VALID_INPUT_TEST_CASES));

struct CacheRotationCalculatorRefCoefficientsTestStruct {
    CacheRotationCalculatorInitParams init_params;
    std::set<size_t> evicted_block_logical_indices;
    size_t num_logical_blocks_before_eviction;
    std::vector<ov::genai::CacheRotationCalculator::BlockRotationData> expected_rotation_data;
};

// clang-format off
const std::vector<CacheRotationCalculatorRefCoefficientsTestStruct> CACHE_ROTATION_CALCULATOR_REF_COEFFICIENTS_TEST_CASES = {
        // 0
        {
                {2, 512, 4, 1.0},
                {1, 2},
                4,
                // pre-eviction block 3 rotated left by 2 blocks, coefficients are cos(4) and -sin(4) due to theta == 1.0
                {
                        {1, 4,
                                {
                                        {0.75680249, 0.75680249},  // block token 0
                                        {0.75680249, 0.75680249}   // block token 1
                                },
                                {
                                        {-0.65364362, -0.65364362},  // block token 0
                                        {-0.65364362, -0.65364362}   // block token 1
                                },
                        }
                }
        },

        // 1 - same as 0, but adjusted theta
        {
                {2, 512, 4, 2.0},
                {1, 2},
                4,
                // coefficients are [cos(4 / 1),  -sin(4 / 1)], [cos(4 / sqrt(2)), -sin(4 / sqrt(2))] now
                {
                        {1, 4,
                                {
                                        {0.75680249, -0.30807174},  // block token 0
                                        {0.75680249, -0.30807174}   // block token 1
                                },
                                {
                                        {-0.65364362, -0.95136312},  // block token 0
                                        {-0.65364362, -0.95136312}   // block token 1
                                },
                        }
                }
        },
        // 2 - same as 0, but corner case blocks
        {
                {2, 512, 4, 2.0},
                {0, 3},
                4,
                // delta of 2 tokens for both blocks
                // coefficients are [cos(2 / 1),  -sin(2 / 1)], [cos(2 / sqrt(2)), -sin(2 / sqrt(2))]
                {
                        {0, 2,
                                {
                                        {-0.90929742, -0.98776594},  // block token 0
                                        {-0.90929742, -0.98776594}   // block token 1
                                },
                                {
                                        {-0.41614683, 0.15594369},  // block token 0
                                        {-0.41614683, 0.15594369}   // block token 1
                                },
                        },
                        {1, 2,
                                {
                                        {-0.90929742, -0.98776594},  // block token 0
                                        {-0.90929742, -0.98776594}   // block token 1
                                },
                                {
                                        {-0.41614683, 0.15594369},  // block token 0
                                        {-0.41614683, 0.15594369}   // block token 1
                                },
                        }
                }
        },
        // 3 - same as 0, but different deltas for each rotated block
        {
                {2, 512, 4, 2.0},
                {0, 2},
                4,
                // delta of 2 tokens for first remaining block:
                //  coefficients are [cos(2 / 1),  -sin(2 / 1)], [cos(2 / sqrt(2)), -sin(2 / sqrt(2))]
                // and 4 tokens for second remaining block
                // coefficients are [cos(4 / 1),  -sin(4 / 1)], [cos(4 / sqrt(2)), -sin(4 / sqrt(2))]
                {
                        {0, 2,
                                {
                                        {-0.90929742, -0.98776594},  // block token 0
                                        {-0.90929742, -0.98776594}   // block token 1
                                },
                                {
                                        {-0.41614683, 0.15594369},  // block token 0
                                        {-0.41614683, 0.15594369}   // block token 1
                                },
                        },
                        {1, 4,
                                {
                                        {0.75680249, -0.30807174},  // block token 0
                                        {0.75680249, -0.30807174}   // block token 1
                                },
                                {
                                        {-0.65364362, -0.95136312},  // block token 0
                                        {-0.65364362, -0.95136312}   // block token 1
                                },
                        }
                }
        },
};
// clang-format on

using CacheRotationCalculatorRefCoefficientsParameterizedTest =
    ::testing::TestWithParam<CacheRotationCalculatorRefCoefficientsTestStruct>;

void compare_rotation_data(const std::vector<ov::genai::CacheRotationCalculator::BlockRotationData>& test_data,
                           const std::vector<ov::genai::CacheRotationCalculator::BlockRotationData>& ref_data,
                           double abs_tol = 1e-8) {
    ASSERT_EQ(test_data.size(), ref_data.size());

    for (size_t i = 0; i < test_data.size(); i++) {
        const auto& test_block_data = test_data[i];
        const auto& ref_block_data = ref_data[i];
        EXPECT_EQ(test_block_data.logical_block_idx, ref_block_data.logical_block_idx);

        ASSERT_EQ(test_block_data.sines.size(), ref_block_data.sines.size());
        for (size_t j = 0; j < test_block_data.sines.size(); j++) {
            EXPECT_THAT(test_block_data.sines[j],
                        ::testing::Pointwise(::testing::DoubleNear(abs_tol), ref_block_data.sines[j]));
        }

        ASSERT_EQ(test_block_data.cosines.size(), ref_block_data.cosines.size());
        for (size_t j = 0; j < test_block_data.cosines.size(); j++) {
            EXPECT_THAT(test_block_data.cosines[j],
                        ::testing::Pointwise(::testing::DoubleNear(abs_tol), ref_block_data.cosines[j]));
        }
    }
}

TEST_P(CacheRotationCalculatorRefCoefficientsParameterizedTest, CalculatedCoefficientsMatchToReference) {
    const auto& test_struct = GetParam();
    const auto& init_params = test_struct.init_params;

    auto calc = ov::genai::CacheRotationCalculator(init_params.block_size,
                                                   init_params.max_context_length,
                                                   init_params.kv_head_size,
                                                   init_params.rope_theta);

    const auto rotation_multipliers = calc.get_rotation_data(test_struct.evicted_block_logical_indices,
                                                             test_struct.num_logical_blocks_before_eviction,
                                                             /* deltas_only = */ false);

    compare_rotation_data(rotation_multipliers, test_struct.expected_rotation_data);
}

INSTANTIATE_TEST_SUITE_P(VariousInputsAndInitParams,
                         CacheRotationCalculatorRefCoefficientsParameterizedTest,
                         testing::ValuesIn(CACHE_ROTATION_CALCULATOR_REF_COEFFICIENTS_TEST_CASES));

using CacheRotationCalculatorPOCRefCoefficientsTest = ::testing::TestWithParam<std::string>;
TEST_P(CacheRotationCalculatorPOCRefCoefficientsTest, CalculatedCoefficientsAreSimilarToPOCResults) {
    std::filesystem::path base_dir("tests/cpp/data/");
    std::ifstream input_file(base_dir / GetParam(), std::ios::in);

    const size_t ref_max_context_length = 1024;
    size_t ref_block_size = 0;
    size_t ref_head_size = 0;

    input_file >> ref_block_size;
    input_file >> ref_head_size;

    size_t num_blocks_before_eviction = 0;
    size_t num_evicted_blocks = 0;
    size_t num_rotated_blocks = 0;

    std::set<size_t> ref_evicted_logical_block_indices;
    std::vector<ov::genai::CacheRotationCalculator::BlockRotationData> ref_data;

    input_file >> num_blocks_before_eviction;
    input_file >> num_evicted_blocks;
    for (size_t i = 0; i < num_evicted_blocks; i++) {
        size_t evicted_block_idx = 0;
        input_file >> evicted_block_idx;
        ref_evicted_logical_block_indices.insert(evicted_block_idx);
    }

    input_file >> num_rotated_blocks;
    ref_data.resize(num_rotated_blocks);

    for (size_t i = 0; i < num_rotated_blocks; i++) {
        size_t logical_block_idx_after_eviction = 0;
        input_file >> logical_block_idx_after_eviction;
        ref_data[i].logical_block_idx = logical_block_idx_after_eviction;
        std::vector<float> coeffts(ref_head_size / 2);

        for (size_t j = 0; j < ref_head_size / 2; j++) {
            input_file >> coeffts[j];
        }
        ref_data[i].sines.resize(ref_block_size);
        for (size_t k = 0; k < ref_block_size; k++) {
            ref_data[i].sines[k] = coeffts;
        }

        for (size_t j = 0; j < ref_head_size / 2; j++) {
            input_file >> coeffts[j];
        }
        ref_data[i].cosines.resize(ref_block_size);
        for (size_t k = 0; k < ref_block_size; k++) {
            ref_data[i].cosines[k] = coeffts;
        }
    }

    auto calc = ov::genai::CacheRotationCalculator(ref_block_size, ref_max_context_length, ref_head_size);
    auto test_data = calc.get_rotation_data(ref_evicted_logical_block_indices, num_blocks_before_eviction,
                                            /* deltas_only = */ false);
    compare_rotation_data(test_data, ref_data, 1e-2);  // the dump values were originally calculated in FP16 precision
}

INSTANTIATE_TEST_SUITE_P(VariousPOCDumps,
                         CacheRotationCalculatorPOCRefCoefficientsTest,
                         testing::Values("cache_rotation_poc_ref_coefficients_per_block_0.txt",
                                         "cache_rotation_poc_ref_coefficients_per_block_1.txt",
                                         "cache_rotation_poc_ref_coefficients_per_block_2.txt",
                                         "cache_rotation_poc_ref_coefficients_per_block_3.txt"));
