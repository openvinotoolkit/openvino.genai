// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cache_eviction.hpp"
#include "gtest/gtest.h"

#include <algorithm>

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
