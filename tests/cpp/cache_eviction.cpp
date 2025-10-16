// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache_eviction.hpp"

#include <algorithm>
#include <fstream>
#include <filesystem>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "openvino/genai/cache_eviction.hpp"

using ov::genai::KVCrushAnchorPointMode;
using ov::genai::KVCrushConfig;

const ov::genai::CacheEvictionConfig DEFAULT_CACHE_EVICTION_CONFIG =
    {32, 32, 192, ov::genai::AggregationMode::NORM_SUM, false, 0, KVCrushConfig(0, KVCrushAnchorPointMode::MEAN)};
const ov::genai::CacheEvictionConfig SHORT_RECENT_EVICTION_CONFIG =
    {32, 32, 72, ov::genai::AggregationMode::NORM_SUM, false, 0, KVCrushConfig(0, KVCrushAnchorPointMode::MEAN)};
const ov::genai::CacheEvictionConfig KVCRUSH_CACHE_EVICTION_CONFIG =
    {32, 32, 192, ov::genai::AggregationMode::NORM_SUM, false, 0, KVCrushConfig(2, KVCrushAnchorPointMode::MEAN)};
constexpr size_t DEFAULT_BLOCK_SIZE = 4;
constexpr size_t DEFAULT_NUM_DECODER_LAYERS = 2;
constexpr size_t DEFAULT_MAX_POOL_WINDOW_SIZE = 8;
constexpr ov::genai::AggregationMode DEFAULT_AGGREGATION_MODE = ov::genai::AggregationMode::NORM_SUM;

AttentionScoresForEachDecoderLayer get_layer_scores_from_2d_vector(const std::vector<std::vector<float>>& src) {
    AttentionScoresForEachDecoderLayer retval;
    retval.reserve(src.size());
    for (const auto& val_vec : src) {
        retval.push_back(ov::Tensor(ov::element::f32, {val_vec.size()}));
        ov::Tensor& created_tensor = retval.back();
        std::copy(val_vec.begin(), val_vec.end(), created_tensor.data<float>());
    }
    return retval;
}



TEST(EvictionScoreManager, CanRegisterNewScores) {
    ov::genai::EvictionScoreManager mgr(DEFAULT_BLOCK_SIZE, DEFAULT_NUM_DECODER_LAYERS, DEFAULT_MAX_POOL_WINDOW_SIZE, DEFAULT_AGGREGATION_MODE, 0);
    auto mock_scores = get_layer_scores_from_2d_vector( { {0.0, 1.0}, {-2.0, -42.0} } );
    EXPECT_NO_THROW(mgr.register_new_token_scores(mock_scores, {}));
}


struct EvictionScoreManagerAddWithSkipsTestStruct {
    size_t block_size;
    std::vector<double> src;
    std::set<size_t> skipped_logical_block_ids;
    std::vector<double> dst_before;
    std::vector<double> ref_dst_after;
};

using EvictionScoreManagerAddWithSkipsParameterizedTest = ::testing::TestWithParam<EvictionScoreManagerAddWithSkipsTestStruct>;

const std::vector<EvictionScoreManagerAddWithSkipsTestStruct> ADD_WITH_SKIPS_TEST_CASES = {
    // basic case
    {
      2,
      {0.0, 1.0,
       2.0, 3.0}, {1},
      {0.0, 0.0,
       0.0, 0.0,
       0.0, 0.0},
      {0.0, 1.0,
       0.0, 0.0,
       2.0, 3.0}
    },

    // another block size
    {
      3,
      {-1.0, 1.5, 12.8,
        3.0, -9.4, 0.1}, {1},
      {1.0, 1.0, 1.0,
       2.0, 2.0, 2.0,
       -3.0, -3.0, -3.0},
      {0.0, 2.5, 13.8,
       2.0, 2.0, 2.0,
       0.0, -12.4, -2.9}
    },

    // corner case skipped blocks
    {
      3,
      {-1.0, 1.5, 12.8,
        3.0, -9.4, 0.1}, {0, 3},
      {1.0, 1.0, 1.0,
       2.0, 2.0, 2.0,
       -3.0, -3.0, -3.0,
       12.5, -7.2, 9.6},
      {1.0, 1.0, 1.0,
       1.0, 3.5, 14.8,
       0.0, -12.4, -2.9,
       12.5, -7.2, 9.6}
    },

    // non-contiguous skipped blocks
    {
      2,
      {-1.0, 1.5,
        12.8, 3.0,
        -13.0, -87.0 }, {1, 2, 4},
      {1.0, 1.0,
       1.0, 2.0,
       2.0, 2.0,
       -3.0, -3.0,
       -3.0, 12.5,
       42.0, -1337.0},
      {0.0, 2.5,
       1.0, 2.0,
       2.0, 2.0,
       9.8, 0.0,
       -3.0, 12.5,
       29.0, -1424.0}
    },

    // no blocks skipped
    {
      2,
      { 5.9, 0.3,
        -1.0, 1.5,
        12.8, 3.0,
        -13.0, -87.0 }, {},
      {1.0, 1.0,
       1.0, 2.0,
       2.0, 2.0,
       -3.0, -3.0},
      {6.9, 1.3,
       0.0, 3.5,
       14.8, 5.0,
       -16.0, -90.0}
    },

    // all blocks skipped
    {
      2,
      {}, {0, 1, 2, 3},
      {1.0, 1.0,
       1.0, 2.0,
       2.0, 2.0,
       -3.0, -3.0},
      {1.0, 1.0,
       1.0, 2.0,
       2.0, 2.0,
       -3.0, -3.0},
    },

    // non-block aligned values
    {
      2,
      { 5.9, 0.3,
        12.8, 3.0,
        -13.0 }, {1},
      {1.0, 1.0,
       1.0, 2.0,
       2.0, 2.0,
       -3.0},
      {6.9, 1.3,
       1.0, 2.0,
       14.8, 5.0,
       -16.0}
    },
};

TEST_P(EvictionScoreManagerAddWithSkipsParameterizedTest, CanAddWithSkips) {
    const auto& test_struct = GetParam();
    ov::genai::EvictionScoreManager mgr(test_struct.block_size, DEFAULT_NUM_DECODER_LAYERS, DEFAULT_MAX_POOL_WINDOW_SIZE, DEFAULT_AGGREGATION_MODE, 0);
    auto dst = test_struct.dst_before;
    mgr.add_with_skips(dst, test_struct.src, test_struct.skipped_logical_block_ids);
    EXPECT_EQ(dst.size(), test_struct.dst_before.size());
    EXPECT_EQ(dst, test_struct.ref_dst_after);
}

INSTANTIATE_TEST_SUITE_P(VariousInputs, EvictionScoreManagerAddWithSkipsParameterizedTest, ::testing::ValuesIn(ADD_WITH_SKIPS_TEST_CASES));


struct EvictionScoreManagerRegisterScoresTestStruct {
    std::string test_id;
    size_t block_size;
    ov::genai::AggregationMode aggregation_mode;
    size_t max_pool_window_size;
    size_t ignore_first_n_blocks;
    size_t snapkv_window_size;

    std::vector<std::pair<std::vector<std::vector<float>>, std::set<size_t>>> scores_and_skips;
    std::vector<std::vector<float>> ref_scores;
    std::vector<std::vector<size_t>> ref_counters;
};

using EvictionScoreManagerRegisterScoresParameterizedTest = ::testing::TestWithParam<EvictionScoreManagerRegisterScoresTestStruct>;

const std::vector<EvictionScoreManagerRegisterScoresTestStruct> REGISTER_SCORES_TEST_CASES = {
    { "basic_case_sum",
      /* block_size =*/ 2, /* aggregation_mode = */ ov::genai::AggregationMode::SUM, /* max_pool_window_size = */ 0, /* ignore_first_n_blocks = */ 0, /* snapkv_window_size = */ 0,
      {
          { {{1.5, -0.8, 4.1, 7.7},
             {-0.9, 1.4, 6.4, -9.0}}, {} }
      },

      { {1.5, -0.8, 4.1, 7.7},
        {-0.9, 1.4, 6.4, -9.0} },
      { {},
        {} }
    },
    { "basic_case_norm_sum",
      /* block_size =*/ 2, /* aggregation_mode = */ ov::genai::AggregationMode::NORM_SUM, /* max_pool_window_size = */ 0, /* ignore_first_n_blocks = */ 0, /* snapkv_window_size = */ 0,
      {
          { {{1.5, -0.8, 4.1, 7.7},
             {-0.9, 1.4, 6.4, -9.0}}, {} }
      },

      { {1.5, -0.8, 4.1, 7.7},
        {-0.9, 1.4, 6.4, -9.0} },
      { {4, 3, 2, 1},
        {4, 3, 2, 1} }
    },
    { "two_scores_sum",
      /* block_size =*/ 2, /* aggregation_mode = */ ov::genai::AggregationMode::SUM, /* max_pool_window_size = */ 0, /* ignore_first_n_blocks = */ 0, /* snapkv_window_size = */ 0,
      {
          { {{1.5, -0.8, 4.1, 7.7},
             {-0.9, 1.4, 6.4, -9.0}}, {} },
          { {{-7.4, 2.6, 8.9, -0.1},
             {-3.1, -8.2, 5.9, 7.6}}, {} }
      },

      { {-5.9, 1.8, 13.0, 7.6},
        {-4.0, -6.8, 12.3, -1.4} },
      { {},
        {} }
    },
    { "two_scores_norm_sum",
      /* block_size =*/ 2, /* aggregation_mode = */ ov::genai::AggregationMode::NORM_SUM, /* max_pool_window_size = */ 0, /* ignore_first_n_blocks = */ 0, /* snapkv_window_size = */ 0,
      {
          { {{1.5, -0.8, 4.1, 7.7},
             {-0.9, 1.4, 6.4, -9.0}}, {} },
          { {{-7.4, 2.6, 8.9, -0.1},
             {-3.1, -8.2, 5.9, 7.6}}, {} }
      },

      { {-5.9, 1.8, 13.0, 7.6},
        {-4.0, -6.8, 12.3, -1.4} },
      { {4, 3, 2, 1},
        {4, 3, 2, 1} }
    },

    { "more_scores_second_time",
      /* block_size =*/ 2, /* aggregation_mode = */ ov::genai::AggregationMode::NORM_SUM, /* max_pool_window_size = */ 0, /* ignore_first_n_blocks = */ 0, /* snapkv_window_size = */ 0,
      {
          { {{1.5, -0.8, 4.1, 7.7},
             {-0.9, 1.4, 6.4, -9.0}}, {} },
          { {{-7.4, 2.6, 8.9, -0.1, 3.5},
             {-3.1, -8.2, 5.9, 7.6, -1.0}}, {} }
      },

      { {-5.9, 1.8, 13.0, 7.6, 3.5},
        {-4.0, -6.8, 12.3, -1.4, -1.0} },
      { {5, 4, 3, 2, 1},
        {5, 4, 3, 2, 1} }
    },
    { "less_scores_second_time_with_skips",
      /* block_size =*/ 2, /* aggregation_mode = */ ov::genai::AggregationMode::NORM_SUM, /* max_pool_window_size = */ 0, /* ignore_first_n_blocks = */ 0, /* snapkv_window_size = */ 0,
      {
          { {{1.5, -0.8, 4.1, 7.7, 3.6, -7.4},
             {-0.9, 1.4, 6.4, -9.0, 8.1, 2.6}}, {} },
          { {{-7.4, 2.6, 8.9},
             {-3.1, -8.2, 5.9}}, {1, 2} }
      },

      { {-5.9, 1.8, 4.1, 7.7, 3.6, -7.4, 8.9},
        {-4.0, -6.8, 6.4, -9.0, 8.1, 2.6, 5.9} },
      { {7, 6, 5, 4, 3, 2, 1},
        {7, 6, 5, 4, 3, 2, 1} }
    },
    { "with_ignore_first_n_blocks_base",
      /* block_size =*/ 2, /* aggregation_mode = */ ov::genai::AggregationMode::NORM_SUM, /* max_pool_window_size = */ 0, /* ignore_first_n_blocks = */ 1, /* snapkv_window_size = */ 0,
      {
          { {{1.5, -0.8, 4.1, 7.7, 3.6, -7.4},
             {-0.9, 1.4, 6.4, -9.0, 8.1, 2.6}}, {} },
          { {{-7.4, 2.6, 8.9, 3.5, 7.4, -3.8},
             {-3.1, -8.2, 5.9, -7.9, -5.8, 1.7}}, {} }
      },

      { {13.0, 11.2, 11.0, -11.2},
        {12.3, -16.9, 2.3, 4.3} },
      { {4, 3, 2, 1},
        {4, 3, 2, 1} }
    },
    { "with_ignore_first_n_blocks_more_scores_second_time",
      /* block_size =*/ 2, /* aggregation_mode = */ ov::genai::AggregationMode::NORM_SUM, /* max_pool_window_size = */ 0, /* ignore_first_n_blocks = */ 1, /* snapkv_window_size = */ 0,
      {
          { {{1.5, -0.8, 4.1, 7.7, 3.6, -7.4},
             {-0.9, 1.4, 6.4, -9.0, 8.1, 2.6}}, {} },
          { {{-7.4, 2.6, 8.9, 3.5, 7.4, -3.8, 0.1},
             {-3.1, -8.2, 5.9, -7.9, -5.8, 1.7, -0.2}}, {} }
      },

      { {13.0, 11.2, 11.0, -11.2, 0.1},
        {12.3, -16.9, 2.3, 4.3, -0.2} },
      { {5, 4, 3, 2, 1},
        {5, 4, 3, 2, 1} }
    },
    { "with_ignore_first_n_blocks_less_second_time_with_skips",
      /* block_size =*/ 2, /* aggregation_mode = */ ov::genai::AggregationMode::NORM_SUM, /* max_pool_window_size = */ 0, /* ignore_first_n_blocks = */ 1, /* snapkv_window_size = */ 0,
      {
          { {{1.5, -0.8, 4.1, 7.7, 3.6, -7.4},
             {-0.9, 1.4, 6.4, -9.0, 8.1, 2.6}}, {} },
          { {{-7.4, 2.6, 8.9},
             {-3.1, -8.2, 5.9}}, {1, 2} }
      },

      { {4.1, 7.7, 3.6, -7.4, 8.9},
        {6.4, -9.0, 8.1, 2.6, 5.9} },
      { {5, 4, 3, 2, 1},
        {5, 4, 3, 2, 1} }
    },
    { "with_max_pool_base",
      /* block_size =*/ 2, /* aggregation_mode = */ ov::genai::AggregationMode::NORM_SUM, /* max_pool_window_size = */ 3, /* ignore_first_n_blocks = */ 0, /* snapkv_window_size = */ 0,
      {
          { {{1.5, -0.8, 4.1, 7.7, 3.6, -7.4},
             {-0.9, 1.4, 6.4, -9.0, 8.1, 2.6}}, {} },
      },

      { {4.1, 7.7, 7.7, 7.7, 3.6, -7.4},
        {6.4, 6.4, 8.1, 8.1, 8.1, 2.6} },
      { {6, 5, 4, 3, 2, 1},
        {6, 5, 4, 3, 2, 1} }
    },
    { "with_max_pool_two_score",
      /* block_size =*/ 2, /* aggregation_mode = */ ov::genai::AggregationMode::NORM_SUM, /* max_pool_window_size = */ 3, /* ignore_first_n_blocks = */ 0, /* snapkv_window_size = */ 0,
      {
          { {{1.5, -0.8, 4.1, 7.7, 3.6, -7.4},
             {-0.9, 1.4, 6.4, -9.0, 8.1, 2.6}}, {} },
          { {{-7.4, 2.6, 8.9, 3.5, 7.4, -3.8, 0.1},
             {-3.1, -8.2, 5.9, -7.9, -5.8, 1.7, -0.2}}, {} }
      },

      { {13.0, 16.6, 16.6, 15.1, 11.0, -7.3, 0.1},
        {12.3, 12.3, 14.0, 9.8, 9.8, 4.3, -0.2} },
      { {7, 6, 5, 4, 3, 2, 1},
        {7, 6, 5, 4, 3, 2, 1} }
    },
    { "with_max_pool_ignore_and_skips",
      /* block_size =*/ 2, /* aggregation_mode = */ ov::genai::AggregationMode::NORM_SUM, /* max_pool_window_size = */ 3, /* ignore_first_n_blocks = */ 1, /* snapkv_window_size = */ 0,
      {
          { {{1.5, -0.8, 4.1, 7.7, 3.6, -7.4},
             {-0.9, 1.4, 6.4, -9.0, 8.1, 2.6}}, {} },
          { {{-7.4, 2.6, 8.9},
             {-3.1, -8.2, 5.9}}, {1, 2} }
      },

      { {7.7, 7.7, 3.6, -7.4, 8.9},
        {8.1, 8.1, 8.1, 2.6, 5.9} },
      { {5, 4, 3, 2, 1},
        {5, 4, 3, 2, 1} }
    },
};

TEST_P(EvictionScoreManagerRegisterScoresParameterizedTest, ScoresAndCountersAfterRegistrationAreCorrect) {
    const auto& test_struct = GetParam();
    ov::genai::EvictionScoreManager mgr(test_struct.block_size, DEFAULT_NUM_DECODER_LAYERS, test_struct.max_pool_window_size, test_struct.aggregation_mode, test_struct.ignore_first_n_blocks, test_struct.snapkv_window_size);
    for (const auto& score_and_skip : test_struct.scores_and_skips) {
        mgr.register_new_token_scores(get_layer_scores_from_2d_vector(score_and_skip.first), score_and_skip.second);
    }
    const auto& test_scores = mgr.get_scores();
    const auto& test_counters = mgr.get_counters();
    ASSERT_EQ(test_scores.size(), DEFAULT_NUM_DECODER_LAYERS);
    ASSERT_EQ(test_counters.size(), DEFAULT_NUM_DECODER_LAYERS);

    float abs_tol = 1e-6;
    for (size_t layer_idx = 0; layer_idx < DEFAULT_NUM_DECODER_LAYERS; layer_idx++) {
        EXPECT_THAT(test_scores[layer_idx], ::testing::Pointwise(::testing::DoubleNear(abs_tol), test_struct.ref_scores[layer_idx]));
        EXPECT_EQ(mgr.get_counters(), test_struct.ref_counters);
    }
}

INSTANTIATE_TEST_SUITE_P(VariousInputs, EvictionScoreManagerRegisterScoresParameterizedTest, ::testing::ValuesIn(REGISTER_SCORES_TEST_CASES),
                         [](const testing::TestParamInfo<EvictionScoreManagerRegisterScoresParameterizedTest::ParamType>& info) {
                             return info.param.test_id;
                         });

struct EvictionScoreManagerSnapKVCounterTestStruct {
    std::string test_id;
    size_t snapkv_window_size;

    std::vector<std::pair<size_t, size_t>> num_scores_and_num_snapkv_scores;
    std::vector<size_t> ref_counters; // expected to be equal for all layers
};

using EvictionScoreManagerSnapKVCounterParameterizedTest = ::testing::TestWithParam<EvictionScoreManagerSnapKVCounterTestStruct>;

const std::vector<EvictionScoreManagerSnapKVCounterTestStruct> SNAPKV_COUNTER_TEST_CASES = {
    {
        "snapkv_window_not_completely_filled",
        3,
        { {4, 1}, {5, 1} },
        {2, 2, 2, 2, 1},
    },
    {
        "snapkv_window_exactly_filled",
        4,
        { {3, 1}, {6, 3} },
        {4, 4, 4, 3, 2, 1},
    },
    {
        "snapkv_window_overflow_by_1",
        4,
        { {3, 1}, {7, 3} },
        {5, 5, 5, 4, 3, 2, 1},
    },
    {
        "snapkv_window_filled_followed_by_generation_step",
        4,
        { {3, 1}, {6, 3}, {7, 1} },
        {5, 5, 5, 4, 3, 2, 1},
    },
    {
        "snapkv_window_filled_followed_by_two_generation_steps",
        4,
        { {3, 1}, {6, 3}, {7, 1}, {8, 1} },
        {6, 6, 6, 5, 4, 3, 2, 1},
    },
    {
        "snapkv_window_not_filled_in_multiple_steps",
        9,
        { {3, 3}, {4, 1}, {5, 1}, {8, 3} },
        {8, 7, 6, 5, 4, 3, 2, 1},
    },
    {
        "snapkv_window_filled_exactly_in_multiple_steps",
        8,
        { {3, 3}, {4, 1}, {5, 1}, {8, 3} },
        {8, 7, 6, 5, 4, 3, 2, 1},
    }
};


TEST_P(EvictionScoreManagerSnapKVCounterParameterizedTest, CountersAfterRegistrationAreCorrect) {
    const auto& test_struct = GetParam();
    ov::genai::EvictionScoreManager mgr(DEFAULT_BLOCK_SIZE, DEFAULT_NUM_DECODER_LAYERS, DEFAULT_MAX_POOL_WINDOW_SIZE, ov::genai::AggregationMode::NORM_SUM, /* ignore_first_n_blocks = */ 0, test_struct.snapkv_window_size);
    for (const auto& num_scores_num_snapkv_scores_pair : test_struct.num_scores_and_num_snapkv_scores) {
        size_t num_scores_to_register = num_scores_num_snapkv_scores_pair.first;
        size_t num_snapkv_scores = num_scores_num_snapkv_scores_pair.second;
        ASSERT_GE(num_scores_to_register, num_snapkv_scores);
        std::vector<std::vector<float>> mock_scores(DEFAULT_NUM_DECODER_LAYERS, std::vector<float>(num_scores_to_register, 0));
        mgr.register_new_token_scores(get_layer_scores_from_2d_vector(mock_scores), {}, num_snapkv_scores);
    }
    const auto& test_counters = mgr.get_counters();
    ASSERT_EQ(test_counters.size(), DEFAULT_NUM_DECODER_LAYERS);

    for (size_t layer_idx = 0; layer_idx < DEFAULT_NUM_DECODER_LAYERS; layer_idx++) {
        EXPECT_EQ(mgr.get_counters()[layer_idx], test_struct.ref_counters);
    }
}

INSTANTIATE_TEST_SUITE_P(VariousInputs, EvictionScoreManagerSnapKVCounterParameterizedTest, ::testing::ValuesIn(SNAPKV_COUNTER_TEST_CASES),
                         [](const testing::TestParamInfo<EvictionScoreManagerSnapKVCounterParameterizedTest::ParamType>& info) {
                             return info.param.test_id;
                         });

struct EvictionScoreManagerRemoveScoresTestStruct {
    std::string test_id;
    size_t block_size;
    ov::genai::AggregationMode aggregation_mode;
    size_t max_pool_window_size;
    size_t ignore_first_n_blocks;

    std::vector<std::pair<std::vector<std::vector<float>>, std::set<size_t>>> scores_and_skips;

    std::vector<std::vector<size_t>> removed_block_ids;

    std::vector<std::vector<float>> ref_scores;
    std::vector<std::vector<size_t>> ref_counters;
};

using EvictionScoreManagerRemoveScoresParameterizedTest = ::testing::TestWithParam<EvictionScoreManagerRemoveScoresTestStruct>;

const std::vector<EvictionScoreManagerRemoveScoresTestStruct> REMOVE_SCORES_TEST_CASES = {
    { "nothing_to_remove",
      /* block_size =*/ 2, /* aggregation_mode = */ ov::genai::AggregationMode::NORM_SUM, /* max_pool_window_size = */ 0, /* ignore_first_n_blocks = */ 0,
      {
          { {{1.5, -0.8, 4.1, 7.7},
             {-0.9, 1.4, 6.4, -9.0}}, {} }
      },

      { {},
        {} },
      { {1.5, -0.8, 4.1, 7.7},
        {-0.9, 1.4, 6.4, -9.0} },
      { {4, 3, 2, 1},
        {4, 3, 2, 1} }
    },
    { "basic_case_sum",
      /* block_size =*/ 2, /* aggregation_mode = */ ov::genai::AggregationMode::SUM, /* max_pool_window_size = */ 0, /* ignore_first_n_blocks = */ 0,
      {
          { {{1.5, -0.8, 4.1, 7.7},
             {-0.9, 1.4, 6.4, -9.0}}, {} }
      },

      { {0},
        {1} },
      { {4.1, 7.7},
        {-0.9, 1.4} },
      { {},
        {} }
    },
    { "basic_case_norm_sum",
      /* block_size =*/ 2, /* aggregation_mode = */ ov::genai::AggregationMode::NORM_SUM, /* max_pool_window_size = */ 0, /* ignore_first_n_blocks = */ 0,
      {
          { {{1.5, -0.8, 4.1, 7.7},
             {-0.9, 1.4, 6.4, -9.0}}, {} }
      },

      { {0},
        {1} },
      { {4.1, 7.7},
        {-0.9, 1.4} },
      { {2, 1},
        {4, 3} }
    },
    { "two_adds_then_remove",
      /* block_size =*/ 2, /* aggregation_mode = */ ov::genai::AggregationMode::NORM_SUM, /* max_pool_window_size = */ 0, /* ignore_first_n_blocks = */ 0,
      {
          { {{1.5, -0.8, 4.1, 7.7, 3.6, -7.4},
             {-0.9, 1.4, 6.4, -9.0, 8.1, 2.6}}, {} },
          { {{-7.4, 2.6, 8.9},
             {-3.1, -8.2, 5.9}}, {1, 2} }
      },

      { {0, 2 },
        {0, 1, 3} },
      { {4.1, 7.7, 8.9},
        {8.1, 2.6} },
      { {5, 4, 1},
        {3, 2} }
    },
};

TEST_P(EvictionScoreManagerRemoveScoresParameterizedTest, ScoresAndCountersAfterRemovalAreCorrect) {
    const auto& test_struct = GetParam();
    ov::genai::EvictionScoreManager mgr(test_struct.block_size, DEFAULT_NUM_DECODER_LAYERS, test_struct.max_pool_window_size, test_struct.aggregation_mode, test_struct.ignore_first_n_blocks);
    for (const auto& score_and_skip : test_struct.scores_and_skips) {
        mgr.register_new_token_scores(get_layer_scores_from_2d_vector(score_and_skip.first), score_and_skip.second);
    }

    for (size_t decoder_layer_idx = 0; decoder_layer_idx < DEFAULT_NUM_DECODER_LAYERS; decoder_layer_idx++) {
        mgr.remove_scores(test_struct.removed_block_ids[decoder_layer_idx], decoder_layer_idx);
    }

    const auto& test_scores = mgr.get_scores();
    const auto& test_counters = mgr.get_counters();
    ASSERT_EQ(test_scores.size(), DEFAULT_NUM_DECODER_LAYERS);
    ASSERT_EQ(test_counters.size(), DEFAULT_NUM_DECODER_LAYERS);

    float abs_tol = 1e-6;
    for (size_t layer_idx = 0; layer_idx < DEFAULT_NUM_DECODER_LAYERS; layer_idx++) {
        EXPECT_THAT(test_scores[layer_idx], ::testing::Pointwise(::testing::DoubleNear(abs_tol), test_struct.ref_scores[layer_idx]));
        EXPECT_EQ(mgr.get_counters(), test_struct.ref_counters);
    }
}

INSTANTIATE_TEST_SUITE_P(VariousInputs, EvictionScoreManagerRemoveScoresParameterizedTest, ::testing::ValuesIn(REMOVE_SCORES_TEST_CASES),
                         [](const testing::TestParamInfo<EvictionScoreManagerRemoveScoresParameterizedTest::ParamType>& info) {
                             return info.param.test_id;
                         });

struct SnapKVScoreAggregationCalculatorTestStruct {
    std::string test_id;
    size_t snapkv_window_size;
    size_t prompt_len;
    size_t num_processed_tokens;
    size_t num_scheduled_tokens;
    size_t ref_num_scores_aggregated;
};

using SnapKVScoreAggregationCalculatorParameterizedTest = ::testing::TestWithParam<SnapKVScoreAggregationCalculatorTestStruct>;

const std::vector<SnapKVScoreAggregationCalculatorTestStruct> SNAPKV_SCORE_AGGREGATOR_TEST_CASES = {
    {
        "snapkv_window_not_reached",
        /* snapkv_window_size = */ 8, /* prompt_len = */ 100, /* num_processed_tokens = */ 30, /* num_scheduled_tokens = */ 45,
        /* ref_num_scores_aggregated = */ 0
    },
    {
        "snapkv_window_begins",
        /* snapkv_window_size = */ 16, /* prompt_len = */ 30, /* num_processed_tokens = */ 13, /* num_scheduled_tokens = */ 4,
        /* ref_num_scores_aggregated = */ 3
    },
    {
        "snapkv_window_in_progress",
        /* snapkv_window_size = */ 16, /* prompt_len = */ 30, /* num_processed_tokens = */ 16, /* num_scheduled_tokens = */ 8,
        /* ref_num_scores_aggregated = */ 8
    },
    {
        "snapkv_window_ends",
        /* snapkv_window_size = */ 16, /* prompt_len = */ 30, /* num_processed_tokens = */ 28, /* num_scheduled_tokens = */ 2,
        /* ref_num_scores_aggregated = */ 2
    },
    {
        "generation_phase",
        /* snapkv_window_size = */ 16, /* prompt_len = */ 30, /* num_processed_tokens = */ 33, /* num_scheduled_tokens = */ 1,
        /* ref_num_scores_aggregated = */ 1
    },
    {
        "snapkv_window_in_one_chunk",
        /* snapkv_window_size = */ 6, /* prompt_len = */ 42, /* num_processed_tokens = */ 33, /* num_scheduled_tokens = */ 9,
        /* ref_num_scores_aggregated = */ 6
    },
    {
        "no_snapkv_prefill_phase",
        /* snapkv_window_size = */ 0, /* prompt_len = */ 89, /* num_processed_tokens = */ 61, /* num_scheduled_tokens = */ 13,
        /* ref_num_scores_aggregated = */ 13
    },
    {
        "no_snapkv_prefill_end",
        /* snapkv_window_size = */ 0, /* prompt_len = */ 89, /* num_processed_tokens = */ 83, /* num_scheduled_tokens = */ 6,
        /* ref_num_scores_aggregated = */ 6
    },
    {
        "no_snapkv_generation_phase",
        /* snapkv_window_size = */ 0, /* prompt_len = */ 89, /* num_processed_tokens = */ 101, /* num_scheduled_tokens = */ 1,
        /* ref_num_scores_aggregated = */ 1
    },
};

TEST_P(SnapKVScoreAggregationCalculatorParameterizedTest, AggregatesCorrectNumberOfScores) {
    const auto& test_struct = GetParam();
    auto calc = ov::genai::SnapKVScoreAggregationCalculator(test_struct.snapkv_window_size);
    size_t test_num_scores_aggregated = calc.get_num_token_scores_to_aggregate(test_struct.prompt_len, test_struct.num_scheduled_tokens, test_struct.num_processed_tokens);
    EXPECT_EQ(test_num_scores_aggregated, test_struct.ref_num_scores_aggregated);
}

INSTANTIATE_TEST_SUITE_P(VariousInputs, SnapKVScoreAggregationCalculatorParameterizedTest, ::testing::ValuesIn(SNAPKV_SCORE_AGGREGATOR_TEST_CASES),
                         [](const testing::TestParamInfo<SnapKVScoreAggregationCalculatorParameterizedTest::ParamType>& info) {
                             return info.param.test_id;
                         });

class DefaultCacheEvictionAlgoTest : public testing::Test {
protected:
    DefaultCacheEvictionAlgoTest() {
        algo = ov::genai::CacheEvictionAlgorithm(eviction_config, block_size, num_decoder_layers, /* max_pool_window_size = */ 1);
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
    auto algo = ov::genai::CacheEvictionAlgorithm(DEFAULT_CACHE_EVICTION_CONFIG, DEFAULT_BLOCK_SIZE, ref_num_layers, /* max_pool_window_size = */ 1);
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
        //multiple blocks in evictable area - with KVCrush
        {
                "three_blocks_kvcrush",
                2 * 4 + 2,  // 2 blocks worth of overflow + 2 tokens, amounting to 3 blocks to be evicted
                KVCRUSH_CACHE_EVICTION_CONFIG,
                {{28, 10, 11}, {18, 8, 31}},
                {{28, 10, 11}, {18, 8, 31}}
        },
        //if there are more blocks with same low score than should be evicted, the lower-indexed ones should take precedence - with KVCrush
        {
                "four_zeroed_two_to_evict_kvcrush",
                1 * 4 + 2,  // 2 blocks to be evicted
                KVCRUSH_CACHE_EVICTION_CONFIG,
                {{15, 36, 13, 10}, {9, 39, 31, 11}},  // 4 zeroed blocks
                {{10, 13}, {9, 11}}
        },
        //will prefer to evict lower-indexed blocks if there are multiple same-scored blocks - with KVCrush
        {
                "less_zeroed_than_to_evict_kvcrush",
                5 * 4 + 2,  // 6 blocks to be evicted
                KVCRUSH_CACHE_EVICTION_CONFIG,
                {{}, {30, 22}},  // 1st layer has no zeroed blocks, 2nd has only 2 zeroed blocks
                {{8, 9, 10, 11, 12, 13}, {8, 9, 10, 11, 22, 30}}  // non-zeroed blocks to evict are taken from the beginning of evictable range
        },

        //low-scored blocks in non-evictable range do not lead to eviction - with KVCrush
        {
                "zeros_also_in_non_evictable_areas_kvcrush",
                5 * 4 + 2,  // 6 blocks to be evicted
                KVCRUSH_CACHE_EVICTION_CONFIG,
                {{0, 2, 7, 24, 31, 49}, {5, 19, 27, 39, 50, 52}},  // 1st layer has 0, 2, 7 in start_area, 49 in recent_area; 2nd has 5 in start_area, 50, 54 in recent_area
                {{8, 9, 10, 11, 24, 31}, {8, 9, 10, 19, 27, 39}}   // eviction padded up to 6 blocks by blocks in the beginning of the evictable_area
        }
};
// clang-format on

TEST_P(CacheEvictionLowScoreBlocksParameterizedTest, EvictsLowestScoredBlocks) {
    auto test_struct = GetParam();
    size_t num_decoder_layers = DEFAULT_NUM_DECODER_LAYERS;
    auto algo = ov::genai::CacheEvictionAlgorithm(test_struct.eviction_config, DEFAULT_BLOCK_SIZE, num_decoder_layers, /* max_pool_window_size = */ 1);
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
        bool should_apply_kvcrush =
            (test_struct.eviction_config.kvcrush_config.budget > 0) &&
            (std::ceil(static_cast<double>(test_struct.tokens_over_max_cache_size) / DEFAULT_BLOCK_SIZE) >=
             (test_struct.eviction_config.kvcrush_config.budget));
        if (should_apply_kvcrush) {
            // check test_evicted_blocks is a subset of ref_evicted_blocks
            EXPECT_TRUE(std::includes(ref_evicted_blocks[layer_idx].begin(),
                                      ref_evicted_blocks[layer_idx].end(),
                                      test_evicted_blocks[layer_idx].begin(),
                                      test_evicted_blocks[layer_idx].end()));
            EXPECT_EQ(static_cast<int>(test_evicted_blocks[layer_idx].size()),
                      static_cast<int>(ref_evicted_blocks[layer_idx].size()) -
                          static_cast<int>(test_struct.eviction_config.kvcrush_config.budget));
        }
        // if kvcrush is disabled
        else {
            EXPECT_EQ(test_evicted_blocks[layer_idx], ref_evicted_blocks[layer_idx]);
        }
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
    auto algo = ov::genai::CacheEvictionAlgorithm(config, DEFAULT_BLOCK_SIZE, NUM_DECODER_LAYERS, /* max_pool_window_size = */ 1);
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
    auto algo = ov::genai::CacheEvictionAlgorithm(config, DEFAULT_BLOCK_SIZE, NUM_DECODER_LAYERS, /* max_pool_window_size = */ 1);

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
    size_t snapkv_window_size;
    size_t kvcrush_budget;
};

using CacheEvictionConfigInitializationTest = ::testing::TestWithParam<CacheEvictionConfigInitParamsForTest>;

const std::vector<CacheEvictionConfigInitParamsForTest> INVALID_CONFIG_INIT_PARAMS_CASES = {
        // zero area sizes
        {32, 32, 64, 8, 10},
        {0, 13, 39, 1, 10},
        {128, 0, 384, 7, 10},

        // max_cache_size less than start_size + recent_size
        {32, 64, 32, 8, 10},
};

TEST_P(CacheEvictionConfigInitializationTest, ThrowsForInvalidConfigParams) {
    auto params = GetParam();
    EXPECT_THROW(ov::genai::CacheEvictionConfig(params.start_size, params.recent_size, params.max_cache_size, ov::genai::AggregationMode::NORM_SUM, /* apply_rotation = */ false, params.snapkv_window_size, KVCrushConfig(2, KVCrushAnchorPointMode::MEAN)), ov::Exception);
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
        { {32, 32, 97, ov::genai::AggregationMode::SUM, false, 8, KVCrushConfig(2, KVCrushAnchorPointMode::MEAN)}, 16, 8},
        { {11, 13, 50, ov::genai::AggregationMode::NORM_SUM, false, 8, KVCrushConfig(2, KVCrushAnchorPointMode::MEAN)}, 13, 1},
        { {128, 200, 584, ov::genai::AggregationMode::NORM_SUM, false, 8, KVCrushConfig(2, KVCrushAnchorPointMode::MEAN)}, 128, 19},

        // zero decoder layers
        { {32, 64, 192, ov::genai::AggregationMode::SUM, false, 8, KVCrushConfig(2, KVCrushAnchorPointMode::MEAN)}, 32, 0},
};
TEST_P(CacheEvictionAlgoInitializationTest, ThrowsForInvalidConfigs) {
    auto params = GetParam();
    EXPECT_THROW(ov::genai::CacheEvictionAlgorithm(params.config, params.block_size, params.num_decoder_layers, /* max_pool_window_size = */ 1), ov::Exception);
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
