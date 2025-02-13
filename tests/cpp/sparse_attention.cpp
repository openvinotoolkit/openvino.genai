#include <gtest/gtest.h>
#include "continuous_batching/sparse_attention.hpp"

using namespace ov::genai;

constexpr size_t BLOCK_SIZE = 32;

struct SparseAttentionBlockSkipperTestStruct {
    size_t prompt_len_in_tokens;
    size_t num_tokens_currently_in_cache;
    size_t num_tokens_in_new_chunk;
    size_t num_last_dense_tokens;
    std::set<size_t> ref_skipped_logical_blocks;
    std::string test_id;
};

using SparseAttentionBlockSkipperTest = ::testing::TestWithParam<SparseAttentionBlockSkipperTestStruct>;

const std::vector<SparseAttentionBlockSkipperTestStruct> SKIPPER_TEST_CASES = {
    { 10, 5, 2, 1, {}, "one_block_prompt_sparse_prefill_phase"},
    { 10, 4, 2, 5, {}, "one_block_prompt_sparse_to_dense_prefill_phase"},
    { 10, 7, 2, 5, {}, "one_block_prompt_dense_prefill_phase"},
    { 10, 5, 2, 5, {}, "one_block_prompt_dense_prefill_phase_start"},
    { 10, 7, 3, 5, {}, "one_block_prompt_dense_prefill_phase_end"},

    { 40, 15, 5, 9, {}, "two_block_prompt_sparse_prefill_phase"},
    { 40, 2, 1, 35, {}, "two_block_prompt_sparse_prefill_phase_multi_block_dense_phase"},
    { 40, 33, 4, 5, {}, "two_block_prompt_sparse_to_dense_prefill_phase_in_same_block"},
    { 40, 20, 17, 5, {}, "two_block_prompt_sparse_to_dense_prefill_phase_crossing_block_borders"},
    { 40, 35, 2, 5, {}, "two_block_prompt_dense_prefill_phase_start_in_same_block"},
    { 40, 25, 10, 15, {}, "two_block_prompt_dense_prefill_phase_start_crossing_block_borders"},
    { 40, 35, 2, 5, {}, "two_block_prompt_dense_prefill_phase_in_same_block"},
    { 40, 30, 8, 15, {}, "two_block_prompt_dense_prefill_phase_crossing_block_borders"},
    { 40, 36, 4, 10, {}, "two_block_prompt_dense_prefill_phase_end"},

    { 158, 15, 5, 1, {}, "five_block_prompt_sparse_prefill_phase_begin"},
    { 158, 15, 5, 100, {}, "five_block_prompt_sparse_prefill_phase_begin_multi_block_dense_phase"},
    { 158, 60, 3, 70, {}, "five_block_prompt_sparse_prefill_phase_multi_block_begin_multi_block_dense_phase"},
    { 158, 68, 5, 15, {}, "five_block_prompt_sparse_prefill_phase_early"},
    { 158, 68, 5, 70, {}, "five_block_prompt_sparse_prefill_phase_early_multi_block_dense_phase"},
    { 158, 68, 35, 40, {}, "five_block_prompt_sparse_prefill_phase_early_multi_block_chunk_multi_block_dense_phase"},
    { 158, 98, 5, 15, {1}, "five_block_prompt_sparse_prefill_phase_with_skips"},
    { 190, 98, 5, 60, {1}, "six_block_prompt_sparse_prefill_phase_with_skips_multi_block_dense_phase"},
    { 158, 98, 35, 10, {1}, "five_block_prompt_sparse_prefill_phase_with_skips_multi_block_chunk"},
    { 190, 98, 35, 40, {1}, "six_block_prompt_sparse_prefill_phase_with_skips_multi_block_chunk_multi_block_dense_phase"},
    { 158, 129, 16, 10, {1, 2}, "five_block_prompt_sparse_prefill_phase_with_skips_nearing_dense"},

    // even though we end on exact block border, the last block containing already cached tokens
    // was not complete, and we should not skip it
    { 158, 124, 4, 10, {1}, "five_block_prompt_sparse_prefill_phase_with_skips_exact_block_border"},
    { 190, 151, 9, 10, {1, 2}, "six_block_prompt_sparse_prefill_phase_with_skips_exact_block_border"},
    // same even if we end up slightly over block border - it's the already cached state that matters
    { 190, 151, 11, 10, {1, 2}, "six_block_prompt_sparse_prefill_phase_with_skips_slightly_over_block_border"},

    { 300, 250, 11, 10, {1, 2, 3, 4, 5}, "ten_block_prompt_sparse_prefill_phase_with_skips"},
    { 158, 100, 5, 50, {1}, "five_block_prompt_sparse_prefill_phase_with_multi_block_dense_starting_in_same_block"},
    { 158, 140, 4, 14, {1, 2}, "five_block_prompt_sparse_prefill_phase_end_in_same_block"},
    { 158, 110, 34, 14, {1}, "five_block_prompt_sparse_prefill_phase_end_crossing_block_borders"},
    { 158, 140, 10, 12, {}, "five_block_prompt_sparse_to_dense_prefill_phase_in_same_block"},
    { 158, 20, 137, 5, {}, "five_block_prompt_sparse_to_dense_prefill_phase_crossing_block_borders"},
    { 158, 45, 2, 113, {}, "five_block_prompt_dense_prefill_phase_start_in_same_block"},
    { 158, 45, 40, 113, {}, "five_block_prompt_dense_prefill_phase_start_crossing_block_borders"},
    { 158, 55, 8, 120, {}, "five_block_prompt_dense_prefill_phase_in_same_block"},
    { 158, 55, 80, 120, {}, "five_block_prompt_dense_prefill_phase_crossing_block_borders"},
    { 158, 55, 103, 120, {}, "five_block_prompt_dense_prefill_phase_end"}
};

class SparseAttentionBlockSkipperReferenceTest : public ::testing::Test, public ::testing::WithParamInterface<SparseAttentionBlockSkipperTestStruct> {
protected:
    SparseAttentionBlockSkipperReferenceTest() {
       auto test_struct = GetParam();
       auto mock_token_ids = TokenIds(test_struct.prompt_len_in_tokens, 0l);
       auto mock_sampling_params = GenerationConfig{};

       // prepare the required state of sequence group
       sequence_group = std::make_shared<SequenceGroup>(0, mock_token_ids, mock_sampling_params, BLOCK_SIZE);
       sequence_group->schedule_tokens(test_struct.num_tokens_currently_in_cache);
       sequence_group->finish_iteration();
       sequence_group->schedule_tokens(test_struct.num_tokens_in_new_chunk);
    }
    SequenceGroup::Ptr sequence_group;
};

TEST_P(SparseAttentionBlockSkipperReferenceTest, CorrectBlocksAreSkipped) {
    auto test_struct = GetParam();
    auto skipper = SparseAttentionTokenSkipper(test_struct.num_last_dense_tokens);
    auto test_skipped_blocks = skipper.get_skipped_blocks(sequence_group);
    EXPECT_EQ(test_skipped_blocks, test_struct.ref_skipped_logical_blocks);
}

INSTANTIATE_TEST_SUITE_P(VariousSequenceGroupStates, SparseAttentionBlockSkipperReferenceTest, ::testing::ValuesIn(SKIPPER_TEST_CASES), [](const testing::TestParamInfo<SparseAttentionBlockSkipperReferenceTest::ParamType>& info) {
      // Can use info.param here to generate the test suffix
      return info.param.test_id;
    });
