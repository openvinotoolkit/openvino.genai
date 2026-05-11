// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <limits>

#include "continuous_batching/scheduler.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/runtime/core.hpp"
#include "sequence_group.hpp"
#include "utils.hpp"

namespace {

ov::genai::SequenceGroup::Ptr create_sequence_group(uint64_t request_id = 0) {
    std::vector<int64_t> tokens = {0, 1, 2, 3};
    return std::make_shared<ov::genai::SequenceGroup>(
        request_id,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        ov::genai::utils::get_beam_search_config());
}

ov::genai::SequenceGroup::Ptr create_sequence_group(const std::vector<int64_t>& tokens, uint64_t request_id) {
    return std::make_shared<ov::genai::SequenceGroup>(
        request_id,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        ov::genai::utils::get_greedy_config());
}

void ensure_two_running_sequences(const ov::genai::SequenceGroup::Ptr& sequence_group) {
    auto parent = sequence_group->get_running_sequences().at(0);
    sequence_group->fork_sequence(parent);
    ASSERT_EQ(sequence_group->num_running_seqs(), 2);
}

}  // namespace

TEST(TestBlockManager, general_test) {
    ov::genai::BlockManager bm = ov::genai::BlockManager(6, false, 4);
    ov::genai::TokenIds prompt_ids = {10, 0};

    ov::genai::SequenceGroup::Ptr sequence_group =
        std::make_shared<ov::genai::SequenceGroup>(0,
                                                   ov::Tensor(ov::element::i64, {prompt_ids.size()}, prompt_ids.data()),
                                                   ov::genai::utils::get_beam_search_config());
    auto sequence = sequence_group->get_not_finished_sequences()[0];
    bm.allocate_tokens(sequence, sequence_group, 24, prompt_ids.size());
    auto seq_id = sequence->get_id();
    EXPECT_TRUE(bm.has_block_table(seq_id));
    EXPECT_EQ(bm.get_block_table(seq_id, 0).size(), 6);
    EXPECT_EQ(bm.num_free_blocks(), 0);

    bm.free_sequence_partially(seq_id, 4);
    EXPECT_EQ(bm.get_block_table(seq_id, 0).size(), 2);
    EXPECT_EQ(bm.num_free_blocks(), 4);

    bm.free_sequence(seq_id);
    EXPECT_FALSE(bm.has_block_table(seq_id));
    EXPECT_EQ(bm.num_free_blocks(), 6);

    bm.allocate_tokens(sequence, sequence_group, 8, prompt_ids.size());
    bm.fork_sequence(seq_id, 1);
    EXPECT_TRUE(bm.has_block_table(1));
    EXPECT_EQ(bm.get_block_table(1, 0).back()->get_references_count(), 2);
    bm.free_sequence(0);
    bm.free_sequence(1);
}

TEST(TestBlockManager, required_blocks_count) {
    ov::genai::BlockManager bm = ov::genai::BlockManager(8, false, 4, 3);

    std::vector<int64_t> tokens = {0, 1, 2, 3, 4};
    ov::genai::SequenceGroup::Ptr sequence_group =
        std::make_shared<ov::genai::SequenceGroup>(0,
                                                   ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                   ov::genai::utils::get_beam_search_config());
    sequence_group->schedule_tokens(5);
    auto required_blocks = bm.required_blocks_count(sequence_group);
    EXPECT_EQ(required_blocks, 2);
    EXPECT_TRUE(bm.can_append_slots(sequence_group));
    bm.append_slots(sequence_group);
    EXPECT_EQ(bm.num_free_blocks(), 6);
    EXPECT_EQ(bm.get_number_of_blocks_occupied_by_sequence(sequence_group), 2);

    sequence_group->finish_iteration();
    auto sequence_to_fork = sequence_group->get_running_sequences()[0];
    for (size_t i = 0; i < 4; ++i) {
        const auto forked_sequence = sequence_group->fork_sequence(sequence_to_fork);
        bm.fork_sequence(sequence_to_fork->get_id(), forked_sequence->get_id());
    }
    EXPECT_EQ(bm.get_number_of_blocks_occupied_by_sequence(sequence_group), 2);
    sequence_group->schedule_tokens(1);
    required_blocks = bm.required_blocks_count(sequence_group);
    // The last block was incomplete before forking, therefore need to allocate an extra block for each new forked
    // sequence (excluding the original)
    EXPECT_EQ(required_blocks, 4);
    EXPECT_TRUE(bm.can_append_slots(sequence_group));
    bm.append_slots(sequence_group);
    EXPECT_EQ(bm.get_number_of_blocks_occupied_by_sequence(sequence_group), 6);
    EXPECT_EQ(bm.num_free_blocks(), 2);
    sequence_group->finish_iteration();

    sequence_group->schedule_tokens(3);
    required_blocks = bm.required_blocks_count(sequence_group);
    // Each sequence in group had 3 tokens scheduled in addition to 6 already processed, e.g. with block size 4 we
    // require 1 extra block for each sequence in group
    EXPECT_EQ(required_blocks, 5);
    EXPECT_FALSE(bm.can_append_slots(sequence_group));

    for (auto& sequence : sequence_group->get_sequences()) {
        bm.free_sequence(sequence->get_id());
    }
}

TEST(TestBlockManager, CanFreeBlocksFromSequence) {
    const size_t BLOCK_SIZE = 2;
    ov::genai::BlockManager bm = ov::genai::BlockManager(8, false, BLOCK_SIZE, 3);

    std::vector<int64_t> tokens = {0, 1, 2, 3, 4};
    ov::genai::SequenceGroup::Ptr sequence_group =
        std::make_shared<ov::genai::SequenceGroup>(0,
                                                   ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                   ov::genai::utils::get_beam_search_config());
    sequence_group->schedule_tokens(5);
    bm.append_slots(sequence_group);
    ASSERT_EQ(bm.num_free_blocks(), 5);

    size_t seq_id = sequence_group->get_sequences()[0]->get_id();
    bm.free_blocks_from_sequence(seq_id, {{0}, {1}, {2}});
    EXPECT_EQ(bm.num_free_blocks(), 6);

    for (auto& sequence : sequence_group->get_sequences()) {
        bm.free_sequence(sequence->get_id());
    }
}

// Linear Attention with fixed-size blocks tests

TEST(TestBlockManager, FixedSizeCanAllocateCumulativeDeficitFails) {
    const size_t fixed_blocks_per_sequence = 2;
    ov::genai::BlockManager bm = ov::genai::BlockManager(
        /*num_blocks=*/2,
        /*enable_prefix_caching=*/false,
        /*block_size=*/1,
        /*num_layers=*/1,
        fixed_blocks_per_sequence);

    auto sequence_group = create_sequence_group(10);
    ensure_two_running_sequences(sequence_group);

    // Two running sequences each need 2 blocks, while pool has only 2 in total.
    EXPECT_FALSE(bm.can_allocate_tokens(sequence_group, /*num_tokens=*/1));
}

TEST(TestBlockManager, FixedSizeCanAllocateExactCumulativeFitPasses) {
    const size_t fixed_blocks_per_sequence = 1;
    ov::genai::BlockManager bm = ov::genai::BlockManager(
        /*num_blocks=*/2,
        /*enable_prefix_caching=*/false,
        /*block_size=*/1,
        /*num_layers=*/1,
        fixed_blocks_per_sequence);

    auto sequence_group = create_sequence_group(11);
    ensure_two_running_sequences(sequence_group);

    // Two running sequences each need 1 block, and pool has exactly 2.
    EXPECT_TRUE(bm.can_allocate_tokens(sequence_group, /*num_tokens=*/1));
}

TEST(TestBlockManager, FixedSizeAllocateTokensOnlyForMissingSequence) {
    const size_t fixed_blocks_per_sequence = 2;
    ov::genai::BlockManager bm = ov::genai::BlockManager(
        /*num_blocks=*/4,
        /*enable_prefix_caching=*/false,
        /*block_size=*/1,
        /*num_layers=*/1,
        fixed_blocks_per_sequence);

    auto sequence_group = create_sequence_group(12);
    ensure_two_running_sequences(sequence_group);

    auto running = sequence_group->get_running_sequences();
    auto parent = running.at(0);
    auto child = running.at(1);

    bm.allocate_tokens(parent, sequence_group, /*num_tokens=*/1, sequence_group->get_prompt_len());
    ASSERT_TRUE(bm.has_block_table(parent->get_id()));
    EXPECT_EQ(bm.get_block_table(parent->get_id(), 0).size(), fixed_blocks_per_sequence);
    EXPECT_EQ(bm.num_free_blocks(), 2);

    bm.allocate_tokens(child, sequence_group, /*num_tokens=*/1, sequence_group->get_prompt_len());
    ASSERT_TRUE(bm.has_block_table(child->get_id()));
    EXPECT_EQ(bm.get_block_table(child->get_id(), 0).size(), fixed_blocks_per_sequence);
    // Parent must remain unchanged and no extra blocks should be consumed for it.
    EXPECT_EQ(bm.get_block_table(parent->get_id(), 0).size(), fixed_blocks_per_sequence);
    EXPECT_EQ(bm.num_free_blocks(), 0);

    bm.free_sequence(parent->get_id());
    bm.free_sequence(child->get_id());
}

TEST(TestBlockManager, FixedSizeAvailableTokenSlotsBeforeAndAfterAllocation) {
    const size_t fixed_blocks_per_sequence = 2;

    // Before allocation: one sequence already has fixed blocks, one does not, and
    // there are not enough blocks left to satisfy the missing sequence.
    ov::genai::BlockManager bm_before = ov::genai::BlockManager(
        /*num_blocks=*/3,
        /*enable_prefix_caching=*/false,
        /*block_size=*/1,
        /*num_layers=*/1,
        fixed_blocks_per_sequence);

    auto sequence_group_before = create_sequence_group(13);
    ensure_two_running_sequences(sequence_group_before);

    auto running_before = sequence_group_before->get_running_sequences();
    bm_before.allocate_tokens(running_before.at(0), sequence_group_before, /*num_tokens=*/1, sequence_group_before->get_prompt_len());
    EXPECT_EQ(bm_before.num_free_blocks(), 1);
    EXPECT_EQ(bm_before.available_token_slots(sequence_group_before), 0);

    bm_before.free_sequence(running_before.at(0)->get_id());

    // After allocation: all running sequences have fixed blocks, so slots are effectively unlimited.
    ov::genai::BlockManager bm_after = ov::genai::BlockManager(
        /*num_blocks=*/4,
        /*enable_prefix_caching=*/false,
        /*block_size=*/1,
        /*num_layers=*/1,
        fixed_blocks_per_sequence);

    auto sequence_group_after = create_sequence_group(14);
    ensure_two_running_sequences(sequence_group_after);
    auto running_after = sequence_group_after->get_running_sequences();

    bm_after.allocate_tokens(running_after.at(0), sequence_group_after, /*num_tokens=*/1, sequence_group_after->get_prompt_len());
    bm_after.allocate_tokens(running_after.at(1), sequence_group_after, /*num_tokens=*/1, sequence_group_after->get_prompt_len());

    EXPECT_EQ(bm_after.available_token_slots(sequence_group_after), std::numeric_limits<size_t>::max());

    bm_after.free_sequence(running_after.at(0)->get_id());
    bm_after.free_sequence(running_after.at(1)->get_id());
}

TEST(TestBlockManager, FixedSizeFreeSequenceReleasesCapacityForNextSequence) {
    const size_t fixed_blocks_per_sequence = 2;
    ov::genai::BlockManager bm = ov::genai::BlockManager(
        /*num_blocks=*/2,
        /*enable_prefix_caching=*/false,
        /*block_size=*/1,
        /*num_layers=*/1,
        fixed_blocks_per_sequence);

    auto first_group = create_sequence_group(15);
    auto first_seq = first_group->get_running_sequences().at(0);

    bm.allocate_tokens(first_seq, first_group, /*num_tokens=*/1, first_group->get_prompt_len());
    EXPECT_EQ(bm.num_free_blocks(), 0);

    bm.free_sequence(first_seq->get_id());
    EXPECT_EQ(bm.num_free_blocks(), 2);

    auto second_group = create_sequence_group(16);
    auto second_seq = second_group->get_running_sequences().at(0);
    EXPECT_TRUE(bm.can_allocate_tokens(second_group, /*num_tokens=*/1));

    bm.allocate_tokens(second_seq, second_group, /*num_tokens=*/1, second_group->get_prompt_len());
    EXPECT_TRUE(bm.has_block_table(second_seq->get_id()));
    EXPECT_EQ(bm.get_block_table(second_seq->get_id(), 0).size(), fixed_blocks_per_sequence);

    bm.free_sequence(second_seq->get_id());
}

TEST(TestBlockManager, PrefixCachingCompleteCheckpointReuseAllocatesOwnedWriteBlocks) {
    constexpr size_t block_size = 4;
    ov::genai::BlockManager block_manager(
        /*num_blocks=*/8,
        /*enable_prefix_caching=*/true,
        block_size,
        /*num_layers=*/1);

    std::vector<int64_t> tokens = {0, 1, 2, 3};
    auto producer_group = create_sequence_group(tokens, 20);
    producer_group->schedule_tokens(tokens.size());
    block_manager.append_slots(producer_group);
    producer_group->finish_iteration();

    const auto producer_seq_id = producer_group->get_running_sequences().at(0)->get_id();
    const auto checkpoint_block_idx = block_manager.get_block_table(producer_seq_id, 0).at(0)->get_index();
    block_manager.free_sequence(producer_seq_id);

    auto first_consumer_group = create_sequence_group(tokens, 21);
    auto second_consumer_group = create_sequence_group(tokens, 22);
    block_manager.restore_cached_blocks(first_consumer_group);
    block_manager.restore_cached_blocks(second_consumer_group);

    const auto first_seq_id = first_consumer_group->get_running_sequences().at(0)->get_id();
    const auto second_seq_id = second_consumer_group->get_running_sequences().at(0)->get_id();
    ASSERT_EQ(block_manager.get_block_table(first_seq_id, 0).size(), 1);
    ASSERT_EQ(block_manager.get_block_table(second_seq_id, 0).size(), 1);
    EXPECT_EQ(block_manager.get_block_table(first_seq_id, 0).at(0)->get_index(), checkpoint_block_idx);
    EXPECT_EQ(block_manager.get_block_table(second_seq_id, 0).at(0)->get_index(), checkpoint_block_idx);
    EXPECT_EQ(block_manager.get_block_table(first_seq_id, 0).at(0)->get_references_count(), 2);

    // Linear-attention complete checkpoints can be shared as read-only inputs; continuation writes
    // must allocate request-owned blocks instead of overwriting the shared checkpoint.
    first_consumer_group->update_processed_tokens_num(tokens.size());
    second_consumer_group->update_processed_tokens_num(tokens.size());
    first_consumer_group->schedule_tokens(1);
    second_consumer_group->schedule_tokens(1);

    const auto first_copy_map = block_manager.append_slots(first_consumer_group);
    const auto second_copy_map = block_manager.append_slots(second_consumer_group);

    ASSERT_EQ(block_manager.get_block_table(first_seq_id, 0).size(), 2);
    ASSERT_EQ(block_manager.get_block_table(second_seq_id, 0).size(), 2);
    EXPECT_TRUE(first_copy_map.empty());
    EXPECT_TRUE(second_copy_map.empty());
    EXPECT_EQ(block_manager.get_block_table(first_seq_id, 0).at(0)->get_index(), checkpoint_block_idx);
    EXPECT_EQ(block_manager.get_block_table(second_seq_id, 0).at(0)->get_index(), checkpoint_block_idx);
    EXPECT_NE(block_manager.get_block_table(first_seq_id, 0).at(1)->get_index(), checkpoint_block_idx);
    EXPECT_NE(block_manager.get_block_table(second_seq_id, 0).at(1)->get_index(), checkpoint_block_idx);
    EXPECT_NE(block_manager.get_block_table(first_seq_id, 0).at(1)->get_index(),
              block_manager.get_block_table(second_seq_id, 0).at(1)->get_index());

    block_manager.free_sequence(first_seq_id);
    block_manager.free_sequence(second_seq_id);
}

TEST(TestBlockManager, SequenceHashRejectsZeroContentLength) {
    auto sequence_group = create_sequence_group();
    auto sequence = sequence_group->get_running_sequences().at(0);

    EXPECT_THROW(sequence->get_hash(0, 4), ov::Exception);
}

TEST(TestBlockManager, PrefixCachingIncompleteCheckpointUsesCopyOnWritePerSequence) {
    constexpr size_t block_size = 4;
    ov::genai::BlockManager block_manager(
        /*num_blocks=*/8,
        /*enable_prefix_caching=*/true,
        block_size,
        /*num_layers=*/1);

    std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5};
    auto producer_group = create_sequence_group(tokens, 23);
    producer_group->schedule_tokens(tokens.size());
    block_manager.append_slots(producer_group);
    producer_group->finish_iteration();

    const auto producer_seq_id = producer_group->get_running_sequences().at(0)->get_id();
    const auto complete_checkpoint_idx = block_manager.get_block_table(producer_seq_id, 0).at(0)->get_index();
    const auto incomplete_checkpoint_idx = block_manager.get_block_table(producer_seq_id, 0).at(1)->get_index();
    block_manager.free_sequence(producer_seq_id);

    auto first_consumer_group = create_sequence_group(tokens, 24);
    auto second_consumer_group = create_sequence_group(tokens, 25);
    block_manager.restore_cached_blocks(first_consumer_group);
    block_manager.restore_cached_blocks(second_consumer_group);

    const auto first_seq_id = first_consumer_group->get_running_sequences().at(0)->get_id();
    const auto second_seq_id = second_consumer_group->get_running_sequences().at(0)->get_id();
    ASSERT_EQ(block_manager.get_block_table(first_seq_id, 0).size(), 2);
    ASSERT_EQ(block_manager.get_block_table(second_seq_id, 0).size(), 2);
    EXPECT_EQ(block_manager.get_block_table(first_seq_id, 0).at(0)->get_index(), complete_checkpoint_idx);
    EXPECT_EQ(block_manager.get_block_table(second_seq_id, 0).at(0)->get_index(), complete_checkpoint_idx);
    EXPECT_EQ(block_manager.get_block_table(first_seq_id, 0).at(1)->get_index(), incomplete_checkpoint_idx);
    EXPECT_EQ(block_manager.get_block_table(second_seq_id, 0).at(1)->get_index(), incomplete_checkpoint_idx);

    // Both consumers resume inside the same incomplete interval, so the shared mutable checkpoint
    // must be split through copy-on-write before either sequence writes its next state.
    first_consumer_group->schedule_tokens(1);
    second_consumer_group->schedule_tokens(1);

    const auto first_copy_map = block_manager.append_slots(first_consumer_group);
    const auto second_copy_map = block_manager.append_slots(second_consumer_group);

    ASSERT_EQ(block_manager.get_block_table(first_seq_id, 0).size(), 2);
    ASSERT_EQ(block_manager.get_block_table(second_seq_id, 0).size(), 2);
    EXPECT_TRUE(first_copy_map.count(incomplete_checkpoint_idx));
    EXPECT_TRUE(second_copy_map.empty());
    EXPECT_EQ(block_manager.get_block_table(first_seq_id, 0).at(0)->get_index(), complete_checkpoint_idx);
    EXPECT_EQ(block_manager.get_block_table(second_seq_id, 0).at(0)->get_index(), complete_checkpoint_idx);
    EXPECT_NE(block_manager.get_block_table(first_seq_id, 0).at(1)->get_index(), incomplete_checkpoint_idx);
    EXPECT_EQ(block_manager.get_block_table(second_seq_id, 0).at(1)->get_index(), incomplete_checkpoint_idx);
    EXPECT_NE(block_manager.get_block_table(first_seq_id, 0).at(1)->get_index(),
              block_manager.get_block_table(second_seq_id, 0).at(1)->get_index());

    block_manager.free_sequence(first_seq_id);
    block_manager.free_sequence(second_seq_id);
}
