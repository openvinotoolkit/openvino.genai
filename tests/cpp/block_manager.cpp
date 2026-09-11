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

ov::Tensor create_embeddings_tensor(const std::vector<std::vector<float>>& embeddings) {
    OPENVINO_ASSERT(!embeddings.empty());
    ov::Tensor tensor(ov::element::f32, {1, embeddings.size(), embeddings.front().size()});
    float* data = tensor.data<float>();
    for (const auto& embedding : embeddings) {
        OPENVINO_ASSERT(embedding.size() == embeddings.front().size());
        std::copy(embedding.begin(), embedding.end(), data);
        data += embedding.size();
    }
    return tensor;
}

ov::genai::SequenceGroup::Ptr create_embedding_sequence_group(const std::vector<std::vector<float>>& embeddings,
                                                              uint64_t request_id) {
    return std::make_shared<ov::genai::SequenceGroup>(request_id,
                                                      create_embeddings_tensor(embeddings),
                                                      ov::genai::utils::get_greedy_config());
}

void ensure_two_running_sequences(const ov::genai::SequenceGroup::Ptr& sequence_group) {
    auto parent = sequence_group->get_running_sequences().at(0);
    sequence_group->fork_sequence(parent);
    ASSERT_EQ(sequence_group->num_running_seqs(), 2);
}

class PrefixCachingCopyOnWriteLayerTest : public testing::TestWithParam<size_t> {};

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

TEST(TestBlockManager, CannotFreeLogicalBlockPastTableEnd) {
    constexpr size_t block_size = 2;
    constexpr size_t num_layers = 3;
    ov::genai::BlockManager block_manager = ov::genai::BlockManager(8, false, block_size, num_layers);

    std::vector<int64_t> tokens = {0, 1, 2, 3, 4};
    ov::genai::SequenceGroup::Ptr sequence_group =
        std::make_shared<ov::genai::SequenceGroup>(0,
                                                   ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                   ov::genai::utils::get_beam_search_config());
    sequence_group->schedule_tokens(tokens.size());
    block_manager.append_slots(sequence_group);

    const size_t seq_id = sequence_group->get_sequences()[0]->get_id();
    ASSERT_EQ(block_manager.get_block_table(seq_id, 0).size(), 3);
    EXPECT_THROW(block_manager.free_blocks_from_sequence(seq_id, {{3}, {3}, {3}}), ov::Exception);

    for (auto& sequence : sequence_group->get_sequences()) {
        block_manager.free_sequence(sequence->get_id());
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

TEST(TestBlockManager, TemporaryBlocksPromoteSelectedCheckpoint) {
    ov::genai::BlockManager bm = ov::genai::BlockManager(
        /*num_blocks=*/4,
        /*enable_prefix_caching=*/false,
        /*block_size=*/1,
        /*num_layers=*/2,
        /*fixed_blocks_per_sequence=*/1);

    auto sequence_group = create_sequence_group(17);
    auto sequence = sequence_group->get_running_sequences().at(0);
    bm.allocate_tokens(sequence, sequence_group, /*num_tokens=*/1, sequence_group->get_prompt_len());
    const auto seq_id = sequence->get_id();
    const int committed_block_layer_0 = bm.get_block_table(seq_id, 0).front()->get_index();
    const int committed_block_layer_1 = bm.get_block_table(seq_id, 1).front()->get_index();
    const auto& empty_live_state = bm.get_linear_attention_live_state(seq_id);
    EXPECT_TRUE(empty_live_state.is_empty);
    EXPECT_EQ(empty_live_state.endpoint, 0u);
    ASSERT_EQ(empty_live_state.rows.size(), 2u);
    EXPECT_EQ(empty_live_state.rows[0]->get_index(), committed_block_layer_0);
    EXPECT_EQ(empty_live_state.rows[1]->get_index(), committed_block_layer_1);
    bm.set_linear_attention_live_state(seq_id, 7, empty_live_state.rows);
    const size_t initial_generation = bm.get_linear_attention_live_state(seq_id).generation;
    EXPECT_EQ(bm.get_block_at_logical_position(seq_id, 0, 0)->get_index(), committed_block_layer_0);
    EXPECT_EQ(bm.get_block_at_logical_position(seq_id, 1, 0)->get_index(), committed_block_layer_1);

    const auto checkpoint_blocks = bm.reserve_temporary_blocks(seq_id, /*num_blocks=*/3);
    ASSERT_EQ(checkpoint_blocks.size(), 3);
    EXPECT_EQ(bm.num_free_blocks(), 0);

    bm.promote_temporary_block(seq_id,
                               /*checkpoint_slot=*/2,
                               /*expected_endpoint=*/7,
                               initial_generation);

    EXPECT_EQ(bm.get_block_table(seq_id, 0).front()->get_index(), checkpoint_blocks[1]);
    EXPECT_NE(bm.get_block_table(seq_id, 0).front()->get_index(), committed_block_layer_0);
    EXPECT_NE(bm.get_block_table(seq_id, 1).front()->get_index(), committed_block_layer_1);
    EXPECT_EQ(bm.get_block_table(seq_id, 0).front()->get_index(), checkpoint_blocks[1]);
    EXPECT_EQ(bm.get_block_table(seq_id, 1).front()->get_index(), checkpoint_blocks[1]);
    EXPECT_EQ(bm.num_free_blocks(), 3);
    const auto& committed_live_state = bm.get_linear_attention_live_state(seq_id);
    EXPECT_FALSE(committed_live_state.is_empty);
    EXPECT_EQ(committed_live_state.endpoint, 9u);
    EXPECT_EQ(committed_live_state.generation, initial_generation + 1);
    ASSERT_EQ(committed_live_state.rows.size(), 2u);
    EXPECT_EQ(committed_live_state.rows[0]->get_index(), checkpoint_blocks[1]);
    EXPECT_EQ(committed_live_state.rows[1]->get_index(), checkpoint_blocks[1]);

    bm.free_sequence(seq_id);
    EXPECT_EQ(bm.num_free_blocks(), 4);
}

TEST(TestBlockManager, TemporaryReservationExcludesEvictableCheckpointsFromWritableCapacity) {
    for (const size_t num_layers : {1u, 2u}) {
        SCOPED_TRACE(num_layers);
        ov::genai::BlockManager manager(4, true, 4, num_layers, 0, true, 4);
        auto producer = create_sequence_group({1, 2, 3, 4}, 0);
        auto active = create_sequence_group({5, 6, 7, 8}, 1);
        const uint64_t producer_id = producer->get_sequences().front()->get_id();
        const uint64_t active_id = active->get_sequences().front()->get_id();
        producer->schedule_tokens(4);
        manager.append_slots(producer);
        producer->finish_iteration();
        const auto published = manager.get_block_table(producer_id, 0).front();
        const auto published_hash = published->get_hash();
        manager.free_sequence(producer_id);
        active->schedule_tokens(4);
        manager.append_slots(active);
        active->finish_iteration();
        ASSERT_TRUE(published->has_published_hash());
        ASSERT_EQ(published->get_references_count(), 0u);
        ASSERT_EQ(manager.num_free_blocks(), 3u);

        EXPECT_FALSE(manager.can_reserve_temporary_blocks(active_id, 0));
        EXPECT_FALSE(manager.can_reserve_temporary_blocks(active_id, 3));
        EXPECT_THROW(std::ignore = manager.reserve_temporary_blocks(active_id, 3), ov::Exception);
        EXPECT_FALSE(manager.has_temporary_blocks(active_id));
        EXPECT_EQ(manager.num_free_blocks(), 3u);
        EXPECT_EQ(published->get_hash(), published_hash);

        ASSERT_TRUE(manager.can_reserve_temporary_blocks(active_id, 2));
        const auto scratch = manager.reserve_temporary_blocks(active_id, 2);
        ASSERT_EQ(scratch.size(), 2u);
        EXPECT_EQ(manager.num_free_blocks(), 1u);
        EXPECT_FALSE(manager.can_reserve_temporary_blocks(active_id, 1));
        for (const int row : scratch) {
            EXPECT_NE(row, published->get_index());
        }
        auto consumer = create_sequence_group({1, 2, 3, 4, 9}, 2);
        const uint64_t consumer_id = consumer->get_sequences().front()->get_id();
        manager.restore_cached_blocks(consumer);
        ASSERT_TRUE(manager.has_block_table(consumer_id));
        EXPECT_EQ(manager.get_block_table(consumer_id, 0).front(), published);
        EXPECT_EQ(manager.get_total_block_count(), 4u);
        consumer->schedule_tokens(1);
        EXPECT_FALSE(manager.can_append_slots(consumer));
        EXPECT_EQ(published->get_hash(), published_hash);
        EXPECT_EQ(manager.get_block_table(consumer_id, 0).front(), published);

        manager.release_temporary_blocks(active_id);
        EXPECT_FALSE(manager.has_temporary_blocks(active_id));
        EXPECT_TRUE(manager.can_reserve_temporary_blocks(active_id, 2));
        EXPECT_TRUE(manager.can_append_slots(consumer));
        std::ignore = manager.append_slots(consumer);
        EXPECT_EQ(manager.get_total_block_count(), 4u);
        EXPECT_EQ(published->get_hash(), published_hash);
        manager.free_sequence(consumer_id);
        manager.free_sequence(active_id);
        EXPECT_EQ(manager.num_free_blocks(), 4u);
    }
}

TEST(TestBlockManager, TemporaryBlocksReleaseWithoutPromotion) {
    ov::genai::BlockManager bm = ov::genai::BlockManager(
        /*num_blocks=*/3,
        /*enable_prefix_caching=*/false,
        /*block_size=*/1,
        /*num_layers=*/1,
        /*fixed_blocks_per_sequence=*/1);

    auto sequence_group = create_sequence_group(18);
    auto sequence = sequence_group->get_running_sequences().at(0);
    bm.allocate_tokens(sequence, sequence_group, /*num_tokens=*/1, sequence_group->get_prompt_len());
    const auto seq_id = sequence->get_id();
    const int committed_block = bm.get_block_table(seq_id, 0).front()->get_index();

    const auto checkpoint_blocks = bm.reserve_temporary_blocks(seq_id, /*num_blocks=*/2);
    ASSERT_EQ(checkpoint_blocks.size(), 2);
    EXPECT_EQ(bm.num_free_blocks(), 0);

    bm.release_temporary_blocks(seq_id);

    EXPECT_EQ(bm.get_block_table(seq_id, 0).front()->get_index(), committed_block);
    EXPECT_EQ(bm.num_free_blocks(), 2);

    bm.free_sequence(seq_id);
    EXPECT_EQ(bm.num_free_blocks(), 3);
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
    EXPECT_THROW(sequence->get_hash(1, 0), ov::Exception);
}

TEST(TestBlockManager, SequenceHashMemoizationIsIndependentPerBlockSize) {
    const std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto kv_then_la_group = create_sequence_group(tokens, 1);
    auto la_then_kv_group = create_sequence_group(tokens, 2);
    auto fresh_kv_group = create_sequence_group(tokens, 3);
    auto fresh_la_group = create_sequence_group(tokens, 4);
    auto kv_then_la = kv_then_la_group->get_running_sequences().at(0);
    auto la_then_kv = la_then_kv_group->get_running_sequences().at(0);
    auto fresh_kv = fresh_kv_group->get_running_sequences().at(0);
    auto fresh_la = fresh_la_group->get_running_sequences().at(0);

    const size_t kv_hash = kv_then_la->get_hash(tokens.size(), 4);
    const size_t la_hash = kv_then_la->get_hash(tokens.size(), 3);
    const size_t reverse_la_hash = la_then_kv->get_hash(tokens.size(), 3);
    const size_t reverse_kv_hash = la_then_kv->get_hash(tokens.size(), 4);
    const size_t fresh_kv_hash = fresh_kv->get_hash(tokens.size(), 4);
    const size_t fresh_la_hash = fresh_la->get_hash(tokens.size(), 3);

    EXPECT_EQ(kv_hash, fresh_kv_hash);
    EXPECT_EQ(la_hash, fresh_la_hash);
    EXPECT_EQ(reverse_kv_hash, fresh_kv_hash);
    EXPECT_EQ(reverse_la_hash, fresh_la_hash);
    EXPECT_EQ(kv_hash, reverse_kv_hash);
    EXPECT_EQ(la_hash, reverse_la_hash);
}

TEST(TestBlockManager, SequenceHashRollbackAndForkMatchFreshSequences) {
    const std::vector<int64_t> prompt = {0, 1, 2, 3};
    const std::vector<int64_t> original_suffix = {4, 5, 6, 7, 8, 9, 10, 11};
    const std::vector<int64_t> divergent_suffix = {40, 41, 42, 43, 44};
    auto sequence_group = create_sequence_group(prompt, 5);
    auto sequence = sequence_group->get_running_sequences().at(0);
    for (const int64_t token : original_suffix) {
        sequence->append_token(token, 0.0f);
    }
    const size_t original_kv_hash = sequence->get_hash(12, 4);
    const size_t original_la_hash = sequence->get_hash(12, 3);
    auto forked_sequence = sequence_group->fork_sequence(sequence);

    forked_sequence->remove_last_tokens(5);
    for (const int64_t token : divergent_suffix) {
        forked_sequence->append_token(token, 0.0f);
    }

    std::vector<int64_t> divergent_identity(prompt);
    divergent_identity.insert(divergent_identity.end(), original_suffix.begin(), original_suffix.begin() + 3);
    divergent_identity.insert(divergent_identity.end(), divergent_suffix.begin(), divergent_suffix.end());
    auto fresh_divergent_group = create_sequence_group(divergent_identity, 6);
    auto fresh_original_group = create_sequence_group({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 7);
    auto fresh_divergent = fresh_divergent_group->get_running_sequences().at(0);
    auto fresh_original = fresh_original_group->get_running_sequences().at(0);

    EXPECT_EQ(forked_sequence->get_hash(12, 3), fresh_divergent->get_hash(12, 3));
    EXPECT_EQ(forked_sequence->get_hash(12, 4), fresh_divergent->get_hash(12, 4));
    EXPECT_EQ(sequence->get_hash(12, 4), original_kv_hash);
    EXPECT_EQ(sequence->get_hash(12, 3), original_la_hash);
    EXPECT_EQ(sequence->get_hash(12, 4), fresh_original->get_hash(12, 4));
    EXPECT_EQ(sequence->get_hash(12, 3), fresh_original->get_hash(12, 3));
}

TEST(TestBlockManager, SequenceHashEmbeddingRollbackMatchesFreshSequence) {
    std::vector<std::vector<float>> embeddings;
    for (size_t token = 0; token < 12; ++token) {
        embeddings.push_back({static_cast<float>(token), static_cast<float>(token) + 0.25f});
    }
    auto sequence_group = create_embedding_sequence_group(
        std::vector<std::vector<float>>(embeddings.begin(), embeddings.begin() + 4), 8);
    auto sequence = sequence_group->get_running_sequences().at(0);
    sequence->append_generated_ids_embeds(
        create_embeddings_tensor(std::vector<std::vector<float>>(embeddings.begin() + 4, embeddings.end())));
    std::vector<int64_t> position_ids = {4, 5, 6, 7, 8, 9, 10, 11};
    sequence->append_position_ids(ov::Tensor(ov::element::i64, {1, position_ids.size()}, position_ids.data()));
    for (size_t token = 4; token < embeddings.size(); ++token) {
        sequence->append_token(static_cast<int64_t>(token), 0.0f);
    }
    sequence->get_hash(embeddings.size(), 4);
    sequence->get_hash(embeddings.size(), 3);

    sequence->remove_last_tokens(5);
    std::vector<std::vector<float>> divergent_embeddings = {{40.0f, 40.25f},
                                                             {41.0f, 41.25f},
                                                             {42.0f, 42.25f},
                                                             {43.0f, 43.25f},
                                                             {44.0f, 44.25f}};
    sequence->append_generated_ids_embeds(create_embeddings_tensor(divergent_embeddings));
    std::vector<int64_t> divergent_position_ids = {7, 8, 9, 10, 11};
    sequence->append_position_ids(
        ov::Tensor(ov::element::i64, {1, divergent_position_ids.size()}, divergent_position_ids.data()));
    for (size_t token = 40; token < 45; ++token) {
        sequence->append_token(static_cast<int64_t>(token), 0.0f);
    }

    std::vector<std::vector<float>> fresh_embeddings(embeddings.begin(), embeddings.begin() + 7);
    fresh_embeddings.insert(fresh_embeddings.end(), divergent_embeddings.begin(), divergent_embeddings.end());
    auto fresh_sequence_group = create_embedding_sequence_group(fresh_embeddings, 9);
    auto fresh_sequence = fresh_sequence_group->get_running_sequences().at(0);

    EXPECT_EQ(sequence->get_hash(12, 3), fresh_sequence->get_hash(12, 3));
    EXPECT_EQ(sequence->get_hash(12, 4), fresh_sequence->get_hash(12, 4));
}

TEST(TestBlockManager, PrefixCachingRollbackUnregistersInvalidatedBoundaryBeforeDivergentRewrite) {
    constexpr size_t block_size = 4;
    ov::genai::BlockManager block_manager(8, true, block_size);

    const std::vector<int64_t> prompt = {0, 1, 2, 3};
    const std::vector<int64_t> original_suffix = {4, 5, 6, 7, 8, 9, 10, 11};
    auto sequence_group = create_sequence_group(prompt, 50);
    const auto sequence = sequence_group->get_running_sequences().front();
    for (const int64_t token : original_suffix) {
        sequence->append_token(token, 0.0f);
    }
    sequence_group->schedule_tokens(prompt.size() + original_suffix.size());
    block_manager.append_slots(sequence_group);
    sequence_group->finish_iteration();

    sequence->remove_last_tokens(5);
    sequence_group->update_processed_tokens_num(7);
    block_manager.free_empty_physical_blocks(sequence_group);

    auto stale_restore_group = create_sequence_group(std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7}, 51);
    ASSERT_TRUE(block_manager.restore_cached_blocks(stale_restore_group));
    EXPECT_EQ(stale_restore_group->get_num_processed_tokens(), 4);
    EXPECT_EQ(block_manager.get_block_table(stale_restore_group->get_running_sequences().front()->get_id(), 0).size(),
              1);
    block_manager.free_sequence(stale_restore_group->get_running_sequences().front()->get_id());

    sequence->append_token(40, 0.0f);
    sequence_group->schedule_tokens(1);
    block_manager.append_slots(sequence_group);
    sequence_group->finish_iteration();

    auto unpublished_restore_group = create_sequence_group(std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 40}, 52);
    ASSERT_TRUE(block_manager.restore_cached_blocks(unpublished_restore_group));
    EXPECT_EQ(unpublished_restore_group->get_num_processed_tokens(), 4);
    EXPECT_EQ(block_manager.get_block_table(unpublished_restore_group->get_running_sequences().front()->get_id(), 0)
                  .size(),
              1);
    block_manager.free_sequence(unpublished_restore_group->get_running_sequences().front()->get_id());

    block_manager.publish_completed_block(sequence, 8);
    auto divergent_restore_group = create_sequence_group(std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 40}, 53);
    ASSERT_TRUE(block_manager.restore_cached_blocks(divergent_restore_group));
    EXPECT_EQ(block_manager.get_block_table(divergent_restore_group->get_running_sequences().front()->get_id(), 0)
                  .back()
                  ->get_index(),
              block_manager.get_block_table(sequence->get_id(), 0).back()->get_index());

    block_manager.free_sequence(sequence->get_id());
    block_manager.free_sequence(divergent_restore_group->get_running_sequences().front()->get_id());
}

TEST(TestBlockManager, PrefixCachingRollbackPreservesIdentityBackedBySharedOwner) {
    constexpr size_t block_size = 4;
    ov::genai::BlockManager block_manager(10, true, block_size);

    auto producer_group = create_sequence_group(std::vector<int64_t>{0, 1, 2, 3}, 54);
    const auto producer = producer_group->get_running_sequences().front();
    for (const int64_t token : std::vector<int64_t>{4, 5, 6, 7}) {
        producer->append_token(token, 0.0f);
    }
    producer_group->schedule_tokens(8);
    block_manager.append_slots(producer_group);
    producer_group->finish_iteration();

    auto owner_group = create_sequence_group(std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7}, 55);
    ASSERT_TRUE(block_manager.restore_cached_blocks(owner_group));
    const uint64_t owner_id = owner_group->get_running_sequences().front()->get_id();
    const int original_index = block_manager.get_block_table(owner_id, 0).back()->get_index();
    ASSERT_EQ(block_manager.get_block_table(producer->get_id(), 0).back()->get_references_count(), 2);

    producer->remove_last_tokens(1);
    producer_group->update_processed_tokens_num(7);
    block_manager.free_empty_physical_blocks(producer_group);

    auto original_restore_group = create_sequence_group(std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7}, 56);
    ASSERT_TRUE(block_manager.restore_cached_blocks(original_restore_group));
    EXPECT_EQ(original_restore_group->get_num_processed_tokens(), 7);
    EXPECT_EQ(block_manager.get_block_table(original_restore_group->get_running_sequences().front()->get_id(), 0)
                  .back()
                  ->get_index(),
              original_index);

    producer->append_token(40, 0.0f);
    producer_group->schedule_tokens(1);
    const auto copy_map = block_manager.append_slots(producer_group);
    ASSERT_EQ(copy_map.count(static_cast<size_t>(original_index)), 1);
    EXPECT_FALSE(block_manager.get_block_table(producer->get_id(), 0).back()->has_published_hash());
    producer_group->finish_iteration();
    block_manager.publish_completed_block(producer, 8);

    auto divergent_restore_group = create_sequence_group(std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 40}, 57);
    ASSERT_TRUE(block_manager.restore_cached_blocks(divergent_restore_group));
    EXPECT_NE(block_manager.get_block_table(divergent_restore_group->get_running_sequences().front()->get_id(), 0)
                  .back()
                  ->get_index(),
              original_index);

    block_manager.free_sequence(producer->get_id());
    block_manager.free_sequence(owner_id);
    block_manager.free_sequence(original_restore_group->get_running_sequences().front()->get_id());
    block_manager.free_sequence(divergent_restore_group->get_running_sequences().front()->get_id());
}

TEST(TestBlockManager, PrefixCachingRollbackTransfersDuplicateIdentityToOlderPhysicalOwner) {
    constexpr size_t block_size = 4;
    ov::genai::BlockManager block_manager(12, true, block_size);

    const std::vector<int64_t> prompt = {0, 1, 2, 3};
    auto older_group = create_sequence_group(prompt, 60);
    auto newer_group = create_sequence_group(prompt, 61);
    const auto older_sequence = older_group->get_running_sequences().front();
    const auto newer_sequence = newer_group->get_running_sequences().front();
    for (const int64_t token : std::vector<int64_t>{4, 5, 6, 7}) {
        older_sequence->append_token(token, 0.0f);
        newer_sequence->append_token(token, 0.0f);
    }
    older_group->schedule_tokens(8);
    block_manager.append_slots(older_group);
    older_group->finish_iteration();
    newer_group->schedule_tokens(8);
    block_manager.append_slots(newer_group);
    newer_group->finish_iteration();

    const int older_index = block_manager.get_block_table(older_sequence->get_id(), 0).back()->get_index();
    const int newer_index = block_manager.get_block_table(newer_sequence->get_id(), 0).back()->get_index();
    ASSERT_NE(older_index, newer_index);

    newer_sequence->remove_last_tokens(1);
    newer_group->update_processed_tokens_num(7);
    block_manager.free_empty_physical_blocks(newer_group);

    auto preserved_restore_group = create_sequence_group(std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7}, 62);
    ASSERT_TRUE(block_manager.restore_cached_blocks(preserved_restore_group));
    EXPECT_EQ(block_manager.get_block_table(preserved_restore_group->get_running_sequences().front()->get_id(), 0)
                  .back()
                  ->get_index(),
              older_index);
    block_manager.free_sequence(preserved_restore_group->get_running_sequences().front()->get_id());

    older_sequence->remove_last_tokens(1);
    older_group->update_processed_tokens_num(7);
    EXPECT_NO_THROW(block_manager.free_empty_physical_blocks(older_group));

    auto invalidated_restore_group = create_sequence_group(std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7}, 63);
    ASSERT_TRUE(block_manager.restore_cached_blocks(invalidated_restore_group));
    EXPECT_EQ(invalidated_restore_group->get_num_processed_tokens(), 4);
    EXPECT_EQ(block_manager.get_block_table(invalidated_restore_group->get_running_sequences().front()->get_id(), 0)
                  .size(),
              1);

    block_manager.free_sequence(older_sequence->get_id());
    block_manager.free_sequence(newer_sequence->get_id());
    block_manager.free_sequence(invalidated_restore_group->get_running_sequences().front()->get_id());
}

TEST(TestBlockManager, PrefixCachingRollbackPreservesNewerDuplicateIdentityOwner) {
    constexpr size_t block_size = 4;
    ov::genai::BlockManager block_manager(12, true, block_size);

    const std::vector<int64_t> prompt = {0, 1, 2, 3};
    auto older_group = create_sequence_group(prompt, 64);
    auto newer_group = create_sequence_group(prompt, 65);
    const auto older_sequence = older_group->get_running_sequences().front();
    const auto newer_sequence = newer_group->get_running_sequences().front();
    for (const int64_t token : std::vector<int64_t>{4, 5, 6, 7}) {
        older_sequence->append_token(token, 0.0f);
        newer_sequence->append_token(token, 0.0f);
    }
    older_group->schedule_tokens(8);
    block_manager.append_slots(older_group);
    older_group->finish_iteration();
    newer_group->schedule_tokens(8);
    block_manager.append_slots(newer_group);
    newer_group->finish_iteration();

    const int older_index = block_manager.get_block_table(older_sequence->get_id(), 0).back()->get_index();
    const int newer_index = block_manager.get_block_table(newer_sequence->get_id(), 0).back()->get_index();
    ASSERT_NE(older_index, newer_index);

    older_sequence->remove_last_tokens(1);
    older_group->update_processed_tokens_num(7);
    block_manager.free_empty_physical_blocks(older_group);

    auto preserved_restore_group = create_sequence_group(std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7}, 66);
    ASSERT_TRUE(block_manager.restore_cached_blocks(preserved_restore_group));
    EXPECT_EQ(block_manager.get_block_table(preserved_restore_group->get_running_sequences().front()->get_id(), 0)
                  .back()
                  ->get_index(),
              newer_index);
    block_manager.free_sequence(preserved_restore_group->get_running_sequences().front()->get_id());

    newer_sequence->remove_last_tokens(1);
    newer_group->update_processed_tokens_num(7);
    block_manager.free_empty_physical_blocks(newer_group);

    auto invalidated_restore_group = create_sequence_group(std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7}, 67);
    ASSERT_TRUE(block_manager.restore_cached_blocks(invalidated_restore_group));
    EXPECT_EQ(invalidated_restore_group->get_num_processed_tokens(), 4);
    EXPECT_EQ(block_manager.get_block_table(invalidated_restore_group->get_running_sequences().front()->get_id(), 0)
                  .size(),
              1);

    block_manager.free_sequence(older_sequence->get_id());
    block_manager.free_sequence(newer_sequence->get_id());
    block_manager.free_sequence(invalidated_restore_group->get_running_sequences().front()->get_id());
}

TEST(TestBlockManager, PrefixCachingRollbackKeepsKvAndLinearAttentionRegistriesIndependent) {
    ov::genai::BlockManager kv_block_manager(8, true, /*block_size=*/4);
    ov::genai::BlockManager la_block_manager(
        8,
        true,
        /*block_size=*/3,
        /*num_layers=*/1,
        /*fixed_blocks_per_sequence=*/0,
        /*restore_latest_prefix_block_only=*/true);

    const std::vector<int64_t> prompt = {0, 1, 2, 3};
    auto sequence_group = create_sequence_group(prompt, 58);
    const auto sequence = sequence_group->get_running_sequences().front();
    for (const int64_t token : std::vector<int64_t>{4, 5, 6, 7, 8}) {
        sequence->append_token(token, 0.0f);
    }
    sequence_group->schedule_tokens(9);
    kv_block_manager.append_slots(sequence_group);
    la_block_manager.append_slots(sequence_group);
    sequence_group->finish_iteration();

    sequence->remove_last_tokens(1);
    sequence_group->update_processed_tokens_num(8);
    kv_block_manager.free_empty_physical_blocks(sequence_group);
    la_block_manager.free_empty_physical_blocks(sequence_group);

    auto original_restore_group = create_sequence_group(std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8}, 59);
    ASSERT_TRUE(kv_block_manager.restore_cached_blocks(original_restore_group));
    EXPECT_EQ(original_restore_group->get_num_processed_tokens(), 8);
    kv_block_manager.free_sequence(original_restore_group->get_running_sequences().front()->get_id());

    original_restore_group->update_processed_tokens_num(0);
    ASSERT_TRUE(la_block_manager.restore_cached_blocks(original_restore_group));
    EXPECT_EQ(original_restore_group->get_num_processed_tokens(), 6);
    EXPECT_EQ(la_block_manager.get_block_table_logical_start(
                  original_restore_group->get_running_sequences().front()->get_id()),
              0);

    kv_block_manager.free_sequence(sequence->get_id());
    la_block_manager.free_sequence(sequence->get_id());
    la_block_manager.free_sequence(original_restore_group->get_running_sequences().front()->get_id());
}

TEST(TestBlockManager, PrefixCachingRollbackPreservesWarmRowsAtZeroAndExactBoundariesAcrossLayers) {
    constexpr size_t block_size = 4;
    constexpr size_t num_layers = 2;
    ov::genai::BlockManager block_manager(8, true, block_size, num_layers);

    auto exact_group = create_sequence_group(std::vector<int64_t>{0, 1, 2, 3}, 68);
    const auto exact_sequence = exact_group->get_running_sequences().front();
    for (const int64_t token : std::vector<int64_t>{4, 5, 6, 7}) {
        exact_sequence->append_token(token, 0.0f);
    }
    exact_group->schedule_tokens(8);
    block_manager.append_slots(exact_group);
    exact_group->finish_iteration();
    const int exact_boundary_index = block_manager.get_block_table(exact_sequence->get_id(), 0).back()->get_index();

    exact_sequence->remove_last_tokens(4);
    exact_group->update_processed_tokens_num(4);
    block_manager.free_empty_physical_blocks(exact_group);

    auto exact_restore_group = create_sequence_group(std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7}, 69);
    ASSERT_TRUE(block_manager.restore_cached_blocks(exact_restore_group));
    const uint64_t exact_restore_id = exact_restore_group->get_running_sequences().front()->get_id();
    EXPECT_EQ(exact_restore_group->get_num_processed_tokens(), 7);
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        EXPECT_EQ(block_manager.get_block_table(exact_restore_id, layer_idx).back()->get_index(),
                  exact_boundary_index);
        EXPECT_EQ(block_manager.get_block_table(exact_restore_id, layer_idx).back()->get_references_count(), 1);
    }

    auto zero_group = create_sequence_group(std::vector<int64_t>{10, 11, 12, 13}, 70);
    const auto zero_sequence = zero_group->get_running_sequences().front();
    zero_group->schedule_tokens(4);
    block_manager.append_slots(zero_group);
    zero_group->finish_iteration();
    const int zero_boundary_index = block_manager.get_block_table(zero_sequence->get_id(), 0).back()->get_index();

    zero_group->update_processed_tokens_num(0);
    block_manager.free_empty_physical_blocks(zero_group);
    EXPECT_FALSE(block_manager.has_block_table(zero_sequence->get_id()));

    auto zero_restore_group = create_sequence_group(std::vector<int64_t>{10, 11, 12, 13}, 71);
    ASSERT_TRUE(block_manager.restore_cached_blocks(zero_restore_group));
    const uint64_t zero_restore_id = zero_restore_group->get_running_sequences().front()->get_id();
    EXPECT_EQ(zero_restore_group->get_num_processed_tokens(), 3);
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        EXPECT_EQ(block_manager.get_block_table(zero_restore_id, layer_idx).back()->get_index(), zero_boundary_index);
        EXPECT_EQ(block_manager.get_block_table(zero_restore_id, layer_idx).back()->get_references_count(), 1);
    }

    block_manager.free_sequence(exact_sequence->get_id());
    block_manager.free_sequence(exact_restore_id);
    block_manager.free_sequence(zero_restore_id);
}

TEST(TestBlockManager, PrefixCachingRollbackInvalidatesLatestOnlyEndpointWithLogicalOffset) {
    constexpr size_t block_size = 4;
    ov::genai::BlockManager block_manager(
        /*num_blocks=*/3,
        /*enable_prefix_caching=*/true,
        block_size,
        /*num_layers=*/1,
        /*fixed_blocks_per_sequence=*/0,
        /*restore_latest_prefix_block_only=*/true);

    std::vector<int64_t> source_tokens(12);
    std::iota(source_tokens.begin(), source_tokens.end(), 0);
    auto producer_group = create_sequence_group(source_tokens, 72);
    producer_group->schedule_tokens(source_tokens.size());
    block_manager.append_slots(producer_group);
    producer_group->finish_iteration();
    block_manager.free_sequence(producer_group->get_running_sequences().front()->get_id());

    auto pressure_group = create_sequence_group(std::vector<int64_t>{20, 21, 22, 23}, 73);
    pressure_group->schedule_tokens(4);
    block_manager.append_slots(pressure_group);
    pressure_group->finish_iteration();

    auto active_group = create_sequence_group(source_tokens, 74);
    ASSERT_TRUE(block_manager.restore_cached_blocks(active_group));
    const auto active_sequence = active_group->get_running_sequences().front();
    ASSERT_EQ(block_manager.get_block_table_logical_start(active_sequence->get_id()), 1);
    ASSERT_EQ(block_manager.get_block_table(active_sequence->get_id(), 0).size(), 2);
    block_manager.free_sequence(pressure_group->get_running_sequences().front()->get_id());

    active_sequence->append_token(12, 0.0f);
    active_sequence->append_token(13, 0.0f);
    active_group->schedule_tokens(3);
    const auto table_before = block_manager.get_block_table(active_sequence->get_id(), 0);
    ASSERT_EQ(table_before.size(), 2u);
    ASSERT_TRUE(table_before.back()->has_published_hash());
    EXPECT_FALSE(block_manager.can_append_slots(active_group));
    EXPECT_THROW(block_manager.append_slots(active_group), ov::Exception);

    const auto& table_after = block_manager.get_block_table(active_sequence->get_id(), 0);
    ASSERT_EQ(table_after.size(), table_before.size());
    for (size_t block_idx = 0; block_idx < table_before.size(); ++block_idx) {
        EXPECT_EQ(table_after[block_idx]->get_index(), table_before[block_idx]->get_index());
        EXPECT_EQ(table_after[block_idx]->get_references_count(), table_before[block_idx]->get_references_count());
        EXPECT_EQ(table_after[block_idx]->has_published_hash(), table_before[block_idx]->has_published_hash());
    }
    EXPECT_EQ(active_group->get_num_processed_tokens(), 11u);
    EXPECT_EQ(block_manager.get_block_table_logical_start(active_sequence->get_id()), 1);

    block_manager.free_sequence(active_sequence->get_id());
}

TEST(TestBlockManager, PrefixCachingRollbackDuplicateOwnerSurvivesOverwriteStoreThenEvictsCleanly) {
    constexpr size_t block_size = 4;
    constexpr size_t num_layers = 2;
    ov::genai::BlockManager block_manager(4, true, block_size, num_layers);

    const std::vector<int64_t> prompt = {0, 1, 2, 3};
    auto older_group = create_sequence_group(prompt, 77);
    auto newer_group = create_sequence_group(prompt, 78);
    const auto older_sequence = older_group->get_running_sequences().front();
    const auto newer_sequence = newer_group->get_running_sequences().front();
    for (const int64_t token : std::vector<int64_t>{4, 5, 6, 7}) {
        older_sequence->append_token(token, 0.0f);
        newer_sequence->append_token(token, 0.0f);
    }
    older_group->schedule_tokens(8);
    block_manager.append_slots(older_group);
    older_group->finish_iteration();
    newer_group->schedule_tokens(8);
    block_manager.append_slots(newer_group);
    newer_group->finish_iteration();
    const int older_boundary_index = block_manager.get_block_table(older_sequence->get_id(), 0).back()->get_index();

    block_manager.free_sequence(older_sequence->get_id());
    newer_sequence->remove_last_tokens(1);
    newer_group->update_processed_tokens_num(7);
    block_manager.free_empty_physical_blocks(newer_group);

    auto restored_group = create_sequence_group(std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7}, 79);
    ASSERT_TRUE(block_manager.restore_cached_blocks(restored_group));
    const uint64_t restored_id = restored_group->get_running_sequences().front()->get_id();
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        EXPECT_EQ(block_manager.get_block_table(restored_id, layer_idx).back()->get_index(), older_boundary_index);
        EXPECT_EQ(block_manager.get_block_table(restored_id, layer_idx).back()->get_references_count(), 1);
    }
    block_manager.free_sequence(restored_id);

    auto rewrite_group = create_sequence_group(std::vector<int64_t>{20, 21, 22, 23, 24, 25, 26, 27}, 80);
    rewrite_group->schedule_tokens(8);
    block_manager.append_slots(rewrite_group);
    rewrite_group->finish_iteration();
    ASSERT_EQ(block_manager.num_free_blocks(), 0);

    auto stale_restore_group = create_sequence_group(std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7}, 81);
    ASSERT_TRUE(block_manager.restore_cached_blocks(stale_restore_group));
    const uint64_t stale_restore_id = stale_restore_group->get_running_sequences().front()->get_id();
    EXPECT_EQ(stale_restore_group->get_num_processed_tokens(), 4);
    EXPECT_EQ(block_manager.get_block_table(stale_restore_id, 0).size(), 1);
    EXPECT_NE(block_manager.get_block_table(stale_restore_id, 0).back()->get_index(), older_boundary_index);
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        EXPECT_EQ(block_manager.get_block_table(stale_restore_id, layer_idx).back()->get_references_count(), 2);
    }

    block_manager.free_sequence(newer_sequence->get_id());
    block_manager.free_sequence(rewrite_group->get_running_sequences().front()->get_id());
    block_manager.free_sequence(stale_restore_id);
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
    EXPECT_FALSE(block_manager.get_block_table(first_seq_id, 0).at(1)->has_published_hash());
    EXPECT_TRUE(block_manager.get_block_table(second_seq_id, 0).at(1)->has_published_hash());

    block_manager.free_sequence(first_seq_id);
    auto third_consumer_group = create_sequence_group(tokens, 28);
    block_manager.restore_cached_blocks(third_consumer_group);
    const auto third_seq_id = third_consumer_group->get_running_sequences().at(0)->get_id();
    ASSERT_EQ(block_manager.get_block_table(third_seq_id, 0).size(), 2);
    EXPECT_EQ(block_manager.get_block_table(third_seq_id, 0).at(1)->get_index(), incomplete_checkpoint_idx);

    block_manager.free_sequence(second_seq_id);
    block_manager.free_sequence(third_seq_id);
}

TEST(TestBlockManager, PrefixCachingIncompleteCheckpointUsesCopyOnWriteWhenTableGrows) {
    constexpr size_t block_size = 4;
    ov::genai::BlockManager block_manager(
        /*num_blocks=*/8,
        /*enable_prefix_caching=*/true,
        block_size,
        /*num_layers=*/1,
        /*fixed_blocks_per_sequence=*/0,
        /*restore_latest_prefix_block_only=*/true);

    std::vector<int64_t> producer_tokens = {0, 1, 2, 3, 4, 5};
    auto producer_group = create_sequence_group(producer_tokens, 26);
    producer_group->schedule_tokens(producer_tokens.size());
    block_manager.append_slots(producer_group);
    producer_group->finish_iteration();

    const auto producer_seq_id = producer_group->get_running_sequences().at(0)->get_id();
    const auto incomplete_checkpoint_idx =
        block_manager.get_block_table(producer_seq_id, 0).at(1)->get_index();

    std::vector<int64_t> consumer_tokens = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto consumer_group = create_sequence_group(consumer_tokens, 27);
    block_manager.restore_cached_blocks(consumer_group);

    const auto consumer_seq_id = consumer_group->get_running_sequences().at(0)->get_id();
    ASSERT_EQ(block_manager.get_block_table(consumer_seq_id, 0).size(), 1);
    EXPECT_EQ(block_manager.get_block_table_logical_start(consumer_seq_id), 1);
    EXPECT_EQ(block_manager.get_block_table(consumer_seq_id, 0).at(0)->get_index(),
              incomplete_checkpoint_idx);
    EXPECT_EQ(block_manager.get_block_table(consumer_seq_id, 0).at(0)->get_references_count(), 3);
    EXPECT_TRUE(block_manager.get_block_table(consumer_seq_id, 0).at(0)->copy_on_write());

    consumer_group->schedule_tokens(4);
    const auto copy_map = block_manager.append_slots(consumer_group);

    ASSERT_EQ(block_manager.get_block_table(consumer_seq_id, 0).size(), 2);
    ASSERT_TRUE(copy_map.count(incomplete_checkpoint_idx));
    ASSERT_EQ(copy_map.at(incomplete_checkpoint_idx).size(), 1);
    EXPECT_EQ(copy_map.at(incomplete_checkpoint_idx).front(),
              block_manager.get_block_table(consumer_seq_id, 0).at(0)->get_index());
    EXPECT_NE(block_manager.get_block_table(consumer_seq_id, 0).at(0)->get_index(),
              incomplete_checkpoint_idx);
    EXPECT_NE(block_manager.get_block_table(consumer_seq_id, 0).at(1)->get_index(),
              incomplete_checkpoint_idx);
    EXPECT_EQ(block_manager.get_block_table(consumer_seq_id, 0).at(0)->get_references_count(), 2);
    EXPECT_FALSE(block_manager.get_block_table(consumer_seq_id, 0).at(0)->copy_on_write());
    EXPECT_FALSE(block_manager.get_block_table(consumer_seq_id, 0).at(0)->has_published_hash());
    EXPECT_EQ(block_manager.get_block_table(producer_seq_id, 0).at(1)->get_index(),
              incomplete_checkpoint_idx);
    EXPECT_EQ(block_manager.get_block_table(producer_seq_id, 0).at(1)->get_references_count(), 1);
    EXPECT_TRUE(block_manager.get_block_table(producer_seq_id, 0).at(1)->has_published_hash());

    block_manager.free_sequence(consumer_seq_id);
    auto second_consumer_group = create_sequence_group(producer_tokens, 29);
    block_manager.restore_cached_blocks(second_consumer_group);
    const auto second_consumer_seq_id = second_consumer_group->get_running_sequences().at(0)->get_id();
    ASSERT_EQ(block_manager.get_block_table(second_consumer_seq_id, 0).size(), 1);
    EXPECT_EQ(block_manager.get_block_table(second_consumer_seq_id, 0).at(0)->get_index(),
              incomplete_checkpoint_idx);

    block_manager.free_sequence(producer_seq_id);
    block_manager.free_sequence(second_consumer_seq_id);
}

TEST(TestBlockManager, PrefixCachingCopyOnWriteAndTableGrowthShareCapacityBudget) {
    constexpr size_t block_size = 4;
    constexpr size_t num_layers = 2;
    ov::genai::BlockManager block_manager(3, true, block_size, num_layers);

    auto sequence_group = create_sequence_group(std::vector<int64_t>{0, 1, 2, 3, 4}, 39);
    sequence_group->schedule_tokens(2);
    block_manager.append_slots(sequence_group);
    sequence_group->finish_iteration();

    const auto parent = sequence_group->get_running_sequences().front();
    const auto child = sequence_group->fork_sequence(parent);
    block_manager.fork_sequence(parent->get_id(), child->get_id());
    const int shared_index = block_manager.get_block_table(parent->get_id(), 0).back()->get_index();
    ASSERT_EQ(block_manager.get_block_table(parent->get_id(), 0).back()->get_references_count(), 2);

    sequence_group->schedule_tokens(3);
    ASSERT_EQ(sequence_group->get_num_processed_tokens(), 2);
    ASSERT_EQ(sequence_group->num_running_seqs(), 2);
    ASSERT_EQ(block_manager.get_block_table(parent->get_id(), 0).back()->get_references_count(), 2);
    EXPECT_EQ(block_manager.required_blocks_count(sequence_group), 3);
    EXPECT_EQ(block_manager.num_free_blocks(), 2);
    EXPECT_FALSE(block_manager.can_append_slots(sequence_group));

    block_manager.increase_block_count(4);
    EXPECT_EQ(block_manager.num_free_blocks(), 3);
    EXPECT_TRUE(block_manager.can_append_slots(sequence_group));

    const auto copy_map = block_manager.append_slots(sequence_group);
    EXPECT_EQ(block_manager.num_free_blocks(), 0);
    ASSERT_EQ(copy_map.count(static_cast<size_t>(shared_index)), 1);
    ASSERT_EQ(copy_map.at(static_cast<size_t>(shared_index)).size(), 1);
    const int destination_index = static_cast<int>(copy_map.at(static_cast<size_t>(shared_index)).front());
    for (const auto& sequence : sequence_group->get_running_sequences()) {
        for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            const auto& block_table = block_manager.get_block_table(sequence->get_id(), layer_idx);
            ASSERT_EQ(block_table.size(), 2);
            if (block_table.front()->get_index() == destination_index) {
                EXPECT_EQ(block_table.front()->get_references_count(), 1);
                EXPECT_FALSE(block_table.front()->has_published_hash());
            }
        }
        block_manager.free_sequence(sequence->get_id());
    }
}

TEST(TestBlockManager, PrefixCachingUniquePublishedSameIntervalRequiresWritableCopy) {
    constexpr size_t block_size = 4;
    constexpr size_t num_layers = 2;
    ov::genai::BlockManager block_manager(
        /*num_blocks=*/3,
        /*enable_prefix_caching=*/true,
        block_size,
        num_layers,
        /*fixed_blocks_per_sequence=*/0,
        /*restore_latest_prefix_block_only=*/true);

    const std::vector<int64_t> producer_tokens = {0, 1, 2, 3, 4, 5};
    auto producer_group = create_sequence_group(producer_tokens, 87);
    producer_group->schedule_tokens(producer_tokens.size());
    block_manager.append_slots(producer_group);
    producer_group->finish_iteration();
    const auto producer = producer_group->get_running_sequences().front();
    const int published_index = block_manager.get_block_table(producer->get_id(), 0).back()->get_index();
    block_manager.free_sequence(producer->get_id());

    auto consumer_group = create_sequence_group(std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6}, 88);
    ASSERT_TRUE(block_manager.restore_cached_blocks(consumer_group));
    const auto consumer = consumer_group->get_running_sequences().front();
    ASSERT_EQ(consumer_group->get_num_processed_tokens(), producer_tokens.size());
    ASSERT_TRUE(block_manager.get_block_table(consumer->get_id(), 0).back()->has_published_hash());
    ASSERT_EQ(block_manager.get_block_table(consumer->get_id(), 0).back()->get_references_count(), 2);

    consumer_group->schedule_tokens(1);
    EXPECT_EQ(block_manager.required_blocks_count(consumer_group), 1);
    EXPECT_TRUE(block_manager.can_append_slots(consumer_group));
    const auto copy_map = block_manager.append_slots(consumer_group);

    ASSERT_EQ(copy_map.count(static_cast<size_t>(published_index)), 1);
    ASSERT_EQ(copy_map.at(static_cast<size_t>(published_index)).size(), 1);
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        const auto& writable_row = block_manager.get_block_table(consumer->get_id(), layer_idx).back();
        EXPECT_NE(writable_row->get_index(), published_index);
        EXPECT_FALSE(writable_row->has_published_hash());
        EXPECT_EQ(writable_row->get_references_count(), 2);
    }

    block_manager.free_sequence(consumer->get_id());
}

TEST(TestBlockManager, PrefixCachingRestoredLiveCheckpointRejectsCombinedDeficitBeforeMutation) {
    constexpr size_t block_size = 4;
    ov::genai::BlockManager block_manager(3, true, block_size, 1, 0, true);

    std::vector<int64_t> tokens(12);
    std::iota(tokens.begin(), tokens.end(), 0);
    auto producer_group = create_sequence_group(tokens, 89);
    producer_group->schedule_tokens(tokens.size());
    block_manager.append_slots(producer_group);
    producer_group->finish_iteration();
    block_manager.free_sequence(producer_group->get_running_sequences().front()->get_id());

    auto pressure_group = create_sequence_group(std::vector<int64_t>{20, 21, 22, 23}, 90);
    pressure_group->schedule_tokens(4);
    block_manager.append_slots(pressure_group);
    pressure_group->finish_iteration();

    std::vector<int64_t> active_tokens(14);
    std::iota(active_tokens.begin(), active_tokens.end(), 0);
    auto active_group = create_sequence_group(active_tokens, 91);
    ASSERT_TRUE(block_manager.restore_cached_blocks(active_group));
    const auto active_sequence = active_group->get_running_sequences().front();
    block_manager.free_sequence(pressure_group->get_running_sequences().front()->get_id());
    active_group->update_processed_tokens_num(11);
    active_group->schedule_tokens(3);

    const auto original_table = block_manager.get_block_table(active_sequence->get_id(), 0);
    const auto original_live_state = block_manager.get_linear_attention_live_state(active_sequence->get_id());
    ASSERT_EQ(original_live_state.endpoint, 12);
    ASSERT_EQ(block_manager.required_blocks_count(active_group), 2);
    ASSERT_EQ(block_manager.num_free_blocks(), 1);
    EXPECT_FALSE(block_manager.can_append_slots(active_group));
    EXPECT_THROW(block_manager.append_slots(active_group), ov::Exception);
    EXPECT_EQ(block_manager.get_block_table(active_sequence->get_id(), 0), original_table);
    const auto& live_state_after_failure = block_manager.get_linear_attention_live_state(active_sequence->get_id());
    EXPECT_EQ(live_state_after_failure.endpoint, original_live_state.endpoint);
    EXPECT_EQ(live_state_after_failure.rows, original_live_state.rows);

    auto restore_group = create_sequence_group(tokens, 92);
    ASSERT_TRUE(block_manager.restore_cached_blocks(restore_group));
    EXPECT_EQ(restore_group->get_num_processed_tokens(), 11);
    block_manager.free_sequence(active_sequence->get_id());
    block_manager.free_sequence(restore_group->get_running_sequences().front()->get_id());
}

TEST(TestBlockManager, PrefixCachingRestoredLiveCheckpointWithSufficientCapacitySurvivesRollback) {
    constexpr size_t block_size = 4;
    ov::genai::BlockManager block_manager(4, true, block_size, 1, 0, true);

    std::vector<int64_t> tokens(12);
    std::iota(tokens.begin(), tokens.end(), 0);
    auto producer_group = create_sequence_group(tokens, 93);
    producer_group->schedule_tokens(tokens.size());
    block_manager.append_slots(producer_group);
    producer_group->finish_iteration();
    block_manager.free_sequence(producer_group->get_running_sequences().front()->get_id());

    auto pressure_group = create_sequence_group(std::vector<int64_t>{20, 21, 22, 23, 24, 25, 26, 27}, 94);
    pressure_group->schedule_tokens(8);
    block_manager.append_slots(pressure_group);
    pressure_group->finish_iteration();

    auto active_group = create_sequence_group(tokens, 95);
    ASSERT_TRUE(block_manager.restore_cached_blocks(active_group));
    const auto active_sequence = active_group->get_running_sequences().front();
    block_manager.free_sequence(pressure_group->get_running_sequences().front()->get_id());
    const int canonical_index = block_manager.get_block_table(active_sequence->get_id(), 0).back()->get_index();
    active_sequence->append_token(12, 0.0f);
    active_sequence->append_token(13, 0.0f);
    active_group->update_processed_tokens_num(11);
    active_group->schedule_tokens(3);

    ASSERT_EQ(block_manager.required_blocks_count(active_group), 2);
    ASSERT_EQ(block_manager.num_free_blocks(), 2);
    ASSERT_TRUE(block_manager.can_append_slots(active_group));
    const auto copy_map = block_manager.append_slots(active_group);
    ASSERT_EQ(copy_map.at(static_cast<size_t>(canonical_index)).size(), 1);
    active_group->finish_iteration();
    active_sequence->remove_last_tokens(1);
    active_group->update_processed_tokens_num(13);
    block_manager.free_empty_physical_blocks(active_group);

    auto restore_group = create_sequence_group(tokens, 96);
    ASSERT_TRUE(block_manager.restore_cached_blocks(restore_group));
    EXPECT_EQ(restore_group->get_num_processed_tokens(), 11);
    EXPECT_EQ(block_manager.get_block_table(restore_group->get_running_sequences().front()->get_id(), 0).back()->get_index(),
              canonical_index);
    block_manager.free_sequence(active_sequence->get_id());
    block_manager.free_sequence(restore_group->get_running_sequences().front()->get_id());
}

TEST(TestBlockManager, PrefixCachingCompletedCopyOnWriteRowIsPublishedOnlyAtAcceptedBoundary) {
    constexpr size_t block_size = 4;
    ov::genai::BlockManager block_manager(8, true, block_size);

    const std::vector<int64_t> source_tokens = {0, 1, 2, 3, 4, 5};
    auto source_group = create_sequence_group(source_tokens, 40);
    source_group->schedule_tokens(source_tokens.size());
    block_manager.append_slots(source_group);
    source_group->finish_iteration();
    const uint64_t source_id = source_group->get_running_sequences().front()->get_id();
    const int shared_incomplete_index = block_manager.get_block_table(source_id, 0).back()->get_index();
    block_manager.free_sequence(source_id);

    const std::vector<int64_t> completed_tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto first_group = create_sequence_group(completed_tokens, 41);
    auto second_group = create_sequence_group(completed_tokens, 42);
    ASSERT_TRUE(block_manager.restore_cached_blocks(first_group));
    ASSERT_TRUE(block_manager.restore_cached_blocks(second_group));
    const auto first_sequence = first_group->get_running_sequences().front();
    const uint64_t first_id = first_sequence->get_id();

    first_group->schedule_tokens(1);
    block_manager.append_slots(first_group);
    const int cow_index = block_manager.get_block_table(first_id, 0).back()->get_index();
    ASSERT_NE(cow_index, shared_incomplete_index);
    EXPECT_FALSE(block_manager.get_block_table(first_id, 0).back()->has_published_hash());
    first_group->finish_iteration();
    block_manager.publish_completed_block(first_sequence, first_group->get_num_processed_tokens());
    EXPECT_FALSE(block_manager.get_block_table(first_id, 0).back()->has_published_hash());

    auto partial_restore_group = create_sequence_group(completed_tokens, 43);
    ASSERT_TRUE(block_manager.restore_cached_blocks(partial_restore_group));
    EXPECT_EQ(partial_restore_group->get_num_processed_tokens(), source_tokens.size());
    block_manager.free_sequence(partial_restore_group->get_running_sequences().front()->get_id());

    first_group->schedule_tokens(1);
    block_manager.append_slots(first_group);
    EXPECT_FALSE(block_manager.get_block_table(first_id, 0).back()->has_published_hash());
    first_group->finish_iteration();
    block_manager.publish_completed_block(first_sequence, first_group->get_num_processed_tokens());
    EXPECT_TRUE(block_manager.get_block_table(first_id, 0).back()->has_published_hash());

    block_manager.free_sequence(first_id);
    block_manager.free_sequence(second_group->get_running_sequences().front()->get_id());
    auto completed_restore_group = create_sequence_group(completed_tokens, 44);
    ASSERT_TRUE(block_manager.restore_cached_blocks(completed_restore_group));
    const uint64_t restored_id = completed_restore_group->get_running_sequences().front()->get_id();
    EXPECT_EQ(block_manager.get_block_table(restored_id, 0).back()->get_index(), cow_index);
    block_manager.free_sequence(restored_id);
}

TEST(TestBlockManager, PublishedLinearAttentionLiveRowStagesPrivateWritesAndKeepsCanonicalIdentity) {
    constexpr size_t block_size = 4;
    ov::genai::BlockManager block_manager(/*num_blocks=*/4,
                                         /*enable_prefix_caching=*/true,
                                         block_size,
                                         /*num_layers=*/1,
                                         /*fixed_blocks_per_sequence=*/0,
                                         /*restore_latest_prefix_block_only=*/true);

    const std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5};
    auto owner_group = create_sequence_group(tokens, 50);
    owner_group->schedule_tokens(tokens.size());
    block_manager.append_slots(owner_group);
    owner_group->finish_iteration();
    const auto owner = owner_group->get_running_sequences().front();
    const uint64_t owner_id = owner->get_id();
    const auto canonical_row = block_manager.get_block_table(owner_id, 0).back();
    const size_t canonical_hash = canonical_row->get_hash();
    block_manager.set_linear_attention_live_state(owner_id, tokens.size(), {canonical_row});
    ASSERT_EQ(canonical_row->get_references_count(), 2);

    owner_group->schedule_tokens(1);
    const auto copy_map = block_manager.append_slots(owner_group);
    const auto private_row = block_manager.get_block_table(owner_id, 0).back();
    ASSERT_NE(private_row, canonical_row);
    EXPECT_FALSE(private_row->has_published_hash());
    ASSERT_EQ(copy_map.at(static_cast<size_t>(canonical_row->get_index())).size(), 1u);
    EXPECT_EQ(copy_map.at(static_cast<size_t>(canonical_row->get_index())).front(), private_row->get_index());
    EXPECT_TRUE(canonical_row->has_published_hash());
    EXPECT_EQ(canonical_row->get_hash(), canonical_hash);
    EXPECT_EQ(block_manager.get_linear_attention_live_state(owner_id).rows.front(), private_row);

    block_manager.free_sequence(owner_id);
}

TEST(TestBlockManager, PrefixCachingCompletedCopyOnWriteKeepsExistingVerifiedIdentity) {
    constexpr size_t block_size = 4;
    ov::genai::BlockManager block_manager(10, true, block_size, /*num_layers=*/2);

    const std::vector<int64_t> completed_tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto completed_group = create_sequence_group(completed_tokens, 45);
    completed_group->schedule_tokens(completed_tokens.size());
    block_manager.append_slots(completed_group);
    completed_group->finish_iteration();
    const uint64_t completed_id = completed_group->get_running_sequences().front()->get_id();
    const int verified_index = block_manager.get_block_table(completed_id, 0).back()->get_index();
    block_manager.free_sequence(completed_id);

    const std::vector<int64_t> source_tokens = {0, 1, 2, 3, 4, 5};
    auto source_group = create_sequence_group(source_tokens, 46);
    source_group->schedule_tokens(source_tokens.size());
    block_manager.append_slots(source_group);
    source_group->finish_iteration();
    const uint64_t source_id = source_group->get_running_sequences().front()->get_id();
    block_manager.free_sequence(source_id);

    auto first_group = create_sequence_group(completed_tokens, 47);
    auto second_group = create_sequence_group(completed_tokens, 48);
    ASSERT_TRUE(block_manager.restore_cached_blocks(
        first_group,
        block_manager.get_prefix_restore_plan(first_group, source_tokens.size())));
    ASSERT_TRUE(block_manager.restore_cached_blocks(
        second_group,
        block_manager.get_prefix_restore_plan(second_group, source_tokens.size())));
    const auto first_sequence = first_group->get_running_sequences().front();
    const uint64_t first_id = first_sequence->get_id();
    first_group->schedule_tokens(2);
    block_manager.append_slots(first_group);
    first_group->finish_iteration();
    const int duplicate_index = block_manager.get_block_table(first_id, 0).back()->get_index();
    ASSERT_NE(duplicate_index, verified_index);
    block_manager.publish_completed_block(first_sequence, first_group->get_num_processed_tokens());
    for (size_t layer_idx = 0; layer_idx < 2; ++layer_idx) {
        EXPECT_FALSE(block_manager.get_block_table(first_id, layer_idx).back()->has_published_hash());
    }

    block_manager.free_sequence(first_id);
    block_manager.free_sequence(second_group->get_running_sequences().front()->get_id());
    auto restore_group = create_sequence_group(completed_tokens, 49);
    ASSERT_TRUE(block_manager.restore_cached_blocks(restore_group));
    const uint64_t restored_id = restore_group->get_running_sequences().front()->get_id();
    EXPECT_EQ(block_manager.get_block_table(restored_id, 0).back()->get_index(), verified_index);
    block_manager.free_sequence(restored_id);
}

TEST_P(PrefixCachingCopyOnWriteLayerTest, EvictsWarmCachedRowWhenFreshPoolIsEmpty) {
    constexpr size_t block_size = 4;
    const size_t num_layers = GetParam();
    ov::genai::BlockManager block_manager(4, true, block_size, num_layers);

    const std::vector<int64_t> shared_tokens = {0, 1, 2, 3, 4, 5};
    auto producer_group = create_sequence_group(shared_tokens, 30);
    producer_group->schedule_tokens(shared_tokens.size());
    block_manager.append_slots(producer_group);
    producer_group->finish_iteration();
    const uint64_t producer_id = producer_group->get_running_sequences().front()->get_id();
    block_manager.free_sequence(producer_id);

    auto first_consumer_group = create_sequence_group(shared_tokens, 31);
    auto second_consumer_group = create_sequence_group(shared_tokens, 32);
    ASSERT_TRUE(block_manager.restore_cached_blocks(first_consumer_group));
    ASSERT_TRUE(block_manager.restore_cached_blocks(second_consumer_group));
    const uint64_t first_consumer_id = first_consumer_group->get_running_sequences().front()->get_id();
    const uint64_t second_consumer_id = second_consumer_group->get_running_sequences().front()->get_id();
    const int shared_checkpoint_index = block_manager.get_block_table(first_consumer_id, 0).back()->get_index();

    const std::vector<int64_t> victim_tokens = {10, 11, 12, 13};
    auto victim_group = create_sequence_group(victim_tokens, 33);
    victim_group->schedule_tokens(victim_tokens.size());
    block_manager.append_slots(victim_group);
    victim_group->finish_iteration();
    const uint64_t victim_id = victim_group->get_running_sequences().front()->get_id();
    const int victim_index = block_manager.get_block_table(victim_id, 0).back()->get_index();

    auto filler_group = create_sequence_group(std::vector<int64_t>{20, 21, 22, 23}, 34);
    filler_group->schedule_tokens(block_size);
    block_manager.append_slots(filler_group);
    filler_group->finish_iteration();
    const uint64_t filler_id = filler_group->get_running_sequences().front()->get_id();
    EXPECT_EQ(block_manager.num_free_blocks(), 0);
    block_manager.free_sequence(victim_id);
    EXPECT_EQ(block_manager.num_free_blocks(), 1);

    first_consumer_group->schedule_tokens(1);
    ASSERT_TRUE(block_manager.can_append_slots(first_consumer_group));
    const auto copy_map = block_manager.append_slots(first_consumer_group);
    ASSERT_EQ(copy_map.at(shared_checkpoint_index).size(), 1);
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        EXPECT_EQ(block_manager.get_block_table(first_consumer_id, layer_idx).back()->get_index(), victim_index);
        EXPECT_FALSE(block_manager.get_block_table(first_consumer_id, layer_idx).back()->has_published_hash());
        EXPECT_EQ(block_manager.get_block_table(second_consumer_id, layer_idx).back()->get_index(), shared_checkpoint_index);
    }

    auto victim_restore_group = create_sequence_group(victim_tokens, 35);
    EXPECT_FALSE(block_manager.restore_cached_blocks(victim_restore_group));

    block_manager.free_sequence(first_consumer_id);
    block_manager.free_sequence(second_consumer_id);
    block_manager.free_sequence(filler_id);
}

TEST_P(PrefixCachingCopyOnWriteLayerTest, FailureRollsBackWholeSequenceGroup) {
    constexpr size_t block_size = 4;
    const size_t num_layers = GetParam();
    ov::genai::BlockManager block_manager(2, true, block_size, num_layers);

    const std::vector<int64_t> shared_tokens = {0, 1};
    auto shared_group = create_sequence_group(shared_tokens, 36);
    shared_group->schedule_tokens(shared_tokens.size());
    block_manager.append_slots(shared_group);
    shared_group->finish_iteration();
    const auto parent = shared_group->get_running_sequences().front();

    const std::vector<int64_t> victim_tokens = {10, 11};
    auto victim_group = create_sequence_group(victim_tokens, 37);
    victim_group->schedule_tokens(victim_tokens.size());
    block_manager.append_slots(victim_group);
    victim_group->finish_iteration();
    const uint64_t victim_id = victim_group->get_running_sequences().front()->get_id();
    const int victim_index = block_manager.get_block_table(victim_id, 0).back()->get_index();
    block_manager.free_sequence(victim_id);

    const auto first_child = shared_group->fork_sequence(parent);
    const auto second_child = shared_group->fork_sequence(parent);
    block_manager.fork_sequence(parent->get_id(), first_child->get_id());
    block_manager.fork_sequence(parent->get_id(), second_child->get_id());
    const int shared_index = block_manager.get_block_table(parent->get_id(), 0).back()->get_index();
    const size_t available_before = block_manager.num_free_blocks();
    ASSERT_EQ(available_before, 1);
    ASSERT_EQ(block_manager.get_block_table(parent->get_id(), 0).back()->get_references_count(), 3);

    shared_group->schedule_tokens(1);
    EXPECT_FALSE(block_manager.can_append_slots(shared_group));
    EXPECT_THROW(block_manager.append_slots(shared_group), ov::Exception);

    EXPECT_EQ(block_manager.num_free_blocks(), available_before);
    for (const auto& sequence : shared_group->get_running_sequences()) {
        for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            EXPECT_EQ(block_manager.get_block_table(sequence->get_id(), layer_idx).back()->get_index(), shared_index);
            EXPECT_EQ(block_manager.get_block_table(sequence->get_id(), layer_idx).back()->get_references_count(), 3);
        }
    }
    auto victim_restore_group = create_sequence_group(victim_tokens, 38);
    EXPECT_TRUE(block_manager.restore_cached_blocks(victim_restore_group));
    const uint64_t restored_victim_id = victim_restore_group->get_running_sequences().front()->get_id();
    EXPECT_EQ(block_manager.get_block_table(restored_victim_id, 0).back()->get_index(), victim_index);
    EXPECT_EQ(block_manager.num_free_blocks(), available_before - 1);

    for (const auto& sequence : shared_group->get_running_sequences()) {
        block_manager.free_sequence(sequence->get_id());
    }
    block_manager.free_sequence(restored_victim_id);
}

INSTANTIATE_TEST_SUITE_P(NumLayers,
                         PrefixCachingCopyOnWriteLayerTest,
                         testing::Values(1, 2));

TEST(TestBlockManager, PrefixCachingLatestOnlyRestoreKeepsLatestBlockWithLogicalOffset) {
    constexpr size_t block_size = 4;
    ov::genai::BlockManager block_manager(
        /*num_blocks=*/8,
        /*enable_prefix_caching=*/true,
        block_size,
        /*num_layers=*/1,
        /*fixed_blocks_per_sequence=*/0,
        /*restore_latest_prefix_block_only=*/true);

    std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto producer_group = create_sequence_group(tokens, 26);
    producer_group->schedule_tokens(tokens.size());
    block_manager.append_slots(producer_group);
    producer_group->finish_iteration();

    const auto producer_seq_id = producer_group->get_running_sequences().at(0)->get_id();
    ASSERT_EQ(block_manager.get_block_table(producer_seq_id, 0).size(), 2);
    const auto older_checkpoint = block_manager.get_block_table(producer_seq_id, 0).at(0);
    const auto latest_checkpoint_idx = block_manager.get_block_table(producer_seq_id, 0).at(1)->get_index();
    const auto latest_checkpoint = block_manager.get_block_table(producer_seq_id, 0).at(1);
    block_manager.free_sequence(producer_seq_id);

    auto consumer_group = create_sequence_group(tokens, 27);
    block_manager.restore_cached_blocks(consumer_group);

    const auto consumer_seq = consumer_group->get_running_sequences().at(0);
    const auto consumer_seq_id = consumer_seq->get_id();
    ASSERT_EQ(block_manager.get_block_table(consumer_seq_id, 0).size(), 2);
    EXPECT_EQ(block_manager.get_block_table(consumer_seq_id, 0).at(1)->get_index(), latest_checkpoint_idx);
    EXPECT_EQ(block_manager.get_block_table_logical_start(consumer_seq_id), 0);
    EXPECT_EQ(older_checkpoint->get_references_count(), 1);
    EXPECT_EQ(latest_checkpoint->get_references_count(), 2);
    EXPECT_FALSE(latest_checkpoint->copy_on_write());
    EXPECT_EQ(consumer_group->get_num_processed_tokens(), tokens.size() - 1);

    consumer_seq->append_token(8, 0.9f);
    consumer_group->update_processed_tokens_num(tokens.size());
    consumer_group->schedule_tokens(1);
    block_manager.append_slots(consumer_group);
    EXPECT_EQ(block_manager.get_block_table(consumer_seq_id, 0).size(), 3);
    EXPECT_EQ(block_manager.get_block_table_logical_start(consumer_seq_id), 0);

    block_manager.free_sequence(consumer_seq_id);
}

TEST(TestBlockManager, PrefixCachingLatestOnlyRestoreKeepsLogicalOffsetWhenOlderBlocksAreMissing) {
    constexpr size_t block_size = 4;
    ov::genai::BlockManager block_manager(
        /*num_blocks=*/2,
        /*enable_prefix_caching=*/true,
        block_size,
        /*num_layers=*/1,
        /*fixed_blocks_per_sequence=*/0,
        /*restore_latest_prefix_block_only=*/true);

    std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto producer_group = create_sequence_group(tokens, 28);
    producer_group->schedule_tokens(tokens.size());
    block_manager.append_slots(producer_group);
    producer_group->finish_iteration();

    const auto producer_seq_id = producer_group->get_running_sequences().at(0)->get_id();
    ASSERT_EQ(block_manager.get_block_table(producer_seq_id, 0).size(), 2);
    const auto older_checkpoint = block_manager.get_block_table(producer_seq_id, 0).at(0);
    const auto latest_checkpoint = block_manager.get_block_table(producer_seq_id, 0).at(1);
    const auto older_checkpoint_idx = block_manager.get_block_table(producer_seq_id, 0).at(0)->get_index();
    const auto latest_checkpoint_idx = block_manager.get_block_table(producer_seq_id, 0).at(1)->get_index();
    block_manager.free_sequence(producer_seq_id);

    std::vector<int64_t> pressure_tokens = {10, 11, 12, 13};
    auto pressure_group = create_sequence_group(pressure_tokens, 29);
    pressure_group->schedule_tokens(pressure_tokens.size());
    block_manager.append_slots(pressure_group);
    const auto pressure_seq_id = pressure_group->get_running_sequences().at(0)->get_id();
    ASSERT_EQ(block_manager.get_block_table(pressure_seq_id, 0).at(0)->get_index(), older_checkpoint_idx);

    auto consumer_group = create_sequence_group(tokens, 30);
    block_manager.restore_cached_blocks(consumer_group);

    const auto consumer_seq_id = consumer_group->get_running_sequences().at(0)->get_id();
    ASSERT_EQ(block_manager.get_block_table(consumer_seq_id, 0).size(), 1);
    EXPECT_EQ(block_manager.get_block_table(consumer_seq_id, 0).at(0)->get_index(), latest_checkpoint_idx);
    EXPECT_EQ(block_manager.get_block_table_logical_start(consumer_seq_id), 1);
    EXPECT_EQ(block_manager.get_block_at_logical_position(consumer_seq_id, 0, 1)->get_index(), latest_checkpoint_idx);
    EXPECT_EQ(older_checkpoint->get_references_count(), 2);
    EXPECT_EQ(latest_checkpoint->get_references_count(), 2);
    EXPECT_FALSE(older_checkpoint->copy_on_write());
    EXPECT_FALSE(latest_checkpoint->copy_on_write());
    EXPECT_EQ(consumer_group->get_num_processed_tokens(), tokens.size() - 1);

    block_manager.free_sequence(pressure_seq_id);
    block_manager.free_sequence(consumer_seq_id);
}

TEST(TestBlockManager, PrefixRestorePlanningCapsFullRestoreToLatestStateRestore) {
    constexpr size_t block_size = 4;
    ov::genai::BlockManager kv_block_manager(
        /*num_blocks=*/8,
        /*enable_prefix_caching=*/true,
        block_size,
        /*num_layers=*/1);
    ov::genai::BlockManager state_block_manager(
        /*num_blocks=*/8,
        /*enable_prefix_caching=*/true,
        block_size,
        /*num_layers=*/1,
        /*fixed_blocks_per_sequence=*/0,
        /*restore_latest_prefix_block_only=*/true);

    std::vector<int64_t> full_prefix_tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    auto kv_producer_group = create_sequence_group(full_prefix_tokens, 28);
    kv_producer_group->schedule_tokens(full_prefix_tokens.size());
    kv_block_manager.append_slots(kv_producer_group);
    kv_producer_group->finish_iteration();
    const auto kv_producer_seq_id = kv_producer_group->get_running_sequences().at(0)->get_id();
    kv_block_manager.free_sequence(kv_producer_seq_id);

    std::vector<int64_t> state_prefix_tokens = {0, 1, 2, 3};
    auto state_producer_group = create_sequence_group(state_prefix_tokens, 29);
    state_producer_group->schedule_tokens(state_prefix_tokens.size());
    state_block_manager.append_slots(state_producer_group);
    state_producer_group->finish_iteration();
    const auto state_producer_seq_id = state_producer_group->get_running_sequences().at(0)->get_id();
    state_block_manager.free_sequence(state_producer_seq_id);

    auto consumer_group = create_sequence_group(full_prefix_tokens, 30);
    const auto kv_full_plan = kv_block_manager.get_prefix_restore_plan(consumer_group);
    ASSERT_EQ(kv_full_plan.cache_token_position, full_prefix_tokens.size());
    ASSERT_EQ(kv_full_plan.block_content_lengths.size(), 2);

    const auto state_plan = state_block_manager.get_prefix_restore_plan(consumer_group, kv_full_plan.cache_token_position);
    ASSERT_EQ(state_plan.cache_token_position, state_prefix_tokens.size());
    ASSERT_EQ(state_plan.block_content_lengths.size(), 1);

    const auto kv_common_plan = kv_block_manager.get_prefix_restore_plan(consumer_group, state_plan.cache_token_position);
    ASSERT_EQ(kv_common_plan.cache_token_position, state_prefix_tokens.size());
    ASSERT_EQ(kv_common_plan.block_content_lengths.size(), 1);

    kv_block_manager.restore_cached_blocks(consumer_group, kv_common_plan);
    state_block_manager.restore_cached_blocks(consumer_group, state_plan);

    const auto consumer_seq_id = consumer_group->get_running_sequences().at(0)->get_id();
    EXPECT_EQ(kv_block_manager.get_block_table(consumer_seq_id, 0).size(), 1);
    EXPECT_EQ(state_block_manager.get_block_table(consumer_seq_id, 0).size(), 1);
    EXPECT_EQ(state_block_manager.get_block_table_logical_start(consumer_seq_id), 0);
    EXPECT_EQ(consumer_group->get_num_processed_tokens(), state_prefix_tokens.size());

    kv_block_manager.free_sequence(consumer_seq_id);
    state_block_manager.free_sequence(consumer_seq_id);
}

TEST(TestBlockManager, PrefixRestoreStalePlanReturnsFalseWithoutRestoring) {
    const size_t block_size = 4;
    ov::genai::BlockManager block_manager(/*num_blocks=*/1, /*enable_prefix_caching=*/true, block_size);

    std::vector<int64_t> tokens = {0, 1, 2, 3};
    auto consumer_group = create_sequence_group(tokens, 31);

    ov::genai::BlockManager::PrefixRestorePlan stale_plan;
    stale_plan.block_content_lengths = {tokens.size()};
    stale_plan.cache_token_position = tokens.size();
    stale_plan.processed_tokens = tokens.size() - 1;

    EXPECT_FALSE(block_manager.restore_cached_blocks(consumer_group, stale_plan));
    EXPECT_FALSE(block_manager.has_block_table(consumer_group->get_running_sequences().at(0)->get_id()));
}
