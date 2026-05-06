// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <numeric>
#include <limits>

#include "openvino/runtime/core.hpp"
#include "openvino/op/concat.hpp"
#include "continuous_batching/cache/cache_orchestrator.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/cache/linear_attention_cache_manager.hpp"
#include "continuous_batching/cache/block_manager.hpp"
#include "sequence_group.hpp"
#include "utils.hpp"
#include "helper.hpp"

namespace {

using namespace ov::genai;

constexpr size_t TEST_BLOCK_SIZE = 4;
constexpr size_t TEST_NUM_DECODER_LAYERS = 12;

/// Helper: Create a model with both KV cache and LinearAttention cache inputs.
std::shared_ptr<ov::Model> create_hybrid_model(ov::Core core, size_t num_layers) {
    ov::NodeVector keys, values, conv_states;
    ov::ParameterVector params;
    ov::element::Type kv_cache_type = core.get_property("CPU", ov::hint::kv_cache_precision);

    // KV cache inputs: key_cache.i, value_cache.i
    auto kv_shape = ov::PartialShape::dynamic(4);
    kv_shape[1] = 12;
    kv_shape[2] = 64;
    kv_shape[3] = 64;

    for (size_t i = 0; i < num_layers; i++) {
        auto key = std::make_shared<ov::op::v0::Parameter>(kv_cache_type, kv_shape);
        auto value = std::make_shared<ov::op::v0::Parameter>(kv_cache_type, kv_shape);
        key->get_output_tensor(0).set_names({"key_cache." + std::to_string(i)});
        value->get_output_tensor(0).set_names({"value_cache." + std::to_string(i)});
        keys.push_back(key);
        values.push_back(value);
        params.push_back(key);
        params.push_back(value);
    }

    // Linear attention cache inputs: conv_state_table.i
    auto la_shape = ov::PartialShape::dynamic(3);
    la_shape[1] = 256;
    la_shape[2] = 128;

    for (size_t i = 0; i < num_layers; i++) {
        auto conv_state = std::make_shared<ov::op::v0::Parameter>(kv_cache_type, la_shape);
        conv_state->get_output_tensor(0).set_names({"conv_state_table." + std::to_string(i)});
        conv_states.push_back(conv_state);
        params.push_back(conv_state);
    }

    const auto& concat1 = std::make_shared<ov::op::v0::Concat>(keys, 1);
    const auto& concat2 = std::make_shared<ov::op::v0::Concat>(values, 1);
    const auto& concat3 = std::make_shared<ov::op::v0::Concat>(conv_states, 1);

    return std::make_shared<ov::Model>(
        ov::OutputVector{concat1, concat2, concat3},
        params);
}

std::shared_ptr<ov::Model> create_state_table_model(ov::Core core, const std::vector<std::string>& state_table_names) {
    ov::OutputVector outputs;
    ov::ParameterVector params;
    ov::element::Type kv_cache_type = core.get_property("CPU", ov::hint::kv_cache_precision);

    auto state_shape = ov::PartialShape::dynamic(3);
    state_shape[1] = 256;
    state_shape[2] = 128;

    for (const std::string& state_table_name : state_table_names) {
        auto state = std::make_shared<ov::op::v0::Parameter>(kv_cache_type, state_shape);
        state->get_output_tensor(0).set_names({state_table_name});
        params.push_back(state);
        outputs.push_back(state->output(0));
    }

    return std::make_shared<ov::Model>(outputs, params);
}

/// Helper: Create a SequenceGroup with specified request ID and forked sequences if needed.
SequenceGroup::Ptr create_sequence_group(uint64_t request_id, size_t num_sequences = 1) {
    std::vector<int64_t> tokens = {0, 1, 2, 3};
    auto group = std::make_shared<SequenceGroup>(
        request_id,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_beam_search_config());

    // Fork additional sequences if needed.
    auto parent = group->get_running_sequences().at(0);
    for (size_t i = 1; i < num_sequences; ++i) {
        group->fork_sequence(parent);
    }

    return group;
}

/// Helper: Create a CacheOrchestrator with both KV and LinearAttention cache types.
/// 
/// @param num_kv_blocks Number of KV blocks to allocate.
/// @param num_la_blocks Number of LinearAttention blocks to allocate.
/// @param kv_block_size Block size for KV cache (in tokens).
/// @param num_layers Number of decoder layers.
/// @param la_fixed_blocks_per_seq Fixed blocks per sequence for LinearAttention cache.
std::shared_ptr<CacheOrchestrator> create_hybrid_orchestrator(
        size_t num_kv_blocks,
        size_t num_la_blocks,
        size_t kv_block_size = TEST_BLOCK_SIZE,
        size_t num_layers = 1,
        size_t la_fixed_blocks_per_seq = 1) {
    ov::Core core;
    ov::InferRequest request = core.compile_model(
        create_hybrid_model(core, TEST_NUM_DECODER_LAYERS)).create_infer_request();

    auto kv_manager = std::make_unique<KVCacheManager>(request);
    auto kv_block_manager = std::make_unique<BlockManager>(
        num_kv_blocks, false, kv_block_size, num_layers);

    auto orchestrator = std::make_shared<CacheOrchestrator>();
    orchestrator->register_cache_type(
        CacheType::KV_CACHE, std::move(kv_manager), std::move(kv_block_manager));

    // Register LinearAttention cache type with fixed-size-per-sequence mode.
    auto la_manager = std::make_unique<LinearAttentionCacheManager>(request);
    auto la_block_manager = std::make_unique<BlockManager>(
        num_la_blocks,
        false,  // no prefix caching
        1,      // block_size = 1 token (one sequence per block)
        num_layers,
        la_fixed_blocks_per_seq);  // fixed blocks per sequence

    orchestrator->register_cache_type(
        CacheType::LINEAR_ATTENTION_CACHE, std::move(la_manager), std::move(la_block_manager));

    return orchestrator;
}

}  // namespace

TEST(TestLinearAttentionCacheManager, HasCacheInputsIgnoresMalformedStateTableSuffixes) {
    ov::Core core;
    ov::CompiledModel compiled_model = core.compile_model(create_state_table_model(core,
        {"conv_state_table.",
         "conv_state_table.12x",
         "conv_state_table.18446744073709551616"}));

    EXPECT_FALSE(LinearAttentionCacheManager::has_cache_inputs(compiled_model));
}

TEST(TestLinearAttentionCacheManager, HasCacheInputsAcceptsLargeStateTableSuffixes) {
    ov::Core core;
    ov::CompiledModel compiled_model = core.compile_model(create_state_table_model(core,
        {"conv_state_table.878332661264156340"}));

    EXPECT_TRUE(LinearAttentionCacheManager::has_cache_inputs(compiled_model));
}

TEST(TestLinearAttentionCacheManager, ConstructorAcceptsLargeStateTableSuffixes) {
    ov::Core core;
    ov::CompiledModel compiled_model = core.compile_model(create_state_table_model(core,
        {"conv_state_table.878332661264156340",
         "conv_state_table.18446744073709551616", // invalid
         "conv_state_table.0"}));  // valid
    ov::InferRequest request = compiled_model.create_infer_request();

    ASSERT_TRUE(LinearAttentionCacheManager::has_cache_inputs(compiled_model));
    LinearAttentionCacheManager manager(request);
    manager.allocate_cache_if_needed(3);

    EXPECT_EQ(manager.get_num_layers(), 2);
    EXPECT_EQ(manager.get_num_cache_tensors(), 2);
    EXPECT_EQ(request.get_tensor("conv_state_table.878332661264156340").get_shape(), (ov::Shape{3, 256, 128}));
    EXPECT_EQ(request.get_tensor("conv_state_table.0").get_shape(), (ov::Shape{3, 256, 128}));
}

/// @test RequiredTokens_UsesMax
/// Verify that required_tokens_count() returns the maximum deficit across all cache types.
///
/// Setup: Create a hybrid orchestrator with:
///   - KV cache: variable-size, block_size=4, 1 block total
///   - LinearAttention cache: fixed-size-per-seq, block_size=1, 2 blocks total (max 2 sequences)
///
/// Scenario: Build a sequence group with 2 running sequences and compare the orchestrator
/// result against direct per-type calculations.
///
/// Expected: required_tokens_count() equals
/// max(kv_required_tokens_count, la_required_tokens_count).
TEST(TestCacheOrchestratorHybrid, RequiredTokens_UsesMax) {
    auto orchestrator = create_hybrid_orchestrator(
        /*num_kv_blocks=*/4,
        /*num_la_blocks=*/2,
        TEST_BLOCK_SIZE,
        /*num_layers=*/1,
        /*la_fixed_blocks_per_seq=*/1);

    auto seq_group = create_sequence_group(100, /*num_sequences=*/2);
    auto running = seq_group->get_running_sequences();
    auto seq1 = running[0];
    auto seq2 = running[1];

    // Allocate one token for each running sequence to establish non-empty block tables.
    orchestrator->allocate_tokens(seq1, seq_group, 1, seq_group->get_prompt_len());
    orchestrator->allocate_tokens(seq2, seq_group, 1, seq_group->get_prompt_len());

    // Increase demand for both cache types and compare orchestrator aggregation with
    // direct per-type computations.
    seq_group->schedule_tokens(5);

    const size_t kv_required_tokens = orchestrator->get_block_manager(CacheType::KV_CACHE).required_tokens_count(seq_group);
    const size_t la_required_tokens = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE).required_tokens_count(seq_group);
    const size_t expected = std::max(kv_required_tokens, la_required_tokens);

    const size_t actual = orchestrator->required_tokens_count(seq_group);
    EXPECT_EQ(actual, expected);

    // Clean up.
    orchestrator->free_sequence(seq1->get_id());
    orchestrator->free_sequence(seq2->get_id());
}

/// @test NumFreeBlocks_UsesMin
/// Verify that num_free_blocks() returns the minimum free blocks across all cache types.
///
/// Setup: Create a hybrid orchestrator with:
///   - KV cache: 8 blocks total
///   - LinearAttention cache: 4 blocks total (fixed-size-per-seq)
///
/// Scenario:
///   1. Initially: KV has 8 free, LA has 4 free. min(8,4) = 4.
///   2. Allocate 2 sequences (each uses 1 LA block, 1 KV block).
///      KV has 7 free, LA has 2 free. min(7,2) = 2.
///   3. Free 1 sequence.
///      KV has 8 free, LA has 3 free. min(8,3) = 3.
///
/// Expected: num_free_blocks() correctly tracks the minimum across types.
TEST(TestCacheOrchestratorHybrid, NumFreeBlocks_UsesMin) {
    auto orchestrator = create_hybrid_orchestrator(
        /*num_kv_blocks=*/8,
        /*num_la_blocks=*/4,
        TEST_BLOCK_SIZE,
        /*num_layers=*/1,
        /*la_fixed_blocks_per_seq=*/1);

    // Initially: min(8, 4) = 4 free blocks.
    EXPECT_EQ(orchestrator->num_free_blocks(), 4);

    // Allocate 2 sequences.
    auto seq_group_1 = create_sequence_group(101, /*num_sequences=*/2);
    auto running_1 = seq_group_1->get_running_sequences();
    orchestrator->allocate_tokens(running_1[0], seq_group_1, 1, seq_group_1->get_prompt_len());
    orchestrator->allocate_tokens(running_1[1], seq_group_1, 1, seq_group_1->get_prompt_len());

    // After allocating 2 sequences: KV has 7 or 8 free (depends on block rounding),
    // LA has 2 free. min should be 2.
    EXPECT_EQ(orchestrator->num_free_blocks(), 2);

    // Free 1 sequence.
    orchestrator->free_sequence(running_1[0]->get_id());

    // After freeing 1: LA has 3 free, KV has more. min should be 3.
    EXPECT_EQ(orchestrator->num_free_blocks(), 3);

    // Clean up.
    orchestrator->free_sequence(running_1[1]->get_id());
}

/// @test GrowFixedSize_OnlyAffectsFixed
/// Verify that grow_fixed_size_capacity() only affects fixed-size-per-sequence cache types,
/// not variable-size types like KV.
///
/// Setup: Create a hybrid orchestrator with:
///   - KV cache (variable-size): 4 blocks initially
///   - LinearAttention cache (fixed-size): 2 blocks initially
///
/// Scenario:
///   1. Record initial KV and LA block counts.
///   2. Call grow_fixed_size_capacity(3) to add capacity for 3 more sequences.
///   3. Verify that:
///      - KV block count remains 4 (unchanged).
///      - LA block count increases to 2 + 3*1 = 5 (or appropriate offset).
///
/// Expected: Only LA cache grows; KV cache is unaffected.
TEST(TestCacheOrchestratorHybrid, GrowFixedSize_OnlyAffectsFixed) {
    auto orchestrator = create_hybrid_orchestrator(
        /*num_kv_blocks=*/4,
        /*num_la_blocks=*/2,
        TEST_BLOCK_SIZE,
        /*num_layers=*/1,
        /*la_fixed_blocks_per_seq=*/1);

    const auto& kv_bm = orchestrator->get_block_manager(CacheType::KV_CACHE);
    const auto& la_bm = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);

    size_t initial_kv_blocks = kv_bm.get_total_number_of_kv_blocks();
    size_t initial_la_blocks = la_bm.get_total_number_of_kv_blocks();

    // Grow fixed-size capacity by 3 sequences.
    orchestrator->grow_fixed_size_capacity(3);

    size_t final_kv_blocks = kv_bm.get_total_number_of_kv_blocks();
    size_t final_la_blocks = la_bm.get_total_number_of_kv_blocks();

    // KV should be unchanged.
    EXPECT_EQ(final_kv_blocks, initial_kv_blocks);

    // LA should grow by 3 blocks (1 per sequence).
    EXPECT_EQ(final_la_blocks, initial_la_blocks + 3);
}

/// @test EnsureTokenCapacity_SkipsFixed
/// Verify that ensure_token_capacity() only affects variable-size cache types,
/// not fixed-size types like LinearAttention.
///
/// Setup: Create a hybrid orchestrator with:
///   - KV cache (variable-size): 4 blocks initially
///   - LinearAttention cache (fixed-size): 2 blocks initially
///
/// Scenario:
///   1. Record initial block counts for both cache types.
///   2. Call ensure_token_capacity(100) to ensure the KV cache can hold 100 tokens.
///   3. Verify that:
///      - KV block count increases to accommodate 100 tokens / block_size.
///      - LA block count remains 2 (fixed-size cache is not affected by token capacity).
///
/// Expected: KV cache grows; LA cache remains unchanged.
TEST(TestCacheOrchestratorHybrid, EnsureTokenCapacity_SkipsFixed) {
    auto orchestrator = create_hybrid_orchestrator(
        /*num_kv_blocks=*/4,
        /*num_la_blocks=*/2,
        TEST_BLOCK_SIZE,
        /*num_layers=*/1,
        /*la_fixed_blocks_per_seq=*/1);

    const auto& kv_bm = orchestrator->get_block_manager(CacheType::KV_CACHE);
    const auto& la_bm = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);

    size_t initial_kv_blocks = kv_bm.get_total_number_of_kv_blocks();
    size_t initial_la_blocks = la_bm.get_total_number_of_kv_blocks();

    // Ensure token capacity for 100 tokens.
    orchestrator->ensure_token_capacity(100);

    size_t final_kv_blocks = kv_bm.get_total_number_of_kv_blocks();
    size_t final_la_blocks = la_bm.get_total_number_of_kv_blocks();

    // KV blocks should increase (at least 100 / TEST_BLOCK_SIZE = 25 blocks).
    EXPECT_GE(final_kv_blocks, (100 + TEST_BLOCK_SIZE - 1) / TEST_BLOCK_SIZE);

    // LA blocks should remain unchanged (fixed-size cache ignores token capacity).
    EXPECT_EQ(final_la_blocks, initial_la_blocks);
}

/// @test TotalCacheBytes_IncludesAll
/// Verify that get_total_cache_size_in_bytes() aggregates memory from all registered cache types.
///
/// Setup: Create a hybrid orchestrator with:
///   - KV cache with block_size_in_bytes = X
///   - LinearAttention cache with block_size_in_bytes = Y
///
/// Scenario:
///   1. Query get_total_cache_size_in_bytes().
///   2. Verify the result equals:
///      (num_kv_blocks * kv_block_size) + (num_la_blocks * la_block_size)
///
/// Expected: Aggregate size correctly includes both cache types.
TEST(TestCacheOrchestratorHybrid, TotalCacheBytes_IncludesAll) {
    const size_t num_kv_blocks = 8;
    const size_t num_la_blocks = 4;

    auto orchestrator = create_hybrid_orchestrator(
        num_kv_blocks,
        num_la_blocks,
        TEST_BLOCK_SIZE,
        /*num_layers=*/1,
        /*la_fixed_blocks_per_seq=*/1);

    const auto& kv_bm = orchestrator->get_block_manager(CacheType::KV_CACHE);
    const auto& kv_cm = orchestrator->get_cache_manager(CacheType::KV_CACHE);
    const auto& la_bm = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
    const auto& la_cm = orchestrator->get_cache_manager(CacheType::LINEAR_ATTENTION_CACHE);

    size_t expected_kv_bytes = kv_bm.get_total_number_of_kv_blocks() * kv_cm.get_block_size_in_bytes();
    size_t expected_la_bytes = la_bm.get_total_number_of_kv_blocks() * la_cm.get_block_size_in_bytes();
    size_t expected_total = expected_kv_bytes + expected_la_bytes;

    size_t actual_total = orchestrator->get_total_cache_size_in_bytes();

    EXPECT_EQ(actual_total, expected_total);
}

TEST(TestCacheOrchestratorHybrid, PartialPreemptionIsDisallowedWhenFixedSizeTargetNeedsBlocks) {
    auto orchestrator = create_hybrid_orchestrator(
        /*num_kv_blocks=*/8,
        /*num_la_blocks=*/1,
        TEST_BLOCK_SIZE,
        /*num_layers=*/1,
        /*la_fixed_blocks_per_seq=*/1);

    auto victim = create_sequence_group(200, /*num_sequences=*/1);
    auto target = create_sequence_group(201, /*num_sequences=*/1);
    auto victim_seq = victim->get_running_sequences()[0];

    // Allocate victim to occupy both KV and fixed-size LA resources.
    orchestrator->allocate_tokens(victim_seq, victim, TEST_BLOCK_SIZE + 1, victim->get_prompt_len());

    // Target has no LA allocation yet, so fixed-size cache needs blocks.
    EXPECT_FALSE(orchestrator->can_partially_preempt(victim, target));

    orchestrator->free_sequence(victim_seq->get_id());
}

TEST(TestCacheOrchestratorHybrid, PartialPreemptionIsDisallowedWhenFixedSizeVictimHasState) {
    auto orchestrator = create_hybrid_orchestrator(
        /*num_kv_blocks=*/16,
        /*num_la_blocks=*/2,
        TEST_BLOCK_SIZE,
        /*num_layers=*/1,
        /*la_fixed_blocks_per_seq=*/1);

    auto victim = create_sequence_group(300, /*num_sequences=*/1);
    auto target = create_sequence_group(301, /*num_sequences=*/1);
    auto victim_seq = victim->get_running_sequences()[0];
    auto target_seq = target->get_running_sequences()[0];

    // Allocate both groups once to consume fixed-size LA blocks. This makes LA deficit zero for target.
    orchestrator->allocate_tokens(victim_seq, victim, TEST_BLOCK_SIZE * 2 + 1, victim->get_prompt_len());
    orchestrator->allocate_tokens(target_seq, target, 1, target->get_prompt_len());

    // Increase target token demand so KV has a non-zero deficit.
    target->schedule_tokens(TEST_BLOCK_SIZE + 1);

    // The target already owns fixed-size LA state, but the victim also owns LA state
    // that cannot represent token-level rollback. Partial preemption must be rejected.
    EXPECT_FALSE(orchestrator->can_partially_preempt(victim, target));

    orchestrator->free_sequence(victim_seq->get_id());
    orchestrator->free_sequence(target_seq->get_id());
}
