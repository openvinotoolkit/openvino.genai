// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <cstring>
#include <numeric>
#include <limits>

#include "openvino/runtime/core.hpp"
#include "openvino/op/concat.hpp"
#include "continuous_batching/cache/cache_orchestrator.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/cache/linear_attention_cache_manager.hpp"
#include "continuous_batching/cache/block_manager.hpp"
#include "continuous_batching/scheduler.hpp"
#include "sequence_group.hpp"
#include "utils.hpp"
#include "helper.hpp"

namespace {

using namespace ov::genai;

constexpr size_t TEST_BLOCK_SIZE = 4;
constexpr size_t TEST_NUM_DECODER_LAYERS = 12;

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

/// Creates a SequenceGroup with optional forks.
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

    auto la_manager = std::make_unique<LinearAttentionCacheManager>(request);
    auto la_block_manager = std::make_unique<BlockManager>(
        num_la_blocks,
        false,  // no prefix caching
        1,      // block_size = 1 token (one sequence per block)
        1,      // one logical block table for all LA layers
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

TEST(TestLinearAttentionCacheManager, ZeroBlocksClearsOnlyRequestedPhysicalBlocks) {
    ov::Core core;
    ov::InferRequest request = core.compile_model(create_state_table_model(core, {"conv_state_table.0"}))
                                      .create_infer_request();

    LinearAttentionCacheManager manager(request);
    manager.allocate_cache_if_needed(2);

    ov::Tensor tensor = request.get_tensor("conv_state_table.0");
    std::memset(tensor.data(), 0x7f, tensor.get_byte_size());

    manager.zero_blocks({1});

    const size_t block_size_in_bytes = tensor.get_byte_size() / tensor.get_shape()[0];
    const uint8_t* data = static_cast<const uint8_t*>(tensor.data());
    EXPECT_TRUE(std::all_of(data, data + block_size_in_bytes, [](uint8_t value) { return value == 0x7f; }));
    EXPECT_TRUE(std::all_of(data + block_size_in_bytes,
                            data + 2 * block_size_in_bytes,
                            [](uint8_t value) { return value == 0; }));
}

TEST(TestCacheOrchestratorHybrid, AppendSlotsZerosReusedLinearAttentionBlockForNewSequence) {
    ov::Core core;
    ov::InferRequest request = core.compile_model(create_hybrid_model(core, TEST_NUM_DECODER_LAYERS))
                                      .create_infer_request();

    auto kv_manager = std::make_unique<KVCacheManager>(request);
    auto kv_block_manager = std::make_unique<BlockManager>(4, false, TEST_BLOCK_SIZE, 1);
    auto la_manager = std::make_unique<LinearAttentionCacheManager>(request);
    auto la_block_manager = std::make_unique<BlockManager>(1, false, 1, 1, 1);

    auto orchestrator = std::make_shared<CacheOrchestrator>();
    orchestrator->register_cache_type(CacheType::KV_CACHE, std::move(kv_manager), std::move(kv_block_manager));
    orchestrator->register_cache_type(CacheType::LINEAR_ATTENTION_CACHE,
                                      std::move(la_manager),
                                      std::move(la_block_manager));

    SchedulerConfig config;
    config.max_num_batched_tokens = 16;
    config.dynamic_split_fuse = false;
    config.max_num_seqs = 1;
    Scheduler scheduler(orchestrator, config);

    std::vector<int64_t> first_tokens = {1, 2, 3, 4};
    auto first_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {first_tokens.size()}, first_tokens.data()),
        utils::get_greedy_config());
    const auto first_seq_id = first_group->get_running_sequences()[0]->get_id();
    std::vector<SequenceGroup::Ptr> first_requests = {first_group};
    std::ignore = scheduler.schedule(first_requests);

    ov::Tensor tensor = request.get_tensor("conv_state_table.0");
    std::memset(tensor.data(), 0x7f, tensor.get_byte_size());
    scheduler.free_sequence(first_seq_id);

    std::vector<int64_t> second_tokens = {5, 6, 7, 8};
    auto second_group = std::make_shared<SequenceGroup>(
        1,
        ov::Tensor(ov::element::i64, {second_tokens.size()}, second_tokens.data()),
        utils::get_greedy_config());
    std::vector<SequenceGroup::Ptr> second_requests = {second_group};
    std::ignore = scheduler.schedule(second_requests);

    const uint8_t* data = static_cast<const uint8_t*>(tensor.data());
    EXPECT_TRUE(std::all_of(data, data + tensor.get_byte_size(), [](uint8_t value) { return value == 0; }));

    const auto second_seq_id = second_group->get_running_sequences()[0]->get_id();
    scheduler.free_sequence(second_seq_id);
}

TEST(TestCacheOrchestratorHybrid, SharedLinearAttentionRegistersSingleBlockTableLayer) {
    ov::Core core;
    ov::InferRequest request = core.compile_model(get_dummy_hybrid_model(core,
                                                                         /*kv_num_layers=*/3,
                                                                         /*la_num_layers=*/3))
                                      .create_infer_request();

    SchedulerConfig config;
    config.num_kv_blocks = 4;
    config.num_linear_attention_blocks = 2;
    config.max_num_seqs = 2;

    auto orchestrator = CacheOrchestrator::create(request,
                                                  config,
                                                  [](const std::string&, size_t) {
                                                      return std::numeric_limits<size_t>::max();
                                                  });

    auto sequence_group = create_sequence_group(99);
    Sequence::Ptr sequence = sequence_group->get_running_sequences().front();
    orchestrator->allocate_tokens(sequence, sequence_group, 1, sequence_group->get_prompt_len());

    const auto& kv_block_tables = orchestrator->get_kv_block_tables(sequence->get_id());
    ASSERT_EQ(kv_block_tables.size(), 1);
    ASSERT_EQ(kv_block_tables[0].size(), 1);

    const auto& la_block_table = orchestrator->get_linear_attention_block_table(sequence->get_id());
    ASSERT_EQ(la_block_table.size(), 1);

    orchestrator->free_sequence(sequence->get_id());
}

TEST(TestCacheOrchestratorHybrid, CreateAcceptsCacheIntervalMultiplierForHybridModel) {
    ov::Core core;
    ov::InferRequest request = core.compile_model(get_dummy_hybrid_model(core,
                                                                         /*kv_num_layers=*/3,
                                                                         /*la_num_layers=*/3))
                                      .create_infer_request();

    SchedulerConfig config;
    config.num_kv_blocks = 4;
    config.num_linear_attention_blocks = 2;
    config.enable_prefix_caching = true;
    config.cache_interval_multiplier = 4;

    EXPECT_NO_THROW(CacheOrchestrator::create(request,
                                              config,
                                              [](const std::string&, size_t) {
                                                  return std::numeric_limits<size_t>::max();
                                              }));
}

TEST(TestCacheOrchestratorHybrid, AdaptiveCacheIntervalMultiplierScalesWithStateSize) {
    using ov::genai::CacheOrchestrator;
    // Small LA state relative to a KV block keeps the default multiplier (fine-grained reuse).
    EXPECT_EQ(CacheOrchestrator::adaptive_cache_interval_multiplier(/*la=*/1024, /*kv=*/4096),
              DEFAULT_LINEAR_ATTENTION_CACHE_INTERVAL_MULTIPLIER);
    EXPECT_EQ(CacheOrchestrator::adaptive_cache_interval_multiplier(/*la=*/4096, /*kv=*/4096),
              DEFAULT_LINEAR_ATTENTION_CACHE_INTERVAL_MULTIPLIER);

    // Large recurrent state (e.g. hybrid SSM): multiplier grows ~ la/kv so one LA
    // checkpoint costs about one KV block, instead of exhausting the cache budget.
    // 51 MiB LA state vs 512 KiB KV block -> ratio ~102.
    EXPECT_EQ(CacheOrchestrator::adaptive_cache_interval_multiplier(/*la=*/size_t(51) * 1024 * 1024,
                                                                    /*kv=*/size_t(512) * 1024),
              102u);

    // Clamped to the upper bound for very large states.
    EXPECT_EQ(CacheOrchestrator::adaptive_cache_interval_multiplier(/*la=*/size_t(4096) * 1024 * 1024,
                                                                    /*kv=*/size_t(64) * 1024),
              CacheOrchestrator::MAX_ADAPTIVE_CACHE_INTERVAL_MULTIPLIER);

    // Degenerate kv block size falls back to the default multiplier (no divide-by-zero).
    EXPECT_EQ(CacheOrchestrator::adaptive_cache_interval_multiplier(/*la=*/1024, /*kv=*/0),
              DEFAULT_LINEAR_ATTENTION_CACHE_INTERVAL_MULTIPLIER);

    // A near-max LA block size must not overflow the ceil-division and still clamps to MAX.
    EXPECT_EQ(CacheOrchestrator::adaptive_cache_interval_multiplier(
                  /*la=*/std::numeric_limits<size_t>::max(), /*kv=*/4096),
              CacheOrchestrator::MAX_ADAPTIVE_CACHE_INTERVAL_MULTIPLIER);
}

// The OOM-drop (GenerationStatus::IGNORED) surfaces an actionable error at the call sites that
// would otherwise discard the status (CB adapter overloads and the VLM result conversion).
TEST(TestCacheOrchestratorHybrid, AssertRequestWasScheduledThrowsOnIgnored) {
    constexpr uint64_t request_id = 7;
    // IGNORED == request dropped by the scheduler (out of cache budget) -> must throw.
    EXPECT_THROW(ov::genai::utils::assert_request_was_scheduled(GenerationStatus::IGNORED, request_id),
                 ov::Exception);

    // All terminal/active states that represent real results must pass through untouched.
    EXPECT_NO_THROW(ov::genai::utils::assert_request_was_scheduled(GenerationStatus::FINISHED, request_id));
    EXPECT_NO_THROW(ov::genai::utils::assert_request_was_scheduled(GenerationStatus::STOP, request_id));
    EXPECT_NO_THROW(ov::genai::utils::assert_request_was_scheduled(GenerationStatus::CANCEL, request_id));
    EXPECT_NO_THROW(ov::genai::utils::assert_request_was_scheduled(GenerationStatus::RUNNING, request_id));
}

TEST(TestCacheOrchestratorHybrid, CreateIgnoresCacheIntervalMultiplierWithoutLinearAttentionCache) {
    ov::Core core;
    ov::InferRequest request = core.compile_model(get_dummy_model(core, /*num_layers=*/3))
                                      .create_infer_request();

    SchedulerConfig config;
    config.num_kv_blocks = 4;
    config.cache_interval_multiplier = 4;

    EXPECT_NO_THROW(CacheOrchestrator::create(request,
                                              config,
                                              [](const std::string&, size_t) {
                                                  return std::numeric_limits<size_t>::max();
                                              }));
}

/// required_tokens_count() returns the maximum deficit across cache types.
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

    // Establish non-empty block tables.
    orchestrator->allocate_tokens(seq1, seq_group, 1, seq_group->get_prompt_len());
    orchestrator->allocate_tokens(seq2, seq_group, 1, seq_group->get_prompt_len());

    // Compare orchestrator aggregation with direct per-type computations.
    seq_group->schedule_tokens(5);

    const size_t kv_required_tokens = orchestrator->get_block_manager(CacheType::KV_CACHE).required_tokens_count(seq_group);
    const size_t la_required_tokens = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE).required_tokens_count(seq_group);
    const size_t expected = std::max(kv_required_tokens, la_required_tokens);

    const size_t actual = orchestrator->required_tokens_count(seq_group);
    EXPECT_EQ(actual, expected);

    orchestrator->free_sequence(seq1->get_id());
    orchestrator->free_sequence(seq2->get_id());
}

/// num_free_blocks() returns the minimum free count across cache types.
TEST(TestCacheOrchestratorHybrid, NumFreeBlocks_UsesMin) {
    auto orchestrator = create_hybrid_orchestrator(
        /*num_kv_blocks=*/8,
        /*num_la_blocks=*/4,
        TEST_BLOCK_SIZE,
        /*num_layers=*/1,
        /*la_fixed_blocks_per_seq=*/1);

    EXPECT_EQ(orchestrator->num_free_blocks(), 4);

    auto seq_group_1 = create_sequence_group(101, /*num_sequences=*/2);
    auto running_1 = seq_group_1->get_running_sequences();
    orchestrator->allocate_tokens(running_1[0], seq_group_1, 1, seq_group_1->get_prompt_len());
    orchestrator->allocate_tokens(running_1[1], seq_group_1, 1, seq_group_1->get_prompt_len());

    EXPECT_EQ(orchestrator->num_free_blocks(), 2);

    orchestrator->free_sequence(running_1[0]->get_id());

    EXPECT_EQ(orchestrator->num_free_blocks(), 3);

    orchestrator->free_sequence(running_1[1]->get_id());
}

/// grow_fixed_size_capacity() affects fixed-size caches only.
TEST(TestCacheOrchestratorHybrid, GrowFixedSize_OnlyAffectsFixed) {
    auto orchestrator = create_hybrid_orchestrator(
        /*num_kv_blocks=*/4,
        /*num_la_blocks=*/2,
        TEST_BLOCK_SIZE,
        /*num_layers=*/1,
        /*la_fixed_blocks_per_seq=*/1);

    const auto& kv_bm = orchestrator->get_block_manager(CacheType::KV_CACHE);
    const auto& la_bm = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);

    size_t initial_kv_blocks = kv_bm.get_total_block_count();
    size_t initial_la_blocks = la_bm.get_total_block_count();

    orchestrator->grow_fixed_size_capacity(3);

    size_t final_kv_blocks = kv_bm.get_total_block_count();
    size_t final_la_blocks = la_bm.get_total_block_count();

    EXPECT_EQ(final_kv_blocks, initial_kv_blocks);
    EXPECT_EQ(final_la_blocks, initial_la_blocks + 3);
}

/// ensure_sequence_token_capacity() skips fixed-size caches.
TEST(TestCacheOrchestratorHybrid, EnsureSequenceTokenCapacity_SkipsFixed) {
    auto orchestrator = create_hybrid_orchestrator(
        /*num_kv_blocks=*/4,
        /*num_la_blocks=*/2,
        TEST_BLOCK_SIZE,
        /*num_layers=*/1,
        /*la_fixed_blocks_per_seq=*/1);

    const auto& kv_bm = orchestrator->get_block_manager(CacheType::KV_CACHE);
    const auto& la_bm = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);

    size_t initial_kv_blocks = kv_bm.get_total_block_count();
    size_t initial_la_blocks = la_bm.get_total_block_count();

    orchestrator->ensure_sequence_token_capacity({{100, 1}});

    size_t final_kv_blocks = kv_bm.get_total_block_count();
    size_t final_la_blocks = la_bm.get_total_block_count();

    EXPECT_GE(final_kv_blocks, (100 + TEST_BLOCK_SIZE - 1) / TEST_BLOCK_SIZE);
    EXPECT_EQ(final_la_blocks, initial_la_blocks);
}

/// get_total_cache_size_in_bytes() aggregates all cache types.
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

    size_t expected_kv_bytes = kv_bm.get_total_block_count() * kv_cm.get_block_size_in_bytes();
    size_t expected_la_bytes = la_bm.get_total_block_count() * la_cm.get_block_size_in_bytes();
    size_t expected_total = expected_kv_bytes + expected_la_bytes;

    size_t actual_total = orchestrator->get_total_cache_size_in_bytes();

    EXPECT_EQ(actual_total, expected_total);
}

/// Provisions one sequence and returns its live sequence id.
namespace {
uint64_t provision_single_sequence(const std::shared_ptr<CacheOrchestrator>& orchestrator,
                                   uint64_t request_id) {
    auto seq_group = create_sequence_group(request_id, /*num_sequences=*/1);
    auto seq = seq_group->get_running_sequences()[0];
    orchestrator->allocate_tokens(seq, seq_group, 1, seq_group->get_prompt_len());
    return seq->get_id();
}

// Owned LA rows other than the current live row.
std::vector<size_t> linear_attention_scratch_blocks(const std::shared_ptr<CacheOrchestrator>& orchestrator,
                                                    uint64_t seq_id) {
    const size_t live_block = orchestrator->get_linear_attention_live_block(seq_id);
    std::vector<size_t> scratch_blocks;
    for (const auto& block : orchestrator->get_linear_attention_block_table(seq_id)) {
        const size_t physical_index = block->get_index();
        if (physical_index != live_block) {
            scratch_blocks.push_back(physical_index);
        }
    }
    return scratch_blocks;
}
}  // namespace

/// Admission reserves 1 + N non-prefix LA rows per sequence.
TEST(TestCacheOrchestratorHybrid, LinearAttentionWorkspace_ReservesOnePlusN) {
    for (size_t n : {size_t{0}, size_t{1}, size_t{3}}) {
        const size_t fixed_blocks = 1 + n;
        auto orchestrator = create_hybrid_orchestrator(
            /*num_kv_blocks=*/16,
            /*num_la_blocks=*/fixed_blocks,
            TEST_BLOCK_SIZE,
            /*num_layers=*/1,
            /*la_fixed_blocks_per_seq=*/fixed_blocks);

        const auto& la_bm = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
        EXPECT_EQ(la_bm.get_fixed_blocks_per_sequence(), fixed_blocks);

        const uint64_t seq_id = provision_single_sequence(orchestrator, 400 + n);

        const auto& owned = orchestrator->get_linear_attention_block_table(seq_id);
        EXPECT_EQ(owned.size(), fixed_blocks);

        orchestrator->free_sequence(seq_id);
    }
}

/// Live LA block defaults to block_table[0] and round-trips through setter.
TEST(TestCacheOrchestratorHybrid, LinearAttentionWorkspace_LiveBlockDefaultsToFrontAndRoundTrips) {
    auto orchestrator = create_hybrid_orchestrator(
        /*num_kv_blocks=*/16,
        /*num_la_blocks=*/4,
        TEST_BLOCK_SIZE,
        /*num_layers=*/1,
        /*la_fixed_blocks_per_seq=*/4);

    const uint64_t seq_id = provision_single_sequence(orchestrator, 410);
    const auto& owned = orchestrator->get_linear_attention_block_table(seq_id);
    ASSERT_GE(owned.size(), 2u);

    EXPECT_EQ(orchestrator->get_linear_attention_live_block(seq_id), owned.front()->get_index());

    const size_t new_live = owned[1]->get_index();
    orchestrator->set_linear_attention_live_block(seq_id, new_live);
    EXPECT_EQ(orchestrator->get_linear_attention_live_block(seq_id), new_live);

    orchestrator->free_sequence(seq_id);
}

/// Non-speculative live-row reads must track current prefill row after copy-on-write.
TEST(TestCacheOrchestratorHybrid, LinearAttentionWorkspace_DefaultLiveBlockTracksReallocatedPrefillRow) {
    auto orchestrator = create_hybrid_orchestrator(
        /*num_kv_blocks=*/16,
        /*num_la_blocks=*/4,
        TEST_BLOCK_SIZE,
        /*num_layers=*/1,
        /*la_fixed_blocks_per_seq=*/1);

    auto seq_group = create_sequence_group(420, /*num_sequences=*/1);
    auto parent = seq_group->get_running_sequences()[0];
    const uint64_t parent_id = parent->get_id();
    orchestrator->allocate_tokens(parent, seq_group, 1, seq_group->get_prompt_len());

    const size_t original_prefill = orchestrator->get_linear_attention_live_block(parent_id);
    auto child = seq_group->fork_sequence(parent);
    const uint64_t child_id = child->get_id();
    orchestrator->fork_sequence(parent_id, child_id);

    seq_group->schedule_tokens(1);
    orchestrator->append_slots(seq_group);

    const auto& parent_owned = orchestrator->get_linear_attention_block_table(parent_id);
    EXPECT_EQ(parent_owned.size(), 1u);
    if (parent_owned.size() == 1u) {
        const size_t reallocated_prefill = parent_owned.front()->get_index();
        EXPECT_NE(reallocated_prefill, original_prefill);
        EXPECT_EQ(orchestrator->get_linear_attention_live_block(parent_id), reallocated_prefill);
    }

    bool extra_child_forked = false;
    try {
        orchestrator->fork_sequence(parent_id, /*child_id=*/9003);
        extra_child_forked = true;
    } catch (const std::exception& ex) {
        ADD_FAILURE() << ex.what();
    }

    if (extra_child_forked) {
        orchestrator->free_sequence(9003);
    }
    orchestrator->free_sequence(child_id);
    orchestrator->free_sequence(parent_id);
}

/// Scratch rows are owned LA rows excluding the live row.
TEST(TestCacheOrchestratorHybrid, LinearAttentionWorkspace_ScratchBlocksAreOwnedSetMinusLive) {
    const size_t n = 3;
    const size_t fixed_blocks = 1 + n;
    auto orchestrator = create_hybrid_orchestrator(
        /*num_kv_blocks=*/16,
        /*num_la_blocks=*/fixed_blocks,
        TEST_BLOCK_SIZE,
        /*num_layers=*/1,
        /*la_fixed_blocks_per_seq=*/fixed_blocks);

    const uint64_t seq_id = provision_single_sequence(orchestrator, 430);

    const auto& owned = orchestrator->get_linear_attention_block_table(seq_id);
    std::set<size_t> owned_indices;
    for (const auto& block : owned) {
        owned_indices.insert(block->get_index());
    }
    ASSERT_EQ(owned_indices.size(), fixed_blocks);

    const size_t initial_live = orchestrator->get_linear_attention_live_block(seq_id);
    auto scratch = linear_attention_scratch_blocks(orchestrator, seq_id);
    EXPECT_EQ(scratch.size(), n);

    std::set<size_t> scratch_set(scratch.begin(), scratch.end());
    EXPECT_EQ(scratch_set.count(initial_live), 0u);
    std::set<size_t> reconstructed = scratch_set;
    reconstructed.insert(initial_live);
    EXPECT_EQ(reconstructed, owned_indices);

    const size_t promoted = *scratch_set.begin();
    orchestrator->set_linear_attention_live_block(seq_id, promoted);
    auto scratch_after = linear_attention_scratch_blocks(orchestrator, seq_id);
    std::set<size_t> scratch_after_set(scratch_after.begin(), scratch_after.end());

    EXPECT_EQ(scratch_after.size(), n);
    EXPECT_EQ(scratch_after_set.count(promoted), 0u);
    EXPECT_EQ(scratch_after_set.count(initial_live), 1u);

    orchestrator->free_sequence(seq_id);
}

/// Live LA registry survives cleanup while sequence runs.
TEST(TestCacheOrchestratorHybrid, LinearAttentionWorkspace_OwnedSetNotReapedByCleanup) {
    const size_t fixed_blocks = 1 + 3;
    auto orchestrator = create_hybrid_orchestrator(
        /*num_kv_blocks=*/16,
        /*num_la_blocks=*/fixed_blocks,
        TEST_BLOCK_SIZE,
        /*num_layers=*/1,
        /*la_fixed_blocks_per_seq=*/fixed_blocks);

    auto seq_group = create_sequence_group(440, /*num_sequences=*/1);
    auto seq = seq_group->get_running_sequences()[0];
    orchestrator->allocate_tokens(seq, seq_group, 1, seq_group->get_prompt_len());
    const uint64_t seq_id = seq->get_id();

    const auto owned_before = orchestrator->get_linear_attention_block_table(seq_id);
    ASSERT_EQ(owned_before.size(), fixed_blocks);
    const size_t promoted = owned_before[2]->get_index();
    orchestrator->set_linear_attention_live_block(seq_id, promoted);

    orchestrator->free_empty_physical_blocks(seq_group);

    const auto& owned_after = orchestrator->get_linear_attention_block_table(seq_id);
    EXPECT_EQ(owned_after.size(), fixed_blocks);
    for (size_t i = 0; i < owned_after.size(); ++i) {
        EXPECT_EQ(owned_after[i]->get_index(), owned_before[i]->get_index());
    }
    EXPECT_EQ(orchestrator->get_linear_attention_live_block(seq_id), promoted);

    orchestrator->free_sequence(seq_id);
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

    orchestrator->allocate_tokens(victim_seq, victim, TEST_BLOCK_SIZE + 1, victim->get_prompt_len());

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

    // Target already owns LA state; only KV has token demand below.
    orchestrator->allocate_tokens(victim_seq, victim, TEST_BLOCK_SIZE * 2 + 1, victim->get_prompt_len());
    orchestrator->allocate_tokens(target_seq, target, 1, target->get_prompt_len());

    target->schedule_tokens(TEST_BLOCK_SIZE + 1);

    EXPECT_FALSE(orchestrator->can_partially_preempt(victim, target));

    orchestrator->free_sequence(victim_seq->get_id());
    orchestrator->free_sequence(target_seq->get_id());
}

/// Live LA block must belong to the sequence-owned table.
TEST(TestCacheOrchestratorHybrid, LinearAttentionWorkspace_SetLiveBlockRejectsNonOwnedBlock) {
    const size_t fixed_blocks = 1 + 3;
    auto orchestrator = create_hybrid_orchestrator(
        /*num_kv_blocks=*/16,
        /*num_la_blocks=*/fixed_blocks,
        TEST_BLOCK_SIZE,
        /*num_layers=*/1,
        /*la_fixed_blocks_per_seq=*/fixed_blocks);

    const uint64_t seq_id = provision_single_sequence(orchestrator, 450);

    const auto& owned = orchestrator->get_linear_attention_block_table(seq_id);
    std::set<size_t> owned_indices;
    for (const auto& block : owned) {
        owned_indices.insert(block->get_index());
    }
    size_t non_owned = 0;
    while (owned_indices.count(non_owned) != 0) {
        ++non_owned;
    }
    EXPECT_THROW(orchestrator->set_linear_attention_live_block(seq_id, non_owned), ov::Exception);

    EXPECT_NO_THROW(orchestrator->set_linear_attention_live_block(seq_id, *owned_indices.begin()));

    orchestrator->free_sequence(seq_id);
}

/// Fork is rejected after live LA row moves off prefill row.
TEST(TestCacheOrchestratorHybrid, LinearAttentionWorkspace_ForkRejectedWhenLiveRowMovedOffPrefill) {
    const size_t fixed_blocks = 1 + 3;
    auto orchestrator = create_hybrid_orchestrator(
        /*num_kv_blocks=*/16,
        /*num_la_blocks=*/2 * fixed_blocks,
        TEST_BLOCK_SIZE,
        /*num_layers=*/1,
        /*la_fixed_blocks_per_seq=*/fixed_blocks);

    const uint64_t parent_id = provision_single_sequence(orchestrator, 460);
    const auto& owned = orchestrator->get_linear_attention_block_table(parent_id);

    // Live at prefill row: fork allowed.
    EXPECT_EQ(orchestrator->get_linear_attention_live_block(parent_id), owned.front()->get_index());
    EXPECT_NO_THROW(orchestrator->fork_sequence(parent_id, /*child_id=*/9001));
    orchestrator->free_sequence(9001);

    // Live off prefill row: fork rejected.
    orchestrator->set_linear_attention_live_block(parent_id, owned.back()->get_index());
    EXPECT_THROW(orchestrator->fork_sequence(parent_id, /*child_id=*/9002), ov::Exception);

    orchestrator->free_sequence(parent_id);
}
