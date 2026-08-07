// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>
#include <set>
#include "openvino/runtime/core.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"
#include "sequence_group.hpp"
#include "continuous_batching/scheduler.hpp"
#include "continuous_batching/pipeline_impl.hpp"
#include "continuous_batching/cache/cache_orchestrator.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "continuous_batching/cache/linear_attention_cache_manager.hpp"
#include "helper.hpp"
#include "utils.hpp"

using namespace ov::genai;

void clear_finished_sequences(std::vector<SequenceGroup::Ptr>& requests) {
    auto new_end = std::remove_if(requests.begin(), requests.end(), [] (SequenceGroup::CPtr seq_group) -> bool {
            return seq_group->has_finished();
    });
    requests.erase(new_end, requests.end());
}

static constexpr size_t TEST_BLOCK_SIZE = 4;
static constexpr size_t TEST_NUM_DECODER_LAYERS = 12;
static constexpr size_t TEST_DEFAULT_CACHE_INTERVAL = TEST_BLOCK_SIZE * DEFAULT_LINEAR_ATTENTION_CACHE_INTERVAL_MULTIPLIER;
static constexpr size_t TEST_CUSTOM_CACHE_INTERVAL_MULTIPLIER = 16;
static constexpr size_t TEST_CUSTOM_CACHE_INTERVAL = TEST_BLOCK_SIZE * TEST_CUSTOM_CACHE_INTERVAL_MULTIPLIER;

size_t get_test_cache_interval(const SchedulerConfig& scheduler_config, size_t kv_block_size = TEST_BLOCK_SIZE) {
    return scheduler_config.get_cache_interval(kv_block_size);
}

std::shared_ptr<CacheOrchestrator> init_cache_orchestrator(SchedulerConfig scheduler_config, size_t block_size = TEST_BLOCK_SIZE, size_t num_layers = 1) {
    ov::Core core = ov::Core();
    ov::InferRequest request = core.compile_model(get_dummy_model(core, TEST_NUM_DECODER_LAYERS)).create_infer_request();
    auto cache_manager = std::make_unique<KVCacheManager>(request);
    auto block_manager = std::make_unique<BlockManager>(scheduler_config.num_kv_blocks, scheduler_config.enable_prefix_caching, block_size, num_layers);
    auto orchestrator = std::make_shared<CacheOrchestrator>();
    orchestrator->register_cache_type(CacheType::KV_CACHE, std::move(cache_manager), std::move(block_manager));
    return orchestrator;
}

// cap_la_pool mirrors an explicitly configured LA block ceiling.
std::shared_ptr<CacheOrchestrator> init_hybrid_cache_orchestrator(SchedulerConfig scheduler_config,
                                                                   size_t kv_block_size = TEST_BLOCK_SIZE,
                                                                   size_t kv_num_layers = 1,
                                                                   size_t la_num_layers = 1,
                                                                   bool cap_la_pool = false) {
    ov::Core core = ov::Core();
    ov::InferRequest request = core.compile_model(get_dummy_hybrid_model(core, kv_num_layers, la_num_layers)).create_infer_request();

    auto kv_cache_manager = std::make_unique<KVCacheManager>(request);
    auto kv_block_manager = std::make_unique<BlockManager>(scheduler_config.num_kv_blocks,
                                                           scheduler_config.enable_prefix_caching,
                                                           kv_block_size,
                                                           kv_num_layers);

    auto la_cache_manager = std::make_unique<LinearAttentionCacheManager>(request);
    std::unique_ptr<BlockManager> la_block_manager;
    if (scheduler_config.enable_prefix_caching) {
        la_block_manager = std::make_unique<BlockManager>(scheduler_config.num_linear_attention_blocks,
                                                          true,
                                                          get_test_cache_interval(scheduler_config, kv_block_size),
                                                          1,
                                                          0,
                                                          true);
    } else {
        const size_t num_la_blocks = scheduler_config.num_linear_attention_blocks > 0
                                         ? scheduler_config.num_linear_attention_blocks
                                         : (scheduler_config.num_kv_blocks > 0 ? scheduler_config.max_num_seqs : 0);
        // One live row per sequence, mirroring CacheOrchestrator::register_linear_attention_cache.
        // One committed row per sequence; speculative scratch is borrowed per step.
        la_block_manager = std::make_unique<BlockManager>(num_la_blocks,
                                                          false,
                                                          1,
                                                          1,  // one logical block table for all LA layers
                                                          /*fixed_blocks_per_sequence=*/1,
                                                          /*restore_latest_prefix_block_only=*/false,
                                                          /*max_total_blocks=*/cap_la_pool ? num_la_blocks : 0);
    }

    auto orchestrator = std::make_shared<CacheOrchestrator>();
    orchestrator->register_cache_type(CacheType::KV_CACHE, std::move(kv_cache_manager), std::move(kv_block_manager));

    orchestrator->register_cache_type(CacheType::LINEAR_ATTENTION_CACHE,
                                      std::move(la_cache_manager),
                                      std::move(la_block_manager));
    return orchestrator;
}


std::shared_ptr<CacheOrchestrator> init_linear_attention_cache_orchestrator(SchedulerConfig scheduler_config,
                                                                            size_t la_num_layers = 1) {
    ov::Core core = ov::Core();
    ov::InferRequest request = core.compile_model(get_dummy_hybrid_model(core, 0, la_num_layers)).create_infer_request();

    auto la_cache_manager = std::make_unique<LinearAttentionCacheManager>(request);
    auto la_block_manager = std::make_unique<BlockManager>(scheduler_config.num_linear_attention_blocks,
                                                           false,
                                                           1,
                                                           1,
                                                           1);

    auto orchestrator = std::make_shared<CacheOrchestrator>();
    orchestrator->register_cache_type(CacheType::LINEAR_ATTENTION_CACHE,
                                      std::move(la_cache_manager),
                                      std::move(la_block_manager));
    return orchestrator;
}

struct HybridCreateContext {
    ov::InferRequest request;
    size_t kv_block_size = 0;
    size_t kv_block_size_in_bytes = 0;
    size_t la_block_size_in_bytes = 0;
};

HybridCreateContext create_hybrid_create_context(size_t kv_num_layers = 1, size_t la_num_layers = 1) {
    ov::Core core;
    ov::InferRequest request = core.compile_model(get_dummy_hybrid_model(core, kv_num_layers, la_num_layers)).create_infer_request();
    auto kv_cache_manager = std::make_shared<KVCacheManager>(request);
    auto la_cache_manager = std::make_shared<LinearAttentionCacheManager>(request);
    return {
        request,
        kv_cache_manager->get_block_size(),
        kv_cache_manager->get_block_size_in_bytes(),
        la_cache_manager->get_block_size_in_bytes(),
    };
}

ov::Tensor embeds_matrix_to_tensor(std::vector<std::vector<float>> vec) {
    size_t hidden_size = vec[0].size();
    ov::Tensor res = ov::Tensor(ov::element::f32, {1, vec.size(), hidden_size});
    auto res_data = res.data<float>();
    size_t pos = 0;
    for (size_t i = 0; i < vec.size(); i ++) {
        for (size_t j = 0; j < hidden_size; j++) {
            res_data[pos++] = vec[i][j];
        }
    }
    return res;
}

TEST(TestScheduler, adaptive_rkv_zero_size_is_not_marked_available) {
    Scheduler::Output output;
    const uint64_t seq_id = 42;

    EXPECT_FALSE(output.has_adaptive_rkv_evictable_size(seq_id));
    EXPECT_EQ(output.get_adaptive_rkv_evictable_size(seq_id), 0);

    output.set_adaptive_rkv_evictable_size(seq_id, 0);
    EXPECT_FALSE(output.has_adaptive_rkv_evictable_size(seq_id));
    EXPECT_EQ(output.get_adaptive_rkv_evictable_size(seq_id), 0);

    output.set_adaptive_rkv_evictable_size(seq_id, 3);
    EXPECT_TRUE(output.has_adaptive_rkv_evictable_size(seq_id));
    EXPECT_EQ(output.get_adaptive_rkv_evictable_size(seq_id), 3);

    output.set_adaptive_rkv_evictable_size(seq_id, 0);
    EXPECT_FALSE(output.has_adaptive_rkv_evictable_size(seq_id));
    EXPECT_EQ(output.get_adaptive_rkv_evictable_size(seq_id), 0);
}

TEST(TestScheduler, output_keeps_shared_kv_global_data_alive) {
    Scheduler::Output output;
    auto mutable_global_data = std::make_shared<Scheduler::KVPagedAttentionGlobalData>();
    mutable_global_data->xattention_block_size = 17;
    mutable_global_data->xattention_stride = 5;
    mutable_global_data->adaptive_rkv_start_size = 3;

    std::shared_ptr<const Scheduler::KVPagedAttentionGlobalData> global_data = mutable_global_data;
    output.set_kv_paged_attention_global_data(global_data);
    mutable_global_data.reset();
    global_data.reset();

    const Scheduler::KVPagedAttentionGlobalData& output_global_data = output.get_kv_paged_attention_global_data();
    EXPECT_EQ(output_global_data.xattention_block_size, 17);
    EXPECT_EQ(output_global_data.xattention_stride, 5);
    EXPECT_EQ(output_global_data.adaptive_rkv_start_size, 3);
}

TEST(TestScheduler, general_test) {
    std::array<SchedulerConfig, 2> configs = {SchedulerConfig(), SchedulerConfig()};
    configs.at(0).max_num_batched_tokens = 32;
    configs.at(0).num_kv_blocks = 6;
    configs.at(0).dynamic_split_fuse = false;
    configs.at(0).max_num_seqs = 5;
    configs.at(1).max_num_batched_tokens = 32;
    configs.at(1).num_kv_blocks = 6;
    configs.at(1).dynamic_split_fuse = true;
    configs.at(1).max_num_seqs = 5;
    for (auto scheduler_config: configs) {
        std::vector<uint64_t> tokens = {0,1,2,3,4,5,6,7};
        SequenceGroup::Ptr sequence_group1 = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                utils::get_greedy_config());
        auto idx0 = (*sequence_group1)[0]->get_id();
        SequenceGroup::Ptr sequence_group2 = std::make_shared<SequenceGroup>(1, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                utils::get_greedy_config());
        auto idx1 = (*sequence_group2)[0]->get_id();
        SequenceGroup::Ptr sequence_group3 = std::make_shared<SequenceGroup>(1, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                utils::get_greedy_config());
        auto idx2 = (*sequence_group3)[0]->get_id();
        std::vector<SequenceGroup::Ptr> requests = {sequence_group1, sequence_group2, sequence_group3};

        // schedule 3 sequence groups that use 6 kv blocks
        Scheduler scheduler = Scheduler(init_cache_orchestrator(scheduler_config), scheduler_config);
        auto out1 = scheduler.schedule(requests);

        std::vector<uint64_t> ref_ids = {0, 1, 2};
        EXPECT_EQ(out1.m_scheduled_sequence_groups_ids, ref_ids);
        EXPECT_EQ(out1.get_kv_block_tables(idx0)[0].size(), 2);
        EXPECT_EQ(out1.get_kv_block_tables(idx1)[0].size(), 2);
        EXPECT_EQ(out1.get_kv_block_tables(idx2)[0].size(), 2);
        // tokens.size() * 2 tokens should be scheduled on prompt phase, corresponding to first three sequences
        EXPECT_EQ(out1.m_total_num_scheduled_tokens, tokens.size() * 3);
        EXPECT_EQ(out1.is_prompt, !scheduler_config.dynamic_split_fuse);

        for (auto seq: requests) {
            std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
            // prompt phase
            seq->finish_iteration();
        }

        // at this point we scheduled all available kv blocks

        // sequence_group3 should be evicted
        auto out3 = scheduler.schedule(requests);

        for (auto seq: requests) {
            std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
            // generate phase, append a token to each sequence
            running_sequences[0]->append_token(16, 0.9);
            seq->finish_iteration();
        }

        std::vector<uint64_t> ref_ids2 = {0, 1};
        EXPECT_EQ(out3.m_scheduled_sequence_groups_ids, ref_ids2);
        EXPECT_EQ(out3.get_kv_block_tables(idx0)[0].size(), 3);
        EXPECT_EQ(out3.get_kv_block_tables(idx1)[0].size(), 3);
        // 2 tokens should be scheduled on generate phase for "0" and "1" sequence, "2" sequence should be preempted
        EXPECT_EQ(out3.m_total_num_scheduled_tokens, 2);
        EXPECT_FALSE(out3.is_prompt);

        // check that scheduler has no block table for sequence_group3
        EXPECT_FALSE(scheduler.has_block_table(idx2));

        // finish first sequence
        requests[0]->get_running_sequences()[0]->set_status(SequenceStatus::FINISHED);
        scheduler.free_sequence(idx0);
        clear_finished_sequences(requests);
        // KV blocks 0,1,5 are free now


        auto out4 = scheduler.schedule(requests);

        // check that sequence_group3 is fully scehuled
        EXPECT_EQ(out4.get_kv_block_tables(idx2)[0].size(), 2);
        EXPECT_FALSE(out4.get_kv_block_tables(idx2)[0][0]->is_free());
        EXPECT_EQ(out4.get_kv_block_tables(idx2)[0][0]->get_index(), 0);
        EXPECT_FALSE(out4.get_kv_block_tables(idx2)[0][1]->is_free());
        EXPECT_EQ(out4.get_kv_block_tables(idx2)[0][1]->get_index(), 1);

        // requests1[1] should be fully scheduled plus 1 slot for requests[0] for generate phase
        EXPECT_EQ(out4.m_total_num_scheduled_tokens, requests[1]->get_context_len() + 1);
        EXPECT_EQ(out4.is_prompt, false);

        for (auto& req : requests) {
            for (auto& seq : req->get_sequences()) {
                scheduler.free_sequence(seq->get_id());
            }
        }
    }

}

TEST(TestScheduler, hybrid_output_fills_linear_attention_block_table_in_prompt_and_generate) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 8;
    scheduler_config.num_linear_attention_blocks = 8;
    scheduler_config.dynamic_split_fuse = true;
    scheduler_config.max_num_seqs = 8;

    std::vector<uint64_t> tokens = {0, 1, 2, 3};
    SequenceGroup::Ptr seq_group1 = std::make_shared<SequenceGroup>(0,
                                                                     ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                     utils::get_greedy_config());
    SequenceGroup::Ptr seq_group2 = std::make_shared<SequenceGroup>(1,
                                                                     ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                     utils::get_greedy_config());
    auto seq_id1 = seq_group1->get_running_sequences()[0]->get_id();
    auto seq_id2 = seq_group2->get_running_sequences()[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {seq_group1, seq_group2};

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                       TEST_BLOCK_SIZE,
                                                       /*kv_num_layers=*/1,
                                                       /*la_num_layers=*/3);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    auto prompt_out = scheduler.schedule(requests);

    EXPECT_EQ(orchestrator->get_cache_manager(CacheType::LINEAR_ATTENTION_CACHE).get_num_layers(), 3);
    EXPECT_EQ(orchestrator->get_cache_manager(CacheType::LINEAR_ATTENTION_CACHE).get_num_cache_tensors(), 6);
    ASSERT_EQ(prompt_out.get_kv_block_tables(seq_id1).size(), 1);
    ASSERT_EQ(prompt_out.get_kv_block_tables(seq_id2).size(), 1);
    EXPECT_EQ(prompt_out.get_kv_block_tables(seq_id1)[0].size(), 1);
    EXPECT_EQ(prompt_out.get_kv_block_tables(seq_id2)[0].size(), 1);
    EXPECT_TRUE(prompt_out.has_linear_attention_paging_data(seq_id1));
    EXPECT_TRUE(prompt_out.has_linear_attention_paging_data(seq_id2));
    EXPECT_EQ(prompt_out.get_linear_attention_paging_data(seq_id1).block_indices.size(), 2);
    EXPECT_EQ(prompt_out.get_linear_attention_paging_data(seq_id1).block_indices[0], prompt_out.get_linear_attention_paging_data(seq_id1).block_indices[1]);
    EXPECT_EQ(prompt_out.get_linear_attention_paging_data(seq_id2).block_indices.size(), 2);
    EXPECT_EQ(prompt_out.get_linear_attention_paging_data(seq_id2).block_indices[0], prompt_out.get_linear_attention_paging_data(seq_id2).block_indices[1]);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id1).size(), 1);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id2).size(), 1);

    for (auto& req : requests) {
        auto running = req->get_running_sequences();
        running[0]->append_token(42, 0.9f);
        req->finish_iteration();
    }

    auto gen_out = scheduler.schedule(requests);
    EXPECT_TRUE(gen_out.has_linear_attention_paging_data(seq_id1));
    EXPECT_TRUE(gen_out.has_linear_attention_paging_data(seq_id2));
    EXPECT_EQ(gen_out.get_linear_attention_paging_data(seq_id1).block_indices.size(), 2);
    EXPECT_EQ(gen_out.get_linear_attention_paging_data(seq_id1).block_indices[0], gen_out.get_linear_attention_paging_data(seq_id1).block_indices[1]);
    EXPECT_EQ(gen_out.get_linear_attention_paging_data(seq_id2).block_indices.size(), 2);
    EXPECT_EQ(gen_out.get_linear_attention_paging_data(seq_id2).block_indices[0], gen_out.get_linear_attention_paging_data(seq_id2).block_indices[1]);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id1).size(), 1);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id2).size(), 1);

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, hybrid_non_prefix_linear_attention_returns_aliased_read_write_blocks) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 8;
    scheduler_config.num_linear_attention_blocks = 8;
    scheduler_config.enable_prefix_caching = false;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 8;

    std::vector<uint64_t> tokens = {0, 1, 2, 3};
    SequenceGroup::Ptr seq_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    const auto seq_id = seq_group->get_running_sequences()[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {seq_group};

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    auto out = scheduler.schedule(requests);

    ASSERT_TRUE(out.has_linear_attention_paging_data(seq_id));
    const auto& paging_data = out.get_linear_attention_paging_data(seq_id);
    ASSERT_EQ(paging_data.block_indices.size(), 2);
    EXPECT_EQ(paging_data.block_indices[0], paging_data.block_indices[1]);
    EXPECT_EQ(paging_data.cache_interval, 0);
    EXPECT_EQ(paging_data.past_length, 0);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id).size(), 1);

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, hybrid_non_prefix_linear_attention_speculative_window_is_atomic_or_deferred_under_megabatch_pressure) {
    // The megabatch fits one full validation window; the second must defer rather than run partially.
    constexpr size_t N = 3;
    constexpr size_t WINDOW = N + 1;  // 4
    SchedulerConfig scheduler_config;
    // Budget fits one full window (4) plus a partial second (2) -- never the full second window.
    scheduler_config.max_num_batched_tokens = WINDOW + (WINDOW - 2);  // 6
    scheduler_config.num_kv_blocks = 32;
    scheduler_config.num_linear_attention_blocks = 16;
    scheduler_config.enable_prefix_caching = false;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 4;

    std::vector<uint64_t> tokens = {0, 1, 2, 3};
    SequenceGroup::Ptr seq_group_a = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    SequenceGroup::Ptr seq_group_b = std::make_shared<SequenceGroup>(
        1,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    const auto seq_id_a = seq_group_a->get_running_sequences()[0]->get_id();
    const auto seq_id_b = seq_group_b->get_running_sequences()[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {seq_group_a, seq_group_b};

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                       TEST_BLOCK_SIZE,
                                                       /*kv_num_layers=*/1,
                                                       /*la_num_layers=*/1);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    // Two committed rows plus one concurrent window.
    scheduler.ensure_linear_attention_pool_blocks(2 + (1 + N));

    // Process both prompts before testing validation-window scheduling.
    while (seq_group_a->get_num_processed_tokens() < tokens.size() ||
           seq_group_b->get_num_processed_tokens() < tokens.size()) {
        std::ignore = scheduler.schedule(requests);
        for (auto& req : requests) {
            if (req->is_scheduled()) {
                req->finish_iteration();
            }
        }
    }

    for (auto& req : requests) {
        req->get_running_sequences()[0]->append_token(42, 0.9f);
        req->update_processed_tokens_num(tokens.size());
        req->set_num_validated_tokens(N);
    }

    const size_t committed_b = orchestrator->get_linear_attention_live_block(seq_id_b);

    auto out1 = scheduler.schedule(requests);
    ASSERT_TRUE(out1.has_linear_attention_paging_data(seq_id_a));
    EXPECT_EQ(out1.get_linear_attention_paging_data(seq_id_a).block_indices.size(), N + 2);
    EXPECT_TRUE(out1.get_linear_attention_paging_data(seq_id_a).is_speculative);
    EXPECT_EQ(seq_group_a->get_num_scheduled_tokens(), WINDOW);

    EXPECT_EQ(seq_group_b->get_num_scheduled_tokens(), 0u);
    EXPECT_FALSE(out1.has_linear_attention_paging_data(seq_id_b));
    EXPECT_EQ(out1.m_scheduled_sequence_groups_ids, std::vector<uint64_t>({0}));

    scheduler.release_linear_attention_checkpoints(seq_id_a);
    seq_group_a->finish_iteration();

    auto out2 = scheduler.schedule(requests);
    ASSERT_TRUE(out2.has_linear_attention_paging_data(seq_id_b));
    const auto& paging_b = out2.get_linear_attention_paging_data(seq_id_b);
    ASSERT_EQ(paging_b.block_indices.size(), N + 2);
    EXPECT_EQ(seq_group_b->get_num_scheduled_tokens(), WINDOW);
    EXPECT_EQ(static_cast<size_t>(paging_b.block_indices[0]), committed_b);
    std::set<int32_t> seen_b = {paging_b.block_indices[0]};
    for (size_t i = 1; i < paging_b.block_indices.size(); ++i) {
        EXPECT_TRUE(seen_b.insert(paging_b.block_indices[i]).second) << "duplicate borrowed row " << i;
    }
    EXPECT_EQ(paging_b.cache_interval, 1);
    EXPECT_TRUE(paging_b.is_speculative);

    scheduler.release_linear_attention_checkpoints(seq_id_b);
    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, hybrid_non_prefix_linear_attention_speculative_window_too_large_for_megabatch_asserts) {
    // A validation window that can never fit the megabatch must fail instead of deferring forever.
    constexpr size_t N = 5;  // window N+1 = 6 > max_num_batched_tokens
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = N;  // strictly less than N+1
    scheduler_config.num_kv_blocks = 32;
    scheduler_config.num_linear_attention_blocks = 16;
    scheduler_config.enable_prefix_caching = false;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 4;

    std::vector<uint64_t> tokens = {0, 1, 2, 3};
    SequenceGroup::Ptr seq_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    const auto seq_id = seq_group->get_running_sequences()[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {seq_group};

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                       TEST_BLOCK_SIZE,
                                                       /*kv_num_layers=*/1,
                                                       /*la_num_layers=*/1);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    scheduler.ensure_linear_attention_pool_blocks(1 + (1 + N));

    std::ignore = scheduler.schedule(requests);
    seq_group->finish_iteration();

    seq_group->get_running_sequences()[0]->append_token(42, 0.9f);
    seq_group->update_processed_tokens_num(tokens.size());
    seq_group->set_num_validated_tokens(N);

    EXPECT_THROW(std::ignore = scheduler.schedule(requests), ov::Exception);

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

namespace {
Scheduler::Output run_one_speculative_step(Scheduler& scheduler,
                                           std::vector<SequenceGroup::Ptr>& requests) {
    return scheduler.schedule(requests);
}
}  // namespace

TEST(TestScheduler, hybrid_non_prefix_linear_attention_uses_full_preemption_for_fixed_size_victim_state) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 4;
    scheduler_config.num_linear_attention_blocks = 2;
    scheduler_config.enable_prefix_caching = false;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 8;

    std::vector<uint64_t> tokens = {0, 1, 2, 3};
    SequenceGroup::Ptr seq_group1 = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    SequenceGroup::Ptr seq_group2 = std::make_shared<SequenceGroup>(
        1,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    const auto seq_id1 = seq_group1->get_running_sequences()[0]->get_id();
    const auto seq_id2 = seq_group2->get_running_sequences()[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {seq_group1, seq_group2};

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    auto prompt_out = scheduler.schedule(requests);

    EXPECT_EQ(prompt_out.m_scheduled_sequence_groups_ids.size(), 2);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id1).size(), 1);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id2).size(), 1);

    for (auto& req : requests) {
        req->finish_iteration();
    }

    for (size_t step = 0; step < TEST_BLOCK_SIZE; ++step) {
        std::ignore = scheduler.schedule(requests);
        for (auto& req : requests) {
            req->get_running_sequences()[0]->append_token(42, 0.9f);
            req->finish_iteration();
        }
    }

    auto gen_out = scheduler.schedule(requests);

    EXPECT_EQ(gen_out.m_scheduled_sequence_groups_ids, std::vector<uint64_t>({0}));
    EXPECT_FALSE(scheduler.has_block_table(seq_id2));
    EXPECT_EQ(seq_group2->get_num_processed_tokens(), 0);

    scheduler.free_sequence(seq_id1);
}

TEST(TestScheduler, hybrid_admission_when_la_pool_is_bottleneck) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 8;
    scheduler_config.num_linear_attention_blocks = 1;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 8;

    std::vector<uint64_t> tokens = {0, 1, 2, 3};
    SequenceGroup::Ptr seq_group1 = std::make_shared<SequenceGroup>(0,
                                                                     ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                     utils::get_greedy_config());
    SequenceGroup::Ptr seq_group2 = std::make_shared<SequenceGroup>(1,
                                                                     ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                     utils::get_greedy_config());
    auto seq_id1 = seq_group1->get_running_sequences()[0]->get_id();
    auto seq_id2 = seq_group2->get_running_sequences()[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {seq_group1, seq_group2};

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    auto out = scheduler.schedule(requests);

    EXPECT_EQ(out.m_scheduled_sequence_groups_ids.size(), 1);
    EXPECT_TRUE(out.has_linear_attention_paging_data(seq_id1) || out.has_linear_attention_paging_data(seq_id2));

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, hybrid_initialize_cache_grows_fixed_size_by_total_concurrent_sequences) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 0;
    scheduler_config.cache_size = 0;
    scheduler_config.num_linear_attention_blocks = 0;
    scheduler_config.dynamic_split_fuse = true;
    scheduler_config.max_num_seqs = 8;

    std::vector<uint64_t> tokens = {0, 1, 2, 3};
    SequenceGroup::Ptr seq_group1 = std::make_shared<SequenceGroup>(0,
                                                                     ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                     utils::get_greedy_config());
    SequenceGroup::Ptr seq_group2 = std::make_shared<SequenceGroup>(1,
                                                                     ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                     utils::get_greedy_config());
    std::vector<SequenceGroup::Ptr> requests = {seq_group1, seq_group2};

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);

    EXPECT_EQ(orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE).get_total_block_count(), 0);
    std::ignore = scheduler.schedule(requests);
    EXPECT_EQ(orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE).get_total_block_count(), 2);

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, initialize_cache_uses_sequence_aware_block_rounding) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 0;
    scheduler_config.cache_size = 0;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 8;

    std::vector<SequenceGroup::Ptr> requests;
    for (size_t request_id = 0; request_id < 4; ++request_id) {
        std::vector<uint64_t> tokens = {request_id};
        requests.push_back(std::make_shared<SequenceGroup>(
            request_id,
            ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
            utils::get_greedy_config()));
    }

    auto orchestrator = init_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);

    std::ignore = scheduler.schedule(requests);

    EXPECT_EQ(orchestrator->get_block_manager(CacheType::KV_CACHE).get_total_block_count(), requests.size());

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, dynamic_alloc_reserves_full_capacity_for_later_larger_prompt) {
    // In dynamic allocation mode, a prompt that arrives after the cache was sized for a smaller
    // one must have its capacity reserved in a single scheduling round (grow-to-fit), rather than
    // growing m_cache_growth_num_tokens at a time across the prefill (which reallocates the whole
    // cache each step). See openvinotoolkit/openvino.genai#3968.
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;  // small batch => prompt is chunked across steps
    scheduler_config.num_kv_blocks = 0;            // dynamic allocation
    scheduler_config.cache_size = 0;
    scheduler_config.dynamic_split_fuse = true;
    scheduler_config.max_num_seqs = 8;

    auto orchestrator = init_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);

    // Round 1: a small prompt sizes the cache for itself.
    std::vector<int64_t> small_tokens = {0, 1, 2, 3};
    SequenceGroup::Ptr small = std::make_shared<SequenceGroup>(
        0, ov::Tensor(ov::element::i64, {small_tokens.size()}, small_tokens.data()),
        utils::get_greedy_config());
    std::vector<SequenceGroup::Ptr> round1 = {small};
    std::ignore = scheduler.schedule(round1);
    for (auto& seq : small->get_sequences()) {
        scheduler.free_sequence(seq->get_id());
    }

    // Round 2: a much larger prompt (well beyond the 256-token growth chunk) arrives.
    const size_t large_prompt_len = 600;
    std::vector<int64_t> large_tokens(large_prompt_len);
    std::iota(large_tokens.begin(), large_tokens.end(), 0);
    SequenceGroup::Ptr large = std::make_shared<SequenceGroup>(
        1, ov::Tensor(ov::element::i64, {large_tokens.size()}, large_tokens.data()),
        utils::get_greedy_config());
    std::vector<SequenceGroup::Ptr> round2 = {large};
    std::ignore = scheduler.schedule(round2);

    // After a single schedule() round, the KV cache must already hold the whole large prompt.
    const size_t blocks = orchestrator->get_block_manager(CacheType::KV_CACHE).get_total_block_count();
    const size_t blocks_needed_for_prompt = (large_prompt_len + TEST_BLOCK_SIZE - 1) / TEST_BLOCK_SIZE;
    EXPECT_GE(blocks, blocks_needed_for_prompt);

    for (auto& seq : large->get_sequences()) {
        scheduler.free_sequence(seq->get_id());
    }
}

TEST(TestScheduler, dynamic_alloc_reserves_capacity_for_new_prompt_on_top_of_running_sequence) {
    // When a large prompt arrives while another sequence is still generating, the reservation must
    // account for the running sequence's footprint too, so the new prompt's capacity is added on
    // top in a single reallocation (not under-reserved). See openvinotoolkit/openvino.genai#3968.
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 0;  // dynamic allocation
    scheduler_config.cache_size = 0;
    scheduler_config.dynamic_split_fuse = true;
    scheduler_config.max_num_seqs = 8;

    auto orchestrator = init_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);

    // Round 1: a small sequence is prefilled and advanced into the generate phase (running).
    std::vector<int64_t> small_tokens = {0, 1, 2, 3};
    SequenceGroup::Ptr small = std::make_shared<SequenceGroup>(
        0, ov::Tensor(ov::element::i64, {small_tokens.size()}, small_tokens.data()),
        utils::get_greedy_config());
    std::vector<SequenceGroup::Ptr> running = {small};
    std::ignore = scheduler.schedule(running);
    small->get_running_sequences()[0]->append_token(42, 0.9f);
    small->finish_iteration();
    ASSERT_TRUE(small->can_generate_tokens());  // now running in generate phase

    // Round 2: a much larger prompt arrives while the small one keeps running.
    const size_t large_prompt_len = 600;
    std::vector<int64_t> large_tokens(large_prompt_len);
    std::iota(large_tokens.begin(), large_tokens.end(), 0);
    SequenceGroup::Ptr large = std::make_shared<SequenceGroup>(
        1, ov::Tensor(ov::element::i64, {large_tokens.size()}, large_tokens.data()),
        utils::get_greedy_config());
    std::vector<SequenceGroup::Ptr> both = {small, large};
    std::ignore = scheduler.schedule(both);

    // Capacity must cover BOTH the running sequence's reservation and the new prompt's, reserved in
    // one round. Targets mirror the scheduler heuristic: min(prompt_len*2, prompt_len + max_new).
    const size_t max_new = utils::get_greedy_config().max_new_tokens;
    auto target_blocks = [&](size_t prompt_len) {
        const size_t tokens = std::min(prompt_len * 2, prompt_len + max_new);
        return (tokens + TEST_BLOCK_SIZE - 1) / TEST_BLOCK_SIZE;
    };
    const size_t blocks = orchestrator->get_block_manager(CacheType::KV_CACHE).get_total_block_count();
    EXPECT_GE(blocks, target_blocks(small_tokens.size()) + target_blocks(large_prompt_len));

    for (auto& req : both) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, linear_attention_only_initializes_fixed_size_capacity) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 0;
    scheduler_config.cache_size = 0;
    scheduler_config.num_linear_attention_blocks = 0;
    scheduler_config.dynamic_split_fuse = true;
    scheduler_config.max_num_seqs = 8;

    std::vector<uint64_t> tokens = {0, 1, 2, 3};
    SequenceGroup::Ptr seq_group = std::make_shared<SequenceGroup>(0,
                                                                    ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                    utils::get_greedy_config());
    const auto seq_id = seq_group->get_running_sequences()[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {seq_group};

    auto orchestrator = init_linear_attention_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);

    EXPECT_FALSE(orchestrator->has_token_capacity());
    auto out = scheduler.schedule(requests);

    EXPECT_EQ(out.m_scheduled_sequence_groups_ids.size(), 1);
    EXPECT_TRUE(out.has_linear_attention_paging_data(seq_id));
    EXPECT_EQ(orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE).get_total_block_count(), 1);

    scheduler.free_sequence(seq_id);
}

TEST(TestScheduler, hybrid_runtime_arrival_beyond_initial_fixed_capacity_schedules_after_growth) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 0;
    scheduler_config.cache_size = 0;
    scheduler_config.num_linear_attention_blocks = 0;
    scheduler_config.dynamic_split_fuse = true;
    scheduler_config.max_num_seqs = 8;

    std::vector<uint64_t> tokens = {0, 1, 2, 3};
    SequenceGroup::Ptr seq_group1 = std::make_shared<SequenceGroup>(0,
                                                                     ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                     utils::get_greedy_config());
    auto seq_id1 = seq_group1->get_running_sequences()[0]->get_id();

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);

    std::vector<SequenceGroup::Ptr> requests = {seq_group1};
    std::ignore = scheduler.schedule(requests);
    EXPECT_EQ(orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE).get_total_block_count(), 1);

    auto running = seq_group1->get_running_sequences();
    running[0]->append_token(42, 0.9f);
    seq_group1->finish_iteration();

    SequenceGroup::Ptr seq_group2 = std::make_shared<SequenceGroup>(1,
                                                                     ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                     utils::get_greedy_config());
    auto seq_id2 = seq_group2->get_running_sequences()[0]->get_id();
    requests.push_back(seq_group2);

    auto out = scheduler.schedule(requests);
    EXPECT_TRUE(out.has_linear_attention_paging_data(seq_id1));
    EXPECT_TRUE(out.has_linear_attention_paging_data(seq_id2));

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            if (scheduler.has_block_table(seq->get_id())) {
                scheduler.free_sequence(seq->get_id());
            }
        }
    }
}

TEST(TestScheduler, hybrid_prefix_caching_prefill_requires_read_and_interval_write_blocks) {
    // Target contract (cache_interval=32):
    // prefill requires 1 read block + ceil((processed % interval + scheduled) / interval) write blocks,
    // with the zero-state read reusing the first write block.
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 512;
    scheduler_config.num_kv_blocks = 128;
    scheduler_config.num_linear_attention_blocks = 32;
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 8;

    // Prompt length 260 => write blocks = ceil((0 + 260) / 32) = 9, plus one read block => 10 total.
    std::vector<uint64_t> tokens(260);
    std::iota(tokens.begin(), tokens.end(), 0);
    SequenceGroup::Ptr seq_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    const auto seq_id = seq_group->get_running_sequences()[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {seq_group};

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    auto out = scheduler.schedule(requests);

    ASSERT_TRUE(out.has_linear_attention_paging_data(seq_id));
    const auto& paging_data = out.get_linear_attention_paging_data(seq_id);
    ASSERT_EQ(paging_data.block_indices.size(), 10);
    EXPECT_EQ(paging_data.past_length, 0);
    EXPECT_EQ(paging_data.cache_interval, TEST_DEFAULT_CACHE_INTERVAL);
    EXPECT_EQ(paging_data.block_indices[0], paging_data.block_indices[1]);
    EXPECT_EQ(std::set<int32_t>(paging_data.block_indices.begin(), paging_data.block_indices.end()).size(), 9);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id).size(), 9);

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, hybrid_prefix_caching_prefill_uses_scheduler_config_cache_interval_multiplier) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 128;
    scheduler_config.num_kv_blocks = 64;
    scheduler_config.num_linear_attention_blocks = 16;
    scheduler_config.cache_interval_multiplier = TEST_CUSTOM_CACHE_INTERVAL_MULTIPLIER;
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 4;

    std::vector<uint64_t> tokens(96);
    std::iota(tokens.begin(), tokens.end(), 0);
    SequenceGroup::Ptr seq_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    const auto seq_id = seq_group->get_running_sequences()[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {seq_group};

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    auto out = scheduler.schedule(requests);

    ASSERT_TRUE(out.has_linear_attention_paging_data(seq_id));
    const auto& paging_data = out.get_linear_attention_paging_data(seq_id);
    ASSERT_EQ(paging_data.block_indices.size(), 3);
    EXPECT_EQ(paging_data.cache_interval, TEST_CUSTOM_CACHE_INTERVAL);
    EXPECT_EQ(paging_data.block_indices[0], paging_data.block_indices[1]);
    EXPECT_NE(paging_data.block_indices[1], paging_data.block_indices[2]);
    EXPECT_EQ(orchestrator->get_block_size(CacheType::LINEAR_ATTENTION_CACHE), TEST_CUSTOM_CACHE_INTERVAL);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id).size(), 2);

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, hybrid_prefix_caching_reuses_active_complete_linear_attention_checkpoint) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 16;
    scheduler_config.num_kv_blocks = 16;
    scheduler_config.num_linear_attention_blocks = 16;
    scheduler_config.cache_interval_multiplier = 1;
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 4;

    std::vector<uint64_t> producer_tokens = {0, 1, 2, 3};
    SequenceGroup::Ptr producer_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {producer_tokens.size()}, producer_tokens.data()),
        utils::get_greedy_config());
    const auto producer_seq_id = producer_group->get_running_sequences()[0]->get_id();

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    std::vector<SequenceGroup::Ptr> producer_requests = {producer_group};
    std::ignore = scheduler.schedule(producer_requests);
    producer_group->finish_iteration();

    ASSERT_EQ(orchestrator->get_linear_attention_block_table(producer_seq_id).size(), 1);
    const auto shared_checkpoint_idx = orchestrator->get_linear_attention_block_table(producer_seq_id).at(0)->get_index();

    std::vector<uint64_t> consumer_tokens = {0, 1, 2, 3, 4};
    SequenceGroup::Ptr consumer_group = std::make_shared<SequenceGroup>(
        1,
        ov::Tensor(ov::element::i64, {consumer_tokens.size()}, consumer_tokens.data()),
        utils::get_greedy_config());
    const auto consumer_seq_id = consumer_group->get_running_sequences()[0]->get_id();

    scheduler.restore_cached_blocks(consumer_group);
    ASSERT_EQ(orchestrator->get_linear_attention_block_table(consumer_seq_id).size(), 1);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(consumer_seq_id).at(0)->get_index(), shared_checkpoint_idx);
    EXPECT_EQ(consumer_group->get_num_processed_tokens(), producer_tokens.size());

    std::vector<SequenceGroup::Ptr> consumer_requests = {consumer_group};
    auto out = scheduler.schedule(consumer_requests);

    EXPECT_EQ(out.m_total_num_scheduled_tokens, 1);
    ASSERT_TRUE(out.has_linear_attention_paging_data(consumer_seq_id));
    const auto& paging_data = out.get_linear_attention_paging_data(consumer_seq_id);
    ASSERT_EQ(paging_data.block_indices.size(), 2);
    EXPECT_EQ(paging_data.past_length, producer_tokens.size());
    EXPECT_EQ(paging_data.cache_interval, TEST_BLOCK_SIZE);
    EXPECT_EQ(paging_data.block_indices[0], shared_checkpoint_idx);
    EXPECT_NE(paging_data.block_indices[1], shared_checkpoint_idx);
    ASSERT_EQ(orchestrator->get_linear_attention_block_table(consumer_seq_id).size(), 2);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(consumer_seq_id).at(0)->get_index(), shared_checkpoint_idx);
    EXPECT_NE(orchestrator->get_linear_attention_block_table(consumer_seq_id).at(1)->get_index(), shared_checkpoint_idx);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(producer_seq_id).at(0)->get_index(), shared_checkpoint_idx);

    scheduler.free_sequence(producer_seq_id);
    scheduler.free_sequence(consumer_seq_id);
}

TEST(TestScheduler, hybrid_prefix_caching_reuses_active_incomplete_linear_attention_checkpoint_with_cow) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 16;
    scheduler_config.num_kv_blocks = 16;
    scheduler_config.num_linear_attention_blocks = 16;
    scheduler_config.cache_interval_multiplier = 1;
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 4;

    std::vector<uint64_t> tokens = {0, 1, 2, 3, 4, 5};
    SequenceGroup::Ptr producer_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    const auto producer_seq_id = producer_group->get_running_sequences()[0]->get_id();

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    std::vector<SequenceGroup::Ptr> producer_requests = {producer_group};
    std::ignore = scheduler.schedule(producer_requests);
    producer_group->finish_iteration();

    ASSERT_EQ(orchestrator->get_linear_attention_block_table(producer_seq_id).size(), 2);
    const auto complete_checkpoint_idx = orchestrator->get_linear_attention_block_table(producer_seq_id).at(0)->get_index();
    const auto incomplete_checkpoint_idx = orchestrator->get_linear_attention_block_table(producer_seq_id).at(1)->get_index();

    SequenceGroup::Ptr consumer_group = std::make_shared<SequenceGroup>(
        1,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    const auto consumer_seq_id = consumer_group->get_running_sequences()[0]->get_id();

    scheduler.restore_cached_blocks(consumer_group);
    ASSERT_EQ(orchestrator->get_linear_attention_block_table(consumer_seq_id).size(), 1);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(consumer_seq_id).at(0)->get_index(), incomplete_checkpoint_idx);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table_logical_start(consumer_seq_id), 1);
    EXPECT_EQ(consumer_group->get_num_processed_tokens(), tokens.size() - 1);

    std::vector<SequenceGroup::Ptr> consumer_requests = {consumer_group};
    auto out = scheduler.schedule(consumer_requests);

    EXPECT_EQ(out.m_total_num_scheduled_tokens, 1);
    ASSERT_TRUE(out.has_linear_attention_paging_data(consumer_seq_id));
    const auto& paging_data = out.get_linear_attention_paging_data(consumer_seq_id);
    ASSERT_EQ(paging_data.block_indices.size(), 2);
    EXPECT_EQ(paging_data.past_length, tokens.size() - 1);
    EXPECT_EQ(paging_data.cache_interval, TEST_BLOCK_SIZE);
    EXPECT_NE(paging_data.block_indices[0], incomplete_checkpoint_idx);
    EXPECT_EQ(paging_data.block_indices[0], paging_data.block_indices[1]);
    ASSERT_EQ(orchestrator->get_linear_attention_block_table(consumer_seq_id).size(), 1);
    EXPECT_NE(orchestrator->get_linear_attention_block_table(consumer_seq_id).at(0)->get_index(), incomplete_checkpoint_idx);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table_logical_start(consumer_seq_id), 1);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(producer_seq_id).at(0)->get_index(), complete_checkpoint_idx);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(producer_seq_id).at(1)->get_index(), incomplete_checkpoint_idx);

    scheduler.free_sequence(producer_seq_id);
    scheduler.free_sequence(consumer_seq_id);
}

TEST(TestScheduler, hybrid_prefix_caching_restore_uses_minimum_common_prefix_across_cache_types) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 8;
    scheduler_config.num_kv_blocks = 4;
    scheduler_config.num_linear_attention_blocks = 2;
    scheduler_config.cache_interval_multiplier = 1;
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 4;

    std::vector<uint64_t> first_tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    SequenceGroup::Ptr first_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {first_tokens.size()}, first_tokens.data()),
        utils::get_greedy_config());
    const auto first_seq_id = first_group->get_running_sequences()[0]->get_id();

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    std::vector<SequenceGroup::Ptr> first_requests = {first_group};
    std::ignore = scheduler.schedule(first_requests);
    first_group->finish_iteration();
    scheduler.free_sequence(first_seq_id);

    std::vector<uint64_t> pressure_tokens = {10, 11, 12, 13, 14, 15, 16, 17};
    SequenceGroup::Ptr pressure_group = std::make_shared<SequenceGroup>(
        1,
        ov::Tensor(ov::element::i64, {pressure_tokens.size()}, pressure_tokens.data()),
        utils::get_greedy_config());
    const auto pressure_seq_id = pressure_group->get_running_sequences()[0]->get_id();
    std::vector<SequenceGroup::Ptr> pressure_requests = {pressure_group};
    std::ignore = scheduler.schedule(pressure_requests);
    pressure_group->finish_iteration();

    SequenceGroup::Ptr restored_group = std::make_shared<SequenceGroup>(
        2,
        ov::Tensor(ov::element::i64, {first_tokens.size()}, first_tokens.data()),
        utils::get_greedy_config());
    const auto restored_seq_id = restored_group->get_running_sequences()[0]->get_id();

    scheduler.restore_cached_blocks(restored_group);

    EXPECT_EQ(restored_group->get_num_processed_tokens(), 0);
    EXPECT_FALSE(scheduler.has_block_table(restored_seq_id));

    scheduler.free_sequence(pressure_seq_id);
}

TEST(TestScheduler, hybrid_prefix_caching_prefill_exactly_interval_uses_single_write_block) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 64;
    scheduler_config.num_kv_blocks = 64;
    scheduler_config.num_linear_attention_blocks = 16;
    scheduler_config.cache_interval_multiplier = TEST_CUSTOM_CACHE_INTERVAL_MULTIPLIER;
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 4;

    std::vector<uint64_t> tokens(64);
    std::iota(tokens.begin(), tokens.end(), 0);
    SequenceGroup::Ptr seq_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    const auto seq_id = seq_group->get_running_sequences()[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {seq_group};

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    auto out = scheduler.schedule(requests);

    ASSERT_TRUE(out.has_linear_attention_paging_data(seq_id));
    const auto& paging_data = out.get_linear_attention_paging_data(seq_id);
    ASSERT_EQ(paging_data.block_indices.size(), 2);
    EXPECT_EQ(paging_data.past_length, 0);
    EXPECT_EQ(paging_data.cache_interval, TEST_CUSTOM_CACHE_INTERVAL);
    EXPECT_EQ(paging_data.block_indices[0], paging_data.block_indices[1]);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id).size(), 1);

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, hybrid_prefix_caching_prefill_interval_plus_one_uses_next_write_block) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 128;
    scheduler_config.num_kv_blocks = 64;
    scheduler_config.num_linear_attention_blocks = 16;
    scheduler_config.cache_interval_multiplier = TEST_CUSTOM_CACHE_INTERVAL_MULTIPLIER;
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 4;

    std::vector<uint64_t> tokens(65);
    std::iota(tokens.begin(), tokens.end(), 0);
    SequenceGroup::Ptr seq_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    const auto seq_id = seq_group->get_running_sequences()[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {seq_group};

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    auto out = scheduler.schedule(requests);

    ASSERT_TRUE(out.has_linear_attention_paging_data(seq_id));
    const auto& paging_data = out.get_linear_attention_paging_data(seq_id);
    ASSERT_EQ(paging_data.block_indices.size(), 3);
    EXPECT_EQ(paging_data.past_length, 0);
    EXPECT_EQ(paging_data.cache_interval, TEST_CUSTOM_CACHE_INTERVAL);
    EXPECT_EQ(paging_data.block_indices[0], paging_data.block_indices[1]);
    EXPECT_NE(paging_data.block_indices[1], paging_data.block_indices[2]);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id).size(), 2);

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, hybrid_prefix_caching_chunked_prefill_crossing_interval_adds_write_block) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 48;
    scheduler_config.num_kv_blocks = 64;
    scheduler_config.num_linear_attention_blocks = 16;
    scheduler_config.cache_interval_multiplier = TEST_CUSTOM_CACHE_INTERVAL_MULTIPLIER;
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.dynamic_split_fuse = true;
    scheduler_config.max_num_seqs = 4;

    std::vector<uint64_t> tokens(96);
    std::iota(tokens.begin(), tokens.end(), 0);
    SequenceGroup::Ptr seq_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    const auto seq_id = seq_group->get_running_sequences()[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {seq_group};

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    auto first_out = scheduler.schedule(requests);

    ASSERT_TRUE(first_out.has_linear_attention_paging_data(seq_id));
    ASSERT_EQ(first_out.get_linear_attention_paging_data(seq_id).block_indices.size(), 2);
    EXPECT_EQ(first_out.get_linear_attention_paging_data(seq_id).past_length, 0);
    EXPECT_EQ(first_out.get_linear_attention_paging_data(seq_id).block_indices[0],
              first_out.get_linear_attention_paging_data(seq_id).block_indices[1]);

    seq_group->finish_iteration();
    auto second_out = scheduler.schedule(requests);

    ASSERT_TRUE(second_out.has_linear_attention_paging_data(seq_id));
    const auto& second_paging_data = second_out.get_linear_attention_paging_data(seq_id);
    ASSERT_EQ(second_paging_data.block_indices.size(), 3);
    EXPECT_EQ(second_paging_data.past_length, 48);
    EXPECT_EQ(second_paging_data.cache_interval, TEST_CUSTOM_CACHE_INTERVAL);
    EXPECT_EQ(second_paging_data.block_indices[0], second_paging_data.block_indices[1]);
    EXPECT_NE(second_paging_data.block_indices[1], second_paging_data.block_indices[2]);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id).size(), 2);

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, direct_scheduler_rejects_prefix_caching_with_tokens_to_validate) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 8;
    scheduler_config.num_kv_blocks = 64;
    scheduler_config.num_linear_attention_blocks = 16;
    scheduler_config.cache_interval_multiplier = TEST_CUSTOM_CACHE_INTERVAL_MULTIPLIER;
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.dynamic_split_fuse = true;
    scheduler_config.max_num_seqs = 4;

    std::vector<uint64_t> tokens = {0, 1, 2, 3};
    SequenceGroup::Ptr seq_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    std::vector<SequenceGroup::Ptr> requests = {seq_group};

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);

    std::ignore = scheduler.schedule(requests);
    seq_group->finish_iteration();

    auto running_sequence = seq_group->get_running_sequences()[0];
    for (size_t token = tokens.size(); token < 62; ++token) {
        running_sequence->append_token(token, 0.9f);
    }
    seq_group->update_processed_tokens_num(62);
    seq_group->set_num_validated_tokens(2);

    EXPECT_THROW(std::ignore = scheduler.schedule(requests), ov::Exception);

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, hybrid_prefix_caching_cache_interval_multiplier_one_allocates_block_per_kv_block) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 8;
    scheduler_config.num_kv_blocks = 64;
    scheduler_config.num_linear_attention_blocks = 16;
    scheduler_config.cache_interval_multiplier = 1;
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 4;

    std::vector<uint64_t> tokens = {0, 1, 2};
    SequenceGroup::Ptr seq_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    const auto seq_id = seq_group->get_running_sequences()[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {seq_group};

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    auto out = scheduler.schedule(requests);

    ASSERT_TRUE(out.has_linear_attention_paging_data(seq_id));
    const auto& paging_data = out.get_linear_attention_paging_data(seq_id);
    ASSERT_EQ(paging_data.block_indices.size(), 2);
    EXPECT_EQ(paging_data.cache_interval, TEST_BLOCK_SIZE);
    EXPECT_EQ(paging_data.block_indices[0], paging_data.block_indices[1]);
    EXPECT_EQ(std::set<int32_t>(paging_data.block_indices.begin(), paging_data.block_indices.end()).size(), 1);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id).size(), 1);

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, hybrid_prefix_caching_dynamic_allocation_honors_custom_cache_interval_multiplier) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 128;
    scheduler_config.num_kv_blocks = 0;
    scheduler_config.cache_size = 0;
    scheduler_config.num_linear_attention_blocks = 0;
    scheduler_config.cache_interval_multiplier = TEST_CUSTOM_CACHE_INTERVAL_MULTIPLIER;
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.dynamic_split_fuse = true;
    scheduler_config.max_num_seqs = 4;

    std::vector<uint64_t> tokens(96);
    std::iota(tokens.begin(), tokens.end(), 0);
    SequenceGroup::Ptr seq_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    const auto seq_id = seq_group->get_running_sequences()[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {seq_group};

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    auto out = scheduler.schedule(requests);

    ASSERT_TRUE(out.has_linear_attention_paging_data(seq_id));
    const auto& paging_data = out.get_linear_attention_paging_data(seq_id);
    ASSERT_EQ(paging_data.block_indices.size(), 3);
    EXPECT_EQ(paging_data.cache_interval, TEST_CUSTOM_CACHE_INTERVAL);
    EXPECT_EQ(paging_data.block_indices[0], paging_data.block_indices[1]);
    EXPECT_NE(paging_data.block_indices[1], paging_data.block_indices[2]);
    EXPECT_EQ(orchestrator->get_block_size(CacheType::LINEAR_ATTENTION_CACHE), TEST_CUSTOM_CACHE_INTERVAL);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id).size(), 2);

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, scheduler_config_zero_cache_interval_multiplier_requires_disabled_prefix_caching) {
    SchedulerConfig scheduler_config;
    scheduler_config.enable_prefix_caching = true;

    EXPECT_FALSE(scheduler_config.cache_interval_multiplier.has_value());
    EXPECT_NO_THROW(scheduler_config.validate());
    EXPECT_EQ(get_test_cache_interval(scheduler_config), TEST_DEFAULT_CACHE_INTERVAL);

    scheduler_config.cache_interval_multiplier = 0;

    EXPECT_ANY_THROW(scheduler_config.validate());

    scheduler_config.enable_prefix_caching = false;
    EXPECT_NO_THROW(scheduler_config.validate());
    ASSERT_TRUE(scheduler_config.cache_interval_multiplier.has_value());
    EXPECT_EQ(scheduler_config.cache_interval_multiplier.value(), 0);
}

TEST(TestScheduler, scheduler_config_custom_cache_interval_multiplier_is_ignored_for_kv_only_model) {
    ov::Core core;
    ov::InferRequest request = core.compile_model(get_dummy_model(core, TEST_NUM_DECODER_LAYERS)).create_infer_request();
    auto get_available_memory = [](const std::string&, size_t) {
        return std::numeric_limits<size_t>::max();
    };

    SchedulerConfig default_config;
    default_config.num_kv_blocks = 64;
    EXPECT_FALSE(default_config.cache_interval_multiplier.has_value());
    EXPECT_NO_THROW(CacheOrchestrator::create(request, default_config, get_available_memory));

    SchedulerConfig explicit_default_config;
    explicit_default_config.num_kv_blocks = 64;
    explicit_default_config.cache_interval_multiplier = DEFAULT_LINEAR_ATTENTION_CACHE_INTERVAL_MULTIPLIER;
    EXPECT_NO_THROW(CacheOrchestrator::create(request, explicit_default_config, get_available_memory));

    SchedulerConfig custom_interval_config;
    custom_interval_config.num_kv_blocks = 64;
    custom_interval_config.cache_interval_multiplier = TEST_CUSTOM_CACHE_INTERVAL_MULTIPLIER;
    EXPECT_NO_THROW(CacheOrchestrator::create(request, custom_interval_config, get_available_memory));
}

TEST(TestScheduler, hybrid_create_explicit_kv_blocks_derives_single_fixed_linear_attention_block_for_client_scenario) {
    HybridCreateContext context = create_hybrid_create_context();
    auto get_available_memory = [](const std::string&, size_t) {
        return std::numeric_limits<size_t>::max();
    };

    SchedulerConfig scheduler_config;
    scheduler_config.num_kv_blocks = 64;
    scheduler_config.max_num_seqs = 7;
    scheduler_config.max_num_batched_tokens = std::numeric_limits<size_t>::max();
    scheduler_config.enable_prefix_caching = false;
    scheduler_config.cache_interval_multiplier = 0;

    auto orchestrator = CacheOrchestrator::create(context.request, scheduler_config, get_available_memory);

    ASSERT_EQ(scheduler_config.num_kv_blocks, 64);
    EXPECT_EQ(scheduler_config.num_linear_attention_blocks, 1);
    EXPECT_EQ(orchestrator->get_block_manager(CacheType::KV_CACHE).get_total_block_count(), 64);
    EXPECT_EQ(orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE).get_total_block_count(), 1);
}

TEST(TestScheduler, hybrid_create_explicit_kv_blocks_derives_fixed_linear_attention_capacity_from_max_num_seqs_for_bounded_batching) {
    HybridCreateContext context = create_hybrid_create_context();
    auto get_available_memory = [](const std::string&, size_t) {
        return std::numeric_limits<size_t>::max();
    };

    SchedulerConfig scheduler_config;
    scheduler_config.num_kv_blocks = 64;
    scheduler_config.max_num_seqs = 7;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.enable_prefix_caching = false;
    scheduler_config.cache_interval_multiplier = 0;

    auto orchestrator = CacheOrchestrator::create(context.request, scheduler_config, get_available_memory);

    ASSERT_EQ(scheduler_config.num_kv_blocks, 64);
    EXPECT_EQ(scheduler_config.num_linear_attention_blocks, scheduler_config.max_num_seqs);
    EXPECT_EQ(orchestrator->get_block_manager(CacheType::KV_CACHE).get_total_block_count(), 64);
    EXPECT_EQ(orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE).get_total_block_count(),
              scheduler_config.max_num_seqs);
}

TEST(TestScheduler, hybrid_create_explicit_kv_blocks_derives_paged_linear_attention_capacity_from_token_target) {
    HybridCreateContext context = create_hybrid_create_context();
    auto get_available_memory = [](const std::string&, size_t) {
        return std::numeric_limits<size_t>::max();
    };

    SchedulerConfig scheduler_config;
    scheduler_config.num_kv_blocks = 10;
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.cache_interval_multiplier = 1;

    auto orchestrator = CacheOrchestrator::create(context.request, scheduler_config, get_available_memory);

    const size_t expected_token_capacity = scheduler_config.num_kv_blocks * context.kv_block_size;
    const size_t cache_interval = scheduler_config.get_cache_interval(context.kv_block_size);
    const size_t expected_la_blocks = (expected_token_capacity + cache_interval - 1) / cache_interval;

    ASSERT_EQ(scheduler_config.num_kv_blocks, 10);
    EXPECT_EQ(scheduler_config.num_linear_attention_blocks, expected_la_blocks);
    EXPECT_EQ(orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE).get_total_block_count(),
              expected_la_blocks);
}

TEST(TestScheduler, hybrid_create_cache_size_budget_reserves_fixed_linear_attention_bytes_before_kv_blocks) {
    HybridCreateContext context = create_hybrid_create_context();
    auto get_available_memory = [](const std::string&, size_t) {
        return std::numeric_limits<size_t>::max();
    };

    SchedulerConfig scheduler_config;
    scheduler_config.cache_size = 1;
    scheduler_config.max_num_seqs = 5;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.enable_prefix_caching = false;
    scheduler_config.cache_interval_multiplier = 0;

    const size_t total_budget_in_bytes = scheduler_config.cache_size * 1024ULL * 1024ULL * 1024ULL;
    const size_t reserved_la_bytes = scheduler_config.max_num_seqs * context.la_block_size_in_bytes;
    ASSERT_LT(reserved_la_bytes, total_budget_in_bytes);
    const size_t expected_kv_blocks = (total_budget_in_bytes - reserved_la_bytes) / context.kv_block_size_in_bytes;

    auto orchestrator = CacheOrchestrator::create(context.request, scheduler_config, get_available_memory);

    EXPECT_EQ(scheduler_config.num_linear_attention_blocks, scheduler_config.max_num_seqs);
    EXPECT_EQ(scheduler_config.num_kv_blocks, expected_kv_blocks);
    EXPECT_EQ(orchestrator->get_total_cache_size_in_bytes(),
              expected_kv_blocks * context.kv_block_size_in_bytes +
                  scheduler_config.num_linear_attention_blocks * context.la_block_size_in_bytes);
    EXPECT_LE(orchestrator->get_total_cache_size_in_bytes(), total_budget_in_bytes);
}

TEST(TestScheduler, hybrid_create_cache_size_budget_derives_paged_linear_attention_capacity_from_shared_token_target) {
    HybridCreateContext context = create_hybrid_create_context();
    auto get_available_memory = [](const std::string&, size_t) {
        return std::numeric_limits<size_t>::max();
    };

    SchedulerConfig scheduler_config;
    scheduler_config.cache_size = 1;
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.cache_interval_multiplier = 1;

    const size_t total_budget_in_bytes = scheduler_config.cache_size * 1024ULL * 1024ULL * 1024ULL;
    const auto bytes_for_token_target = [&](size_t token_target) {
        const size_t kv_blocks = (token_target + context.kv_block_size - 1) / context.kv_block_size;
        const size_t cache_interval = scheduler_config.get_cache_interval(context.kv_block_size);
        const size_t la_blocks = (token_target + cache_interval - 1) / cache_interval;
        return kv_blocks * context.kv_block_size_in_bytes + la_blocks * context.la_block_size_in_bytes;
    };

    size_t low = 0;
    size_t high = (total_budget_in_bytes / context.kv_block_size_in_bytes) * context.kv_block_size;
    while (low < high) {
        const size_t mid = low + (high - low + 1) / 2;
        if (bytes_for_token_target(mid) <= total_budget_in_bytes) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }

    const size_t expected_token_target = low;
    const size_t expected_kv_blocks = (expected_token_target + context.kv_block_size - 1) / context.kv_block_size;
    const size_t cache_interval = scheduler_config.get_cache_interval(context.kv_block_size);
    const size_t expected_la_blocks = (expected_token_target + cache_interval - 1) / cache_interval;

    auto orchestrator = CacheOrchestrator::create(context.request, scheduler_config, get_available_memory);

    EXPECT_EQ(scheduler_config.num_kv_blocks, expected_kv_blocks);
    EXPECT_EQ(scheduler_config.num_linear_attention_blocks, expected_la_blocks);
    EXPECT_EQ(orchestrator->get_total_cache_size_in_bytes(), bytes_for_token_target(expected_token_target));
    EXPECT_LE(orchestrator->get_total_cache_size_in_bytes(), total_budget_in_bytes);
    EXPECT_GT(bytes_for_token_target(expected_token_target + 1), total_budget_in_bytes);
}

TEST(TestScheduler, hybrid_create_zero_budget_keeps_all_cache_pools_dynamic) {
    HybridCreateContext context = create_hybrid_create_context();
    auto get_available_memory = [](const std::string&, size_t) {
        return std::numeric_limits<size_t>::max();
    };

    SchedulerConfig scheduler_config;
    scheduler_config.enable_prefix_caching = false;
    scheduler_config.cache_interval_multiplier = 0;

    auto orchestrator = CacheOrchestrator::create(context.request, scheduler_config, get_available_memory);

    EXPECT_EQ(scheduler_config.num_kv_blocks, 0);
    EXPECT_EQ(scheduler_config.num_linear_attention_blocks, 0);
    EXPECT_EQ(orchestrator->get_block_manager(CacheType::KV_CACHE).get_total_block_count(), 0);
    EXPECT_EQ(orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE).get_total_block_count(), 0);
}

TEST(TestScheduler, scheduler_config_explicit_linear_attention_blocks_require_linear_attention_model) {
    ov::Core core;
    ov::InferRequest request = core.compile_model(get_dummy_model(core, TEST_NUM_DECODER_LAYERS)).create_infer_request();
    auto get_available_memory = [](const std::string&, size_t) {
        return std::numeric_limits<size_t>::max();
    };

    SchedulerConfig scheduler_config;
    scheduler_config.num_kv_blocks = 64;
    scheduler_config.num_linear_attention_blocks = 4;
    scheduler_config.cache_interval_multiplier = 0;

    EXPECT_ANY_THROW(CacheOrchestrator::create(request, scheduler_config, get_available_memory));
}

TEST(TestScheduler, hybrid_prefix_caching_generation_finishing_interval_reuses_same_write_block) {
    // Target contract: finishing an interval writes the checkpoint in-place.
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 64;
    scheduler_config.num_linear_attention_blocks = 16;
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.dynamic_split_fuse = true;
    scheduler_config.max_num_seqs = 4;

    std::vector<uint64_t> tokens = {0, 1, 2, 3};
    SequenceGroup::Ptr seq_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    const auto seq_id = seq_group->get_running_sequences()[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {seq_group};

    Scheduler scheduler = Scheduler(init_hybrid_cache_orchestrator(scheduler_config), scheduler_config);

    // Prime cache/tables with prompt scheduling.
    std::ignore = scheduler.schedule(requests);
    for (auto& req : requests) {
        req->finish_iteration();
    }

    auto running_sequence = seq_group->get_running_sequences()[0];
    for (size_t token = tokens.size(); token < TEST_DEFAULT_CACHE_INTERVAL - 1; ++token) {
        running_sequence->append_token(token, 0.9f);
    }

    // processed=31, scheduled=1 stores the 32-token checkpoint in the current block.
    seq_group->update_processed_tokens_num(TEST_DEFAULT_CACHE_INTERVAL - 1);
    auto out = scheduler.schedule(requests);

    ASSERT_TRUE(out.has_linear_attention_paging_data(seq_id));
    ASSERT_EQ(out.get_linear_attention_paging_data(seq_id).block_indices.size(), 2);
    const auto read_idx = out.get_linear_attention_paging_data(seq_id).block_indices[0];
    const auto write_idx = out.get_linear_attention_paging_data(seq_id).block_indices[1];
    EXPECT_EQ(read_idx, write_idx);
    EXPECT_EQ(out.get_linear_attention_paging_data(seq_id).cache_interval, TEST_DEFAULT_CACHE_INTERVAL);

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, hybrid_prefix_caching_generation_after_completed_interval_switches_write_block) {
    // Target contract: after a completed interval, the next generation step reads the checkpoint
    // from the previous interval and writes to the next block.
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 64;
    scheduler_config.num_linear_attention_blocks = 16;
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.dynamic_split_fuse = true;
    scheduler_config.max_num_seqs = 4;

    std::vector<uint64_t> tokens = {0, 1, 2, 3};
    SequenceGroup::Ptr seq_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    const auto seq_id = seq_group->get_running_sequences()[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {seq_group};

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);

    std::ignore = scheduler.schedule(requests);
    for (auto& req : requests) {
        req->finish_iteration();
    }

    auto running_sequence = seq_group->get_running_sequences()[0];
    for (size_t token = tokens.size(); token < TEST_DEFAULT_CACHE_INTERVAL; ++token) {
        running_sequence->append_token(token, 0.9f);
    }

    seq_group->update_processed_tokens_num(TEST_DEFAULT_CACHE_INTERVAL);
    auto out = scheduler.schedule(requests);

    ASSERT_TRUE(out.has_linear_attention_paging_data(seq_id));
    ASSERT_EQ(out.get_linear_attention_paging_data(seq_id).block_indices.size(), 2);
    const auto read_idx = out.get_linear_attention_paging_data(seq_id).block_indices[0];
    const auto write_idx = out.get_linear_attention_paging_data(seq_id).block_indices[1];
    EXPECT_NE(read_idx, write_idx);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id).size(), 2);

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, hybrid_prefix_caching_generation_inside_interval_reuses_same_write_block) {
    // Target contract: when generation does not cross interval boundary, [read, write] must use same block.
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 64;
    scheduler_config.num_linear_attention_blocks = 16;
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.dynamic_split_fuse = true;
    scheduler_config.max_num_seqs = 4;

    std::vector<uint64_t> tokens = {0, 1, 2, 3};
    SequenceGroup::Ptr seq_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    const auto seq_id = seq_group->get_running_sequences()[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {seq_group};

    Scheduler scheduler = Scheduler(init_hybrid_cache_orchestrator(scheduler_config), scheduler_config);

    // Prime cache/tables with prompt scheduling.
    std::ignore = scheduler.schedule(requests);
    for (auto& req : requests) {
        req->finish_iteration();
    }

    auto running_sequence = seq_group->get_running_sequences()[0];
    for (size_t token = tokens.size(); token < TEST_DEFAULT_CACHE_INTERVAL - 1; ++token) {
        running_sequence->append_token(token, 0.9f);
    }

    // processed=30, scheduled=1 does not cross cache_interval=32 boundary.
    seq_group->update_processed_tokens_num(TEST_DEFAULT_CACHE_INTERVAL - 2);
    auto out = scheduler.schedule(requests);

    ASSERT_TRUE(out.has_linear_attention_paging_data(seq_id));
    ASSERT_EQ(out.get_linear_attention_paging_data(seq_id).block_indices.size(), 2);
    const auto read_idx = out.get_linear_attention_paging_data(seq_id).block_indices[0];
    const auto write_idx = out.get_linear_attention_paging_data(seq_id).block_indices[1];
    EXPECT_EQ(read_idx, write_idx);
    EXPECT_EQ(out.get_linear_attention_paging_data(seq_id).cache_interval, TEST_DEFAULT_CACHE_INTERVAL);

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

SchedulerConfig get_scheduler_config(size_t max_num_batched_tokens,
                                     size_t num_kv_blocks,
                                     bool dynamic_split_fuse,
                                     size_t max_num_seqs,
                                     std::optional<ov::genai::CacheEvictionConfig> cache_eviction_config = std::nullopt) {
    auto retval = SchedulerConfig();
    retval.max_num_batched_tokens = max_num_batched_tokens;
    retval.num_kv_blocks = num_kv_blocks;
    retval.dynamic_split_fuse = dynamic_split_fuse;
    retval.max_num_seqs = max_num_seqs;
    retval.use_cache_eviction = false;
    if (cache_eviction_config.has_value()) {
        retval.cache_eviction_config = cache_eviction_config.value();
    }
    return retval;
}

const ov::genai::CacheEvictionConfig LONG_EVICTION_CONFIG = ov::genai::CacheEvictionConfig(32, 32, 128, ov::genai::AggregationMode::NORM_SUM);


using AppendSlotsSchedulerTest = ::testing::TestWithParam<SchedulerConfig>;
const std::vector<SchedulerConfig> APPEND_SLOTS_TEST_CASES = {
        get_scheduler_config(32, 5, false, 5),
        get_scheduler_config(32, 5, true, 5),
};

TEST_P(AppendSlotsSchedulerTest, test_append_slots_considers_all_sequences) {
    auto scheduler_config = GetParam();
    std::vector<uint64_t> tokens = {0,1,2,3,4,5,6,7};
    SequenceGroup::Ptr sequence_group1 = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                            utils::get_greedy_config());
    auto idx0 = (*sequence_group1)[0]->get_id();
    SequenceGroup::Ptr sequence_group2 = std::make_shared<SequenceGroup>(1, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                            utils::get_greedy_config());
    auto idx1 = (*sequence_group2)[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {sequence_group1, sequence_group2};

    Scheduler scheduler = Scheduler(init_cache_orchestrator(scheduler_config), scheduler_config);
    auto out1 = scheduler.schedule(requests);

    std::vector<uint64_t> ref_ids = {0, 1};
    EXPECT_EQ(out1.m_scheduled_sequence_groups_ids, ref_ids);
    EXPECT_EQ(out1.get_kv_block_tables(idx0)[0].size(), 2);
    EXPECT_EQ(out1.get_kv_block_tables(idx1)[0].size(), 2);
    EXPECT_FALSE(out1.get_kv_block_tables(idx0)[0][0]->is_free());
    EXPECT_EQ(out1.get_kv_block_tables(idx0)[0][0]->get_index(), 0);
    EXPECT_FALSE(out1.get_kv_block_tables(idx0)[0][1]->is_free());
    EXPECT_EQ(out1.get_kv_block_tables(idx0)[0][1]->get_index(), 1);
    EXPECT_FALSE(out1.get_kv_block_tables(idx1)[0][0]->is_free());
    EXPECT_EQ(out1.get_kv_block_tables(idx1)[0][0]->get_index(), 2);
    EXPECT_FALSE(out1.get_kv_block_tables(idx1)[0][1]->is_free());
    EXPECT_EQ(out1.get_kv_block_tables(idx1)[0][1]->get_index(), 3);
    EXPECT_EQ(out1.m_total_num_scheduled_tokens, tokens.size() * 2);
    EXPECT_EQ(out1.is_prompt, !scheduler_config.dynamic_split_fuse);
    for (auto seq: requests) {
        std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
        // prompt phase
        seq->finish_iteration();
    }

    // at this point we used 4/5 KV blocks. Both sequences require new KV block, but we have space for only one.
    auto out2 = scheduler.schedule(requests);

    // 1-st sequence now should use 3 kv-blocks
    EXPECT_EQ(out2.get_kv_block_tables(idx0)[0].size(), 3);
    EXPECT_FALSE(out2.get_kv_block_tables(idx0)[0][0]->is_free());
    EXPECT_EQ(out2.get_kv_block_tables(idx0)[0][0]->get_index(), 0);
    EXPECT_FALSE(out2.get_kv_block_tables(idx0)[0][1]->is_free());
    EXPECT_EQ(out2.get_kv_block_tables(idx0)[0][1]->get_index(), 1);
    EXPECT_FALSE(out2.get_kv_block_tables(idx0)[0][2]->is_free());
    EXPECT_EQ(out2.get_kv_block_tables(idx0)[0][2]->get_index(), 4);

    // 1 token was scheduled for generate phase
    EXPECT_EQ(out2.m_total_num_scheduled_tokens, 1);

    EXPECT_FALSE(out2.is_prompt);

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

INSTANTIATE_TEST_SUITE_P(VariousSchedulerConfigs, AppendSlotsSchedulerTest,
                         ::testing::ValuesIn(APPEND_SLOTS_TEST_CASES));

using PartialPreemptionSchedulerTest = ::testing::TestWithParam<SchedulerConfig>;
const std::vector<SchedulerConfig> PARTIAL_PREEMPTION_TEST_CASES = {
        get_scheduler_config(32, 6, false, 5),
        get_scheduler_config(32, 6, true, 5),

        // Cache eviction should not impact preemption for cache eviction's max_cache_size larger than the sequence lengths at preemption time
        get_scheduler_config(32, 6, false, 5, LONG_EVICTION_CONFIG),
        get_scheduler_config(32, 6, true, 5, LONG_EVICTION_CONFIG)
};

TEST_P(PartialPreemptionSchedulerTest, test_partial_preemption) {
    auto scheduler_config = GetParam();
    std::vector<uint64_t> tokens1 = {0,1,2,3,4,5,6,7,8,9,10};
    SequenceGroup::Ptr sequence_group1 = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens1.size()}, tokens1.data()),
                                                                            utils::get_greedy_config());
    std::vector<uint64_t> tokens2 = {0,1,2,3,4,5,6,7};
    auto idx0 = (*sequence_group1)[0]->get_id();
    SequenceGroup::Ptr sequence_group2 = std::make_shared<SequenceGroup>(1, ov::Tensor(ov::element::i64, {tokens2.size()}, tokens2.data()),
                                                                            utils::get_greedy_config());
    auto idx1 = (*sequence_group2)[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {sequence_group1, sequence_group2};


    // schedule 2 sequence groups that use 5 kv blocks
    Scheduler scheduler = Scheduler(init_cache_orchestrator(scheduler_config), scheduler_config);
    auto out0 = scheduler.schedule(requests);

    for (auto seq: requests) {
        std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
        // prompt phase
        seq->finish_iteration();
    }


    // schedule generate, all 6 kv blocks are used.
    auto out1 = scheduler.schedule(requests);

    for (auto seq: requests) {
        std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
        // generate phase
        running_sequences[0]->append_token(16, 0.9);
        seq->finish_iteration();
    }

    // sequence_group2 should be partially preempted
    auto out2 = scheduler.schedule(requests);

    std::vector<uint64_t> ref_ids = {0};
    EXPECT_EQ(out2.m_scheduled_sequence_groups_ids, ref_ids);
    auto block_table1 = scheduler.get_kv_block_tables(*(*sequence_group1)[0])[0];
    auto block_table2 = scheduler.get_kv_block_tables(*(*sequence_group2)[0])[0];
    EXPECT_EQ(block_table1.size(), 4);
    EXPECT_EQ(block_table1[0]->get_index(), 0);
    EXPECT_EQ(block_table1[1]->get_index(), 1);
    EXPECT_EQ(block_table1[2]->get_index(), 2);
    EXPECT_EQ(block_table1[3]->get_index(), 5);
    EXPECT_EQ(block_table2.size(), 2);
    EXPECT_EQ(block_table2[0]->get_index(), 3);
    EXPECT_EQ(block_table2[1]->get_index(), 4);

    EXPECT_EQ(out2.m_total_num_scheduled_tokens, 1);
    EXPECT_EQ(out2.get_kv_block_tables(idx0)[0][0]->get_index(), 0);
    EXPECT_EQ(out2.get_kv_block_tables(idx0)[0][1]->get_index(), 1);
    EXPECT_EQ(out2.get_kv_block_tables(idx0)[0][2]->get_index(), 2);
    EXPECT_EQ(out2.get_kv_block_tables(idx0)[0][3]->get_index(), 5);

    // finish first sequence
    requests[0]->get_running_sequences()[0]->set_status(SequenceStatus::FINISHED);
    scheduler.free_sequence(idx0);
    clear_finished_sequences(requests);
    // KV blocks 0,1,2,5 are free now

    // sequence_group2 should be scheduled
    auto out3 = scheduler.schedule(requests);

    // last token should be recomputed
    EXPECT_EQ(out3.m_total_num_scheduled_tokens, 1);
    EXPECT_EQ(out3.get_kv_block_tables(idx1)[0][0]->get_index(), 3);
    EXPECT_EQ(out3.get_kv_block_tables(idx1)[0][1]->get_index(), 4);
    EXPECT_EQ(out3.get_kv_block_tables(idx1)[0][2]->get_index(), 0);

    block_table2 = scheduler.get_kv_block_tables(*(*sequence_group2)[0])[0];
    EXPECT_EQ(block_table2.size(), 3);
    EXPECT_EQ(block_table2[0]->get_index(), 3);
    EXPECT_EQ(block_table2[1]->get_index(), 4);
    EXPECT_EQ(block_table2[2]->get_index(), 0);

    EXPECT_FALSE(scheduler.has_block_table(idx0));

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

INSTANTIATE_TEST_SUITE_P(VariousSchedulerConfigs, PartialPreemptionSchedulerTest ,
                         ::testing::ValuesIn(PARTIAL_PREEMPTION_TEST_CASES));

TEST(TestScheduler, test_partial_preemption_beam_search) {
    std::array<SchedulerConfig, 2> configs = {SchedulerConfig(), SchedulerConfig()};
    configs.at(0).num_kv_blocks = 10;
    configs.at(0).dynamic_split_fuse = false;
    configs.at(1).num_kv_blocks = 10;
    configs.at(1).dynamic_split_fuse = true;
    for (auto scheduler_config: configs) {
        std::vector<uint64_t> tokens = {0,1,2,3};
        int64_t token = 4;

        // create beam search group
        SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                utils::get_beam_search_config());
        std::vector<SequenceGroup::Ptr> requests = {sequence_group};
        EXPECT_NO_THROW(requests[0]->get_running_sequences()[0]->get_sequence_group_ptr());

        Scheduler scheduler = Scheduler(init_cache_orchestrator(scheduler_config), scheduler_config);
        auto out = scheduler.schedule(requests);
        for (auto sequence: sequence_group->get_not_finished_sequences()) {
            sequence->append_token(token, 0.7);
        }
        sequence_group->finish_iteration();

        // make 2 forked sequence
        auto sequence_to_fork = sequence_group->get_running_sequences()[0];
        for (size_t i = 0; i < 2; ++i) {
            const auto forked_sequence = sequence_group->fork_sequence(sequence_to_fork);
            scheduler.fork_sequence(sequence_to_fork->get_id(), forked_sequence->get_id());
        }
        size_t num_scheduled_tokens = 4;

        // generate 4 tokens
        for (size_t i = 0; i < num_scheduled_tokens; i++) {
            scheduler.schedule(requests);
            for (auto sequence: sequence_group->get_not_finished_sequences()) {
                token += 3;
                sequence->append_token(token, 0.5);
            }
            sequence_group->finish_iteration();
        }
        // currently sequence occupies 4 blocks (1 shared, 3 not shared)

        // make another 2 forked sequence
        for (size_t i = 0; i < 2; ++i) {
            const auto forked_sequence = sequence_group->fork_sequence(sequence_to_fork);
            scheduler.fork_sequence(sequence_to_fork->get_id(), forked_sequence->get_id());
        }

        // generate 4 tokens
        for (size_t i = 0; i < num_scheduled_tokens; i++) {
            scheduler.schedule(requests);
            for (auto sequence: sequence_group->get_not_finished_sequences()) {
                token += 3;
                sequence->append_token(token, 0.5);
            }
            sequence_group->finish_iteration();
        }
        // currently sequence occupies 9 blocks (4 blocks previously created + 5 blocks for each sequence)

        // create group, which requires 1 block
        SequenceGroup::Ptr sequence_group_greedy = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                utils::get_greedy_config());

        // set greedy group at the beginning of list to make it higher priority
        std::vector<SequenceGroup::Ptr> new_requests = {sequence_group_greedy, sequence_group};

        // process prompt of greedy group, at this point all blocks are used
        scheduler.schedule(new_requests);
        sequence_group_greedy->get_sequences()[0]->append_token(token, 0.8);
        sequence_group_greedy->finish_iteration();

        EXPECT_EQ(sequence_group->get_num_processed_tokens(), 12);
        EXPECT_EQ(sequence_group->get_context_len(), 12);

        // beam search group should be partially preempted and 5 blocks should be released
        out = scheduler.schedule(new_requests);
        sequence_group_greedy->get_sequences()[0]->append_token(token, 0.5);
        sequence_group_greedy->finish_iteration();

        EXPECT_EQ(sequence_group->get_num_processed_tokens(), 8);
        auto seqs = sequence_group->get_sequences();
        EXPECT_EQ(scheduler.get_kv_block_tables(*seqs[0])[0].size(), 2);
        EXPECT_EQ(scheduler.get_kv_block_tables(*seqs[1])[0].size(), 2);
        EXPECT_EQ(scheduler.get_kv_block_tables(*seqs[2])[0].size(), 2);
        EXPECT_EQ(scheduler.get_kv_block_tables(*seqs[3])[0].size(), 2);
        EXPECT_EQ(scheduler.get_kv_block_tables(*seqs[4])[0].size(), 2);

        // append another 20 tokens to greedy group, this should result in usage of all free blocks and
        // another partial preemption of beam search group
        for (size_t i = 0; i < 20; i++) {
            out = scheduler.schedule(new_requests);
            sequence_group_greedy->get_sequences()[0]->append_token(token, 0.5);
            sequence_group_greedy->finish_iteration();
        }

        EXPECT_EQ(sequence_group->get_num_processed_tokens(), 4);
        seqs = sequence_group->get_sequences();
        EXPECT_EQ(scheduler.get_kv_block_tables(*seqs[0])[0].size(), 1);
        EXPECT_EQ(scheduler.get_kv_block_tables(*seqs[1])[0].size(), 1);
        EXPECT_EQ(scheduler.get_kv_block_tables(*seqs[2])[0].size(), 1);
        EXPECT_EQ(scheduler.get_kv_block_tables(*seqs[3])[0].size(), 1);
        EXPECT_EQ(scheduler.get_kv_block_tables(*seqs[4])[0].size(), 1);

        for (auto& req : new_requests) {
            for (auto& seq : req->get_sequences()) {
                scheduler.free_sequence(seq->get_id());
            }
        }
    }
}

TEST(TestScheduler, test_partially_preempted_prompt) {
    std::array<SchedulerConfig, 2> configs = {SchedulerConfig(), SchedulerConfig()};
    configs.at(0).max_num_batched_tokens = 32;
    configs.at(0).num_kv_blocks = 6;
    configs.at(0).dynamic_split_fuse = false;
    configs.at(0).max_num_seqs = 5;
    configs.at(1).max_num_batched_tokens = 32;
    configs.at(1).num_kv_blocks = 6;
    configs.at(1).dynamic_split_fuse = true;
    configs.at(1).max_num_seqs = 5;
    for (auto scheduler_config: configs) {
        std::vector<uint64_t> tokens = {0,1,2,3,4,5,6,7,8,9,10,11};
        SequenceGroup::Ptr sequence_group1 = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                utils::get_greedy_config());
        auto idx0 = (*sequence_group1)[0]->get_id();
        SequenceGroup::Ptr sequence_group2 = std::make_shared<SequenceGroup>(1, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                utils::get_greedy_config());
        auto idx1 = (*sequence_group2)[0]->get_id();
        std::vector<SequenceGroup::Ptr> requests = {sequence_group1, sequence_group2};

        // schedule 2 sequence groups that use all available 2*3 kv blocks, we used all available kv-blocks.
        Scheduler scheduler = Scheduler(init_cache_orchestrator(scheduler_config), scheduler_config);
        auto out1 = scheduler.schedule(requests);

        for (auto seq: requests) {
            std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
            // prompt phase
            seq->finish_iteration();
        }

        // sequence_group2 should be fully preempted
        auto out2 = scheduler.schedule(requests);

        // check that sequence_group1 has one more allocated block
        auto block_tables_for_all_layers = scheduler.get_kv_block_tables(*(*sequence_group1)[0]);
        auto block_table1 = block_tables_for_all_layers[0];
        EXPECT_EQ(block_table1.size(), 4);
        EXPECT_EQ(block_table1[0]->get_index(), 0);
        EXPECT_EQ(block_table1[1]->get_index(), 1);
        EXPECT_EQ(block_table1[2]->get_index(), 2);
        EXPECT_EQ(block_table1[3]->get_index(), 5);
        EXPECT_EQ(out2.get_kv_block_tables(idx0)[0].size(), 4);
        EXPECT_EQ(out2.get_kv_block_tables(idx0)[0][0]->get_index(), 0);
        EXPECT_EQ(out2.get_kv_block_tables(idx0)[0][1]->get_index(), 1);
        EXPECT_EQ(out2.get_kv_block_tables(idx0)[0][2]->get_index(), 2);
        EXPECT_EQ(out2.get_kv_block_tables(idx0)[0][3]->get_index(), 5);

        std::vector<uint64_t> ref_ids = {0};
        EXPECT_EQ(out2.m_scheduled_sequence_groups_ids, ref_ids);
        EXPECT_EQ(out2.m_total_num_scheduled_tokens, 1);

        if (scheduler_config.dynamic_split_fuse) {
            // for dynamic_split_fuse sequence_group2 is preemted partially, part of prompt is left
            EXPECT_TRUE(scheduler.has_block_table(idx1));
            auto block_table2 = scheduler.get_kv_block_tables(*(*sequence_group2)[0])[0];
            EXPECT_EQ(block_table2.size(), 2); // full prompt requires 3 blocks, 2 are left in scheduler

        } else {
            // for vllm case sequence_group2 is fully preempted
            EXPECT_FALSE(scheduler.has_block_table(idx1));
        }

        for (auto seq: requests) {
            std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
            seq->finish_iteration();
        }

        // finish first sequence
        requests[0]->get_running_sequences()[0]->set_status(SequenceStatus::FINISHED);
        scheduler.free_sequence(idx0);
        clear_finished_sequences(requests);
        // KV blocks 0,1,2,5 are free now

        // sequence_group2 should be scheduled
        auto out3 = scheduler.schedule(requests);

        if (scheduler_config.dynamic_split_fuse) {
            // remaining part of prompt should be scheduled
            EXPECT_EQ(out3.m_total_num_scheduled_tokens, 4);
        }
        else {
            // prompt should be fully scheduled
            EXPECT_EQ(out3.m_total_num_scheduled_tokens, 12);
        }

        EXPECT_EQ(out3.get_kv_block_tables(idx1)[0][0]->get_index(), 3);
        EXPECT_EQ(out3.get_kv_block_tables(idx1)[0][1]->get_index(), 4);
        EXPECT_EQ(out3.get_kv_block_tables(idx1)[0][2]->get_index(), 0);

        auto block_table2 = scheduler.get_kv_block_tables(*(*sequence_group2)[0])[0];
        EXPECT_EQ(block_table2.size(), 3);
        EXPECT_EQ(block_table2[0]->get_index(), 3);
        EXPECT_EQ(block_table2[1]->get_index(), 4);
        EXPECT_EQ(block_table2[2]->get_index(), 0);

        EXPECT_FALSE(scheduler.has_block_table(idx0));

        for (auto& req : requests) {
            for (auto& seq : req->get_sequences()) {
                scheduler.free_sequence(seq->get_id());
            }
        }
    }
}

TEST(TestScheduler, prefix_caching_test) {
    std::array<SchedulerConfig, 2> configs = {SchedulerConfig(), SchedulerConfig()};
    configs.at(0).max_num_batched_tokens = 32;
    configs.at(0).num_kv_blocks = 100;
    configs.at(0).dynamic_split_fuse = false;
    configs.at(0).max_num_seqs = 5;
    configs.at(0).enable_prefix_caching = true;
    configs.at(1).max_num_batched_tokens = 32;
    configs.at(1).num_kv_blocks = 100;
    configs.at(1).dynamic_split_fuse = true;
    configs.at(1).max_num_seqs = 5;
    configs.at(1).enable_prefix_caching = true;
    for (auto scheduler_config: configs) {
        std::vector<uint64_t> prompt_tokens = {0,1,2,3,4,5,6,7};
        std::vector<uint64_t> histrory_tokens = {};
        // schedule prompt
        Scheduler scheduler = Scheduler(init_cache_orchestrator(scheduler_config), scheduler_config);

        size_t chat_iterations = 10;

        for (size_t chat_iteration = 0; chat_iteration < chat_iterations; chat_iteration++) {
            std::vector<uint64_t> tokens = histrory_tokens;
            tokens.insert(tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
            SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                    utils::get_greedy_config());
            scheduler.restore_cached_blocks(sequence_group);
            std::vector<SequenceGroup::Ptr> requests = {sequence_group};

            auto out1 = scheduler.schedule(requests);
            if (chat_iteration == 0)
                EXPECT_EQ(out1.m_total_num_scheduled_tokens, prompt_tokens.size());
            else
                EXPECT_EQ(out1.m_total_num_scheduled_tokens, prompt_tokens.size() + 1);
            for (auto seq: requests) {
                std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
                running_sequences[0]->append_token(23, 0.7);
                seq->finish_iteration();
            }

            // schedule generate
            size_t num_generate_tokens = 10;
            for (size_t i = 0; i < num_generate_tokens; i++) {
                auto out2 = scheduler.schedule(requests);
                EXPECT_EQ(out2.m_total_num_scheduled_tokens, 1);
                for (auto seq: requests) {
                    std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
                    running_sequences[0]->append_token(16, 0.9);
                    seq->finish_iteration();
                }
            }

            // finish sequence
            auto sequence = requests[0]->get_running_sequences()[0];
            sequence->set_status(SequenceStatus::FINISHED);
            auto idx0 = sequence->get_id();
            scheduler.free_sequence(idx0);
            auto generated_ids = sequence->get_generated_ids();

            histrory_tokens.insert(histrory_tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
            histrory_tokens.insert(histrory_tokens.end(), generated_ids.begin(), generated_ids.end());

            for (auto& seq : sequence_group->get_sequences()) {
                if (seq->get_id() == idx0) {
                    continue;
                }
                scheduler.free_sequence(seq->get_id());
            }
        }
    }

}

TEST(TestScheduler, prefix_caching_test_two_identical_sequences) {
    std::array<SchedulerConfig, 2> configs = {SchedulerConfig(), SchedulerConfig()};
    configs.at(0).num_kv_blocks = 100;
    configs.at(0).dynamic_split_fuse = false;
    configs.at(0).enable_prefix_caching = true;
    configs.at(1).num_kv_blocks = 100;
    configs.at(1).dynamic_split_fuse = true;
    configs.at(1).enable_prefix_caching = true;
    for (auto scheduler_config: configs) {
        std::vector<uint64_t> prompt_tokens = {0,1,2,3,4,5,6,7};
        std::vector<uint64_t> histrory_tokens = {};
        // schedule prompt
        Scheduler scheduler = Scheduler(init_cache_orchestrator(scheduler_config), scheduler_config);

        size_t chat_iterations = 10;

        for (size_t chat_iteration = 0; chat_iteration < chat_iterations; chat_iteration++) {
            std::vector<uint64_t> tokens = histrory_tokens;
            tokens.insert(tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
            SequenceGroup::Ptr sequence_group1 = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                    utils::get_greedy_config());

            SequenceGroup::Ptr sequence_group2 = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                                    utils::get_greedy_config());
            std::vector<SequenceGroup::Ptr> requests = {sequence_group1, sequence_group2};
            // restore cached blocks
            for (auto request: requests) {
                scheduler.restore_cached_blocks(request);
            }

            // schedule prompt
            auto out1 = scheduler.schedule(requests);
            if (chat_iteration == 0)
                EXPECT_EQ(out1.m_total_num_scheduled_tokens, prompt_tokens.size() * 2);
            else
                EXPECT_EQ(out1.m_total_num_scheduled_tokens, (prompt_tokens.size() + 1) * 2);
            for (auto seq: requests) {
                std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
                running_sequences[0]->append_token(23, 0.7);
                seq->finish_iteration();
            }

            // schedule generate
            size_t num_generate_tokens = 10;
            for (size_t i = 0; i < num_generate_tokens; i++) {
                auto out2 = scheduler.schedule(requests);
                EXPECT_EQ(out2.m_total_num_scheduled_tokens, 2);
                for (auto request: requests) {
                    std::vector<Sequence::Ptr> running_sequences = request->get_running_sequences();
                    running_sequences[0]->append_token(16, 0.9);
                    request->finish_iteration();
                }
            }

            for (auto request: requests) {
                // finish sequences
                auto sequence = request->get_running_sequences()[0];
                sequence->set_status(SequenceStatus::FINISHED);
                auto idx0 = sequence->get_id();
                scheduler.free_sequence(idx0);
            }
            auto generated_ids = requests[0]->get_sequences()[0]->get_generated_ids();

            histrory_tokens.insert(histrory_tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
            histrory_tokens.insert(histrory_tokens.end(), generated_ids.begin(), generated_ids.end());
        }
    }

}


TEST(TestScheduler, prefix_caching_with_max_new_tokens_equal_1) {
    std::array<SchedulerConfig, 2> configs = {SchedulerConfig(), SchedulerConfig()};
    configs.at(0).num_kv_blocks = 10;
    configs.at(0).dynamic_split_fuse = false;
    configs.at(0).enable_prefix_caching = true;
    configs.at(1).num_kv_blocks = 10;
    configs.at(1).dynamic_split_fuse = true;
    configs.at(1).enable_prefix_caching = true;
    for (auto scheduler_config: configs) {
        std::vector<uint64_t> prompt_tokens = {0,1,2,3,4,5,6,7};
        // schedule prompt
        Scheduler scheduler = Scheduler(init_cache_orchestrator(scheduler_config, 32), scheduler_config);

        size_t chat_iterations = 2;

        for (size_t chat_iteration = 0; chat_iteration < chat_iterations; chat_iteration++) {
            SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {prompt_tokens.size()}, prompt_tokens.data()),
                                                                                    utils::get_greedy_config());

            std::vector<SequenceGroup::Ptr> requests = {sequence_group};
            // restore cached blocks
            for (auto request: requests) {
                scheduler.restore_cached_blocks(request);
            }

            // schedule prompt
            auto out1 = scheduler.schedule(requests);
            if (chat_iteration == 0)
                EXPECT_EQ(out1.m_total_num_scheduled_tokens, prompt_tokens.size());
            else
                EXPECT_EQ(out1.m_total_num_scheduled_tokens, 1);
            for (auto seq: requests) {
                std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
                running_sequences[0]->append_token(23, 0.7);
                seq->finish_iteration();
            }

            // In case max_new_tokens == 1 no generate phase happens

            for (auto request: requests) {
                // finish sequences
                auto sequence = request->get_running_sequences()[0];
                sequence->set_status(SequenceStatus::FINISHED);
                auto idx0 = sequence->get_id();
                scheduler.free_sequence(idx0);
            }
        }
    }

}

TEST(TestScheduler, test_partially_preempted_prompt_not_allowed) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 6;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 5;

    std::vector<uint64_t> tokens = {0,1,2,3,4,5,6,7,8,9,10,11};
    SequenceGroup::Ptr sequence_group1 = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                            utils::get_greedy_config());
    auto idx0 = (*sequence_group1)[0]->get_id();
    SequenceGroup::Ptr sequence_group2 = std::make_shared<SequenceGroup>(1, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                            utils::get_greedy_config());
    auto idx1 = (*sequence_group2)[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {sequence_group1, sequence_group2};

    // schedule 2 sequence groups that use all available 2*3 kv blocks, we used all available kv-blocks.
    const bool can_use_partial_preemption = false;
    Scheduler scheduler = Scheduler(init_cache_orchestrator(scheduler_config), scheduler_config, can_use_partial_preemption);
    auto out1 = scheduler.schedule(requests);

    for (auto req : requests)
        req->finish_iteration();

    // sequence_group2 should be fully preempted
    auto out2 = scheduler.schedule(requests);

    // check that sequence_group1 has one more allocated block
    auto block_table1 = scheduler.get_kv_block_tables(*(*sequence_group1)[0]);
    ASSERT_EQ(block_table1[0].size(), 4);
    ASSERT_EQ(block_table1[0][0]->get_index(), 0);
    ASSERT_EQ(block_table1[0][1]->get_index(), 1);
    ASSERT_EQ(block_table1[0][2]->get_index(), 2);
    ASSERT_EQ(block_table1[0][3]->get_index(), 3);
    ASSERT_EQ(out2.get_kv_block_tables(idx0)[0].size(), 4);
    ASSERT_EQ(out2.get_kv_block_tables(idx0)[0][0]->get_index(), 0);
    ASSERT_EQ(out2.get_kv_block_tables(idx0)[0][1]->get_index(), 1);
    ASSERT_EQ(out2.get_kv_block_tables(idx0)[0][2]->get_index(), 2);
    ASSERT_EQ(out2.get_kv_block_tables(idx0)[0][3]->get_index(), 3);

    std::vector<uint64_t> ref_ids = {0};
    ASSERT_EQ(out2.m_scheduled_sequence_groups_ids, ref_ids);
    ASSERT_EQ(out2.m_total_num_scheduled_tokens, 1);

    // for vllm case sequence_group2 is fully preempted
    EXPECT_FALSE(scheduler.has_block_table(idx1));

    for (auto req : requests)
        req->finish_iteration();

    // finish first sequence
    requests[0]->get_running_sequences()[0]->set_status(SequenceStatus::FINISHED);
    scheduler.free_sequence(idx0);
    clear_finished_sequences(requests);

    // sequence_group2 should be scheduled
    auto out3 = scheduler.schedule(requests);

    // prompt should be fully scheduled
    ASSERT_EQ(out3.m_total_num_scheduled_tokens, 12);

    ASSERT_EQ(out3.get_kv_block_tables(idx1)[0][0]->get_index(), 4);
    ASSERT_EQ(out3.get_kv_block_tables(idx1)[0][1]->get_index(), 5);
    ASSERT_EQ(out3.get_kv_block_tables(idx1)[0][2]->get_index(), 0);

    auto block_table2 = scheduler.get_kv_block_tables(*(*sequence_group2)[0]);
    ASSERT_EQ(block_table2[0].size(), 3);
    ASSERT_EQ(block_table2[0][0]->get_index(), 4);
    ASSERT_EQ(block_table2[0][1]->get_index(), 5);
    ASSERT_EQ(block_table2[0][2]->get_index(), 0);

    EXPECT_FALSE(scheduler.has_block_table(idx0));

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            if (seq->get_id() == idx0) {
                continue;
            }
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, test_partially_preempted_prompt_not_allowed2) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 6;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 5;

    std::vector<uint64_t> tokens = {0,1,2,3,4,5,6,7,8,9};
    SequenceGroup::Ptr sequence_group1 = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                            utils::get_greedy_config());
    auto idx0 = (*sequence_group1)[0]->get_id();
    SequenceGroup::Ptr sequence_group2 = std::make_shared<SequenceGroup>(1, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                            utils::get_greedy_config());
    auto idx1 = (*sequence_group2)[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {sequence_group1, sequence_group2};

    // schedule 2 sequence groups that use all available 2*3 kv blocks, we used all available kv-blocks.
    const bool can_use_partial_preemption = false;
    Scheduler scheduler = Scheduler(init_cache_orchestrator(scheduler_config), scheduler_config, can_use_partial_preemption);
    scheduler.schedule(requests);
    for (auto req: requests)
        req->finish_iteration();

    scheduler.schedule(requests);
    for (auto req: requests)
        req->finish_iteration();

    scheduler.schedule(requests);
    for (auto req: requests)
        req->finish_iteration();

    // sequence_group2 should be fully preempted
    scheduler.schedule(requests);
    for (auto req: requests)
        req->finish_iteration();

    auto out2 = scheduler.schedule(requests);

    // check that sequence_group1 has one more allocated block
    auto block_table1 = scheduler.get_kv_block_tables(*(*sequence_group1)[0]);
    ASSERT_EQ(block_table1[0].size(), 4);
    ASSERT_EQ(block_table1[0][0]->get_index(), 0);
    ASSERT_EQ(block_table1[0][1]->get_index(), 1);
    ASSERT_EQ(block_table1[0][2]->get_index(), 2);
    ASSERT_EQ(block_table1[0][3]->get_index(), 3);
    ASSERT_EQ(out2.get_kv_block_tables(idx0)[0].size(), 4);
    ASSERT_EQ(out2.get_kv_block_tables(idx0)[0][0]->get_index(), 0);
    ASSERT_EQ(out2.get_kv_block_tables(idx0)[0][1]->get_index(), 1);
    ASSERT_EQ(out2.get_kv_block_tables(idx0)[0][2]->get_index(), 2);
    ASSERT_EQ(out2.get_kv_block_tables(idx0)[0][3]->get_index(), 3);

    std::vector<uint64_t> ref_ids = {0};
    ASSERT_EQ(out2.m_scheduled_sequence_groups_ids, ref_ids);
    ASSERT_EQ(out2.m_total_num_scheduled_tokens, 1);

    // for vllm case sequence_group2 is fully preempted
    EXPECT_FALSE(scheduler.has_block_table(idx1));

    for (auto req: requests)
        req->finish_iteration();

    // finish first sequence
    requests[0]->get_running_sequences()[0]->set_status(SequenceStatus::FINISHED);
    scheduler.free_sequence(idx0);
    clear_finished_sequences(requests);

    // sequence_group2 should be scheduled
    auto out3 = scheduler.schedule(requests);

    // prompt should be fully scheduled + generated tokens concatenated to prompt (10 + 2)
    ASSERT_EQ(out3.m_total_num_scheduled_tokens, 12);

    ASSERT_EQ(out3.get_kv_block_tables(idx1)[0][0]->get_index(), 4);
    ASSERT_EQ(out3.get_kv_block_tables(idx1)[0][1]->get_index(), 5);
    ASSERT_EQ(out3.get_kv_block_tables(idx1)[0][2]->get_index(), 0);

    auto block_table2 = scheduler.get_kv_block_tables(*(*sequence_group2)[0]);
    ASSERT_EQ(block_table2[0].size(), 3);
    ASSERT_EQ(block_table2[0][0]->get_index(), 4);
    ASSERT_EQ(block_table2[0][1]->get_index(), 5);
    ASSERT_EQ(block_table2[0][2]->get_index(), 0);

    EXPECT_FALSE(scheduler.has_block_table(idx0));

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            if (seq->get_id() == idx0) {
                continue;
            }
            scheduler.free_sequence(seq->get_id());
        }
    }
}


std::vector<size_t> _get_indices(const std::vector<CacheBlock::Ptr>& block_table_for_layer) {
    std::vector<size_t> retval(block_table_for_layer.size());
    for (size_t i = 0; i < block_table_for_layer.size(); i++) {
        retval[i] = block_table_for_layer[i]->get_index();
    }
    return retval;
}

Scheduler::Output _schedule_one_mock_generation_token_for_each_sequence_group(Scheduler& scheduler, std::vector<SequenceGroup::Ptr>& requests) {
    auto out = scheduler.schedule(requests);
    for (auto& req : requests) {
        std::vector<Sequence::Ptr> running_sequences = req->get_running_sequences();
        running_sequences[0]->append_token(16, 0.9);
        req->finish_iteration();
    }
    return out;
}

TEST(TestScheduler, FullyPreemptsCacheEvictedSequences) {
    // NB: only eviction at prompt phase is tested here. Eviction during generation would happen only for beam search/parallel sampling cases
    // (since greedy sampling doesn't exceed the max cache size at generation phase), but should currently execute the same code path as
    // the preemption at prompt stage anyway
    SchedulerConfig scheduler_config;

    scheduler_config.max_num_batched_tokens = 32;
    scheduler_config.num_kv_blocks = 6;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 5;
    scheduler_config.use_cache_eviction = true;
    scheduler_config.cache_eviction_config = ov::genai::CacheEvictionConfig(2, 2, 6, ov::genai::AggregationMode::NORM_SUM);

    std::vector<uint64_t> tokens1 = {0, 1};  // 1 full block
    SequenceGroup::Ptr sequence_group1 = std::make_shared<SequenceGroup>(0,
                                                                         ov::Tensor(ov::element::i64, {tokens1.size()},
                                                                                    tokens1.data()),
                                                                         utils::get_greedy_config());
    std::vector<uint64_t> tokens2 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // 5 full blocks, larger than eviction arena size (3 blocks) - will start evicting already at prompt stage
    auto idx1 = (*sequence_group1)[0]->get_id();
    SequenceGroup::Ptr sequence_group2 = std::make_shared<SequenceGroup>(1, ov::Tensor(ov::element::i64, {tokens2.size()}, tokens2.data()),
                                                                         utils::get_greedy_config());
    auto idx2 = (*sequence_group2)[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {sequence_group1, sequence_group2};

    Scheduler scheduler = Scheduler(init_cache_orchestrator(scheduler_config, 2), scheduler_config);
    // prompt phase - schedules 1 block for seq 1, 5 blocks for seq 2
    auto out = scheduler.schedule(requests);

    for (auto seq: requests) {
        std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
        seq->finish_iteration();
    }

    // evict 2 blocks from seq 2 immediately to formally satisfy eviction arena size
    std::vector<std::set<size_t>> blocks_to_evict(1, {0, 1});
    scheduler.free_blocks_from_sequence(idx2, blocks_to_evict, CacheType::KV_CACHE);
    sequence_group2->register_token_eviction(2 * 2);

    // 4 blocks are taken up at this stage

    // mock-generate 4 more tokens in the 1-st sequence group so that the remaining 2 blocks are filled up
    std::vector<SequenceGroup::Ptr> first_seq_group_only = { requests[0] };
    for (size_t i = 0; i < 4; i++) {
        // Since eviction arena size is less than the cache_size - BLOCK_SIZE, no preemption is expected to occur yet
        // - tokens are added 1 by 1 and once a new block fills, an older one is evicted automatically
        _schedule_one_mock_generation_token_for_each_sequence_group(scheduler, first_seq_group_only);
    }

    // ensure we are in expected cache state just before preemption
    auto block_table1 = _get_indices(scheduler.get_kv_block_tables(*(*sequence_group1)[0])[0]);
    auto block_table2 = _get_indices(scheduler.get_kv_block_tables(*(*sequence_group2)[0])[0]);

    const std::vector<size_t> ref_block_table1{0, 1, 2};
    EXPECT_EQ(block_table1, ref_block_table1);

    const std::vector<size_t> ref_block_table2{3, 4, 5};
    EXPECT_EQ(block_table2, ref_block_table2);

    // Next generation in 1-st sequence group should lead to preemption of 2-nd, but tokens from it were evicted already
    // Should ensure that the 2-nd sequence can only be preempted completely
    out = _schedule_one_mock_generation_token_for_each_sequence_group(scheduler, requests);

    block_table1 = _get_indices(scheduler.get_kv_block_tables(*(*sequence_group1)[0])[0]);

    const std::vector<size_t> ref_block_table1_after_preemption{0, 1, 2, 3};  // 3 was the first to be freed after preemption
    EXPECT_EQ(block_table1, ref_block_table1_after_preemption);
    EXPECT_FALSE(scheduler.has_block_table(idx2));

    // finish first sequence
    requests[0]->get_running_sequences()[0]->set_status(SequenceStatus::FINISHED);
    scheduler.free_sequence(idx1);
    clear_finished_sequences(requests);

    // sequence_group2 should be scheduled
    out = scheduler.schedule(requests);

    // last token should be recomputed
    EXPECT_FALSE(scheduler.has_block_table(idx1));
    EXPECT_TRUE(scheduler.has_block_table(idx2));
    block_table2 = _get_indices(scheduler.get_kv_block_tables(*(*sequence_group2)[0])[0]);
    const std::vector<size_t> ref_block_table2_after_recompute{4, 5, 0, 1, 2};  // should restore the old state before first eviction in terms of block count
    EXPECT_EQ(block_table2, ref_block_table2_after_recompute);

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            if (seq->get_id() == idx1) {
                continue;
            }
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, prefix_caching_embeddings_test) {
    std::array<SchedulerConfig, 2> configs = {SchedulerConfig(), SchedulerConfig()};
    configs.at(0).max_num_batched_tokens = 32;
    configs.at(0).num_kv_blocks = 100;
    configs.at(0).dynamic_split_fuse = false;
    configs.at(0).max_num_seqs = 5;
    configs.at(0).enable_prefix_caching = true;
    configs.at(1).max_num_batched_tokens = 32;
    configs.at(1).num_kv_blocks = 100;
    configs.at(1).dynamic_split_fuse = true;
    configs.at(1).max_num_seqs = 5;
    configs.at(1).enable_prefix_caching = true;
    for (auto scheduler_config: configs) {
        size_t hidden_size = 300;
        std::vector<std::vector<float>> prompt_embeddings;
        for (size_t i = 0; i < 8; i++) {
            prompt_embeddings.emplace_back(std::vector<float>());
            for (size_t j = 0; j < hidden_size; j++) {
                prompt_embeddings[i].push_back(i * hidden_size + j + (float)j * 0.05);
            }
        }
        std::vector<std::vector<float>> histrory_embeddings = {};
        // schedule prompt
        Scheduler scheduler = Scheduler(init_cache_orchestrator(scheduler_config), scheduler_config);

        size_t chat_iterations = 10;

        for (size_t chat_iteration = 0; chat_iteration < chat_iterations; chat_iteration++) {
            std::vector<std::vector<float>> embeddings = histrory_embeddings;
            embeddings.insert(embeddings.end(), prompt_embeddings.begin(), prompt_embeddings.end());
            SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, embeds_matrix_to_tensor(embeddings), utils::get_greedy_config());
            scheduler.restore_cached_blocks(sequence_group);
            std::vector<SequenceGroup::Ptr> requests = {sequence_group};

            auto out1 = scheduler.schedule(requests);
            if (chat_iteration == 0)
                EXPECT_EQ(out1.m_total_num_scheduled_tokens, prompt_embeddings.size());
            else
            {
                EXPECT_EQ(out1.m_total_num_scheduled_tokens, prompt_embeddings.size() + 1);
            }
            for (auto seq: requests) {
                std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
                running_sequences[0]->append_token(chat_iteration, 0.7);

                std::vector<float> embed(hidden_size);
                for (size_t i = 0; i < hidden_size; i++) {
                    embed[i] = chat_iteration + i * hidden_size + (float)i * 0.05; 
                }
                running_sequences[0]->append_generated_ids_embeds(embeds_matrix_to_tensor({embed}));
                seq->finish_iteration();
            }

            // schedule generate
            size_t num_generate_tokens = 10;
            for (size_t i = 0; i < num_generate_tokens; i++) {
                auto out2 = scheduler.schedule(requests);
                EXPECT_EQ(out2.m_total_num_scheduled_tokens, 1);
                for (auto seq: requests) {
                    std::vector<Sequence::Ptr> running_sequences = seq->get_running_sequences();
                    running_sequences[0]->append_token(16 + chat_iteration, 0.9);
                    std::vector<float> embed(hidden_size);
                    for (size_t i = 0; i < hidden_size; i++) {
                        embed[i] = chat_iteration + i * hidden_size + (float)i * 0.05; 
                    }
                    running_sequences[0]->append_generated_ids_embeds(embeds_matrix_to_tensor({embed}));
                    seq->finish_iteration();
                }
            }

            // finish sequence
            auto sequence = requests[0]->get_running_sequences()[0];
            sequence->set_status(SequenceStatus::FINISHED);
            auto idx0 = sequence->get_id();
            scheduler.free_sequence(idx0);
            auto generated_embeddings = sequence->get_generated_ids_embeds();

            histrory_embeddings.insert(histrory_embeddings.end(), prompt_embeddings.begin(), prompt_embeddings.end());
            histrory_embeddings.insert(histrory_embeddings.end(), generated_embeddings.begin(), generated_embeddings.end());

            for (auto& seq : sequence_group->get_sequences()) {
                if (seq->get_id() == idx0) {
                    continue;
                }
                scheduler.free_sequence(seq->get_id());
            }
         }
    }
}

TEST(TestScheduler, expected_num_scheduled_tokens_overrides_default_schedule) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 8;
    scheduler_config.num_kv_blocks = 10;
    scheduler_config.dynamic_split_fuse = true;
    scheduler_config.max_num_seqs = 5;

    std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const uint64_t request_id = 42;
    SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(
        request_id,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    std::vector<SequenceGroup::Ptr> requests = {sequence_group};

    Scheduler scheduler = Scheduler(init_cache_orchestrator(scheduler_config), scheduler_config);

    scheduler.set_expected_num_scheduled_tokens(request_id, 5);
    EXPECT_EQ(scheduler.get_expected_num_scheduled_tokens(request_id), 5);

    auto out = scheduler.schedule(requests);
    EXPECT_EQ(out.m_total_num_scheduled_tokens, 5);
    EXPECT_FALSE(out.m_scheduled_sequence_groups_ids.empty());

    // Release scheduled sequences and acknowledge the iteration so that
    // Scheduler / BlockManager state is consistent at destruction time.
    for (auto& seq : sequence_group->get_sequences()) {
        scheduler.free_sequence(seq->get_id());
    }
    sequence_group->finish_iteration();
}

TEST(TestScheduler, expected_num_scheduled_tokens_does_not_override_if_greater_than_available) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 8;
    scheduler_config.num_kv_blocks = 10;
    scheduler_config.dynamic_split_fuse = true;
    scheduler_config.max_num_seqs = 5;

    std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const uint64_t request_id = 43;
    SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(
        request_id,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    std::vector<SequenceGroup::Ptr> requests = {sequence_group};

    Scheduler scheduler = Scheduler(init_cache_orchestrator(scheduler_config), scheduler_config);

    // Available tokens for the request are 12; expected value above it must be ignored.
    scheduler.set_expected_num_scheduled_tokens(request_id, 13);

    auto out = scheduler.schedule(requests);
    // Default scheduling is min(max_num_batched_tokens, available_tokens) = min(8, 12) = 8.
    EXPECT_EQ(out.m_total_num_scheduled_tokens, 8);
    EXPECT_FALSE(out.m_scheduled_sequence_groups_ids.empty());

    for (auto& seq : sequence_group->get_sequences()) {
        scheduler.free_sequence(seq->get_id());
    }
    sequence_group->finish_iteration();
}

TEST(TestScheduler, clear_expected_num_scheduled_tokens_restores_default_schedule) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 8;
    scheduler_config.num_kv_blocks = 10;
    scheduler_config.dynamic_split_fuse = true;
    scheduler_config.max_num_seqs = 5;

    std::vector<int64_t> tokens = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const uint64_t request_id = 44;
    SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(
        request_id,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    std::vector<SequenceGroup::Ptr> requests = {sequence_group};

    Scheduler scheduler = Scheduler(init_cache_orchestrator(scheduler_config), scheduler_config);

    scheduler.set_expected_num_scheduled_tokens(request_id, 5);
    auto out1 = scheduler.schedule(requests);
    EXPECT_EQ(out1.m_total_num_scheduled_tokens, 5);

    requests[0]->finish_iteration();

    scheduler.clear_expected_num_scheduled_tokens(request_id);
    EXPECT_EQ(scheduler.get_expected_num_scheduled_tokens(request_id), 0);

    auto out2 = scheduler.schedule(requests);
    // 7 prompt tokens remain after the first scheduling; default scheduling should now apply.
    EXPECT_EQ(out2.m_total_num_scheduled_tokens, 7);
    EXPECT_FALSE(out2.m_scheduled_sequence_groups_ids.empty());

    for (auto& seq : sequence_group->get_sequences()) {
        scheduler.free_sequence(seq->get_id());
    }
    sequence_group->finish_iteration();
}

// ---------------------------------------------------------------------------
// Speculative linear attention with shared scratch rows.
// ---------------------------------------------------------------------------
namespace {
SchedulerConfig make_speculative_linear_attention_scheduler_config() {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 64;
    scheduler_config.num_kv_blocks = 64;
    scheduler_config.num_linear_attention_blocks = 16;
    scheduler_config.enable_prefix_caching = false;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 4;
    return scheduler_config;
}

// Prompt-processes one sequence and leaves it ready for verification.
SequenceGroup::Ptr make_prompt_processed_sequence_group(Scheduler& scheduler,
                                                       std::vector<SequenceGroup::Ptr>& requests,
                                                       const std::vector<uint64_t>& tokens) {
    SequenceGroup::Ptr seq_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {tokens.size()}, const_cast<uint64_t*>(tokens.data())),
        utils::get_greedy_config());
    requests.push_back(seq_group);

    std::ignore = scheduler.schedule(requests);
    seq_group->finish_iteration();
    seq_group->get_running_sequences()[0]->append_token(42, 0.9f);
    seq_group->update_processed_tokens_num(tokens.size());
    return seq_group;
}
}  // namespace

TEST(TestScheduler, hybrid_non_prefix_linear_attention_plain_step_rejects_outstanding_scratch) {
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                       TEST_BLOCK_SIZE,
                                                       /*kv_num_layers=*/1,
                                                       /*la_num_layers=*/1);
    auto& la_block_manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);

    std::vector<SequenceGroup::Ptr> requests;
    auto seq_group = make_prompt_processed_sequence_group(scheduler, requests, {0, 1, 2, 3});
    const auto seq_id = seq_group->get_running_sequences()[0]->get_id();
    ASSERT_EQ(seq_group->get_num_tokens_to_validate(), 0u);

    const auto borrowed = orchestrator->reserve_linear_attention_temporary_blocks(seq_id, 2);
    ASSERT_EQ(borrowed.size(), 2u);
    ASSERT_TRUE(la_block_manager.has_temporary_blocks(seq_id));

    EXPECT_THROW(std::ignore = scheduler.schedule(requests), ov::Exception);
    EXPECT_TRUE(la_block_manager.has_temporary_blocks(seq_id))
        << "plain paging silently released scratch instead of reporting the violated invariant";

    scheduler.release_linear_attention_checkpoints(seq_id);
    EXPECT_FALSE(la_block_manager.has_temporary_blocks(seq_id));
    scheduler.free_sequence(seq_id);
}

// The paging window contains one committed row and N+1 distinct borrowed rows.
TEST(TestScheduler, hybrid_non_prefix_linear_attention_borrowed_speculative_emits_committed_plus_temporaries) {
    constexpr size_t N = 3;
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                       TEST_BLOCK_SIZE,
                                                       /*kv_num_layers=*/1,
                                                       /*la_num_layers=*/1);
    auto& la_block_manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    scheduler.ensure_linear_attention_pool_blocks(1 + (1 + N));

    std::vector<SequenceGroup::Ptr> requests;
    auto seq_group = make_prompt_processed_sequence_group(scheduler, requests, {0, 1, 2, 3});
    const auto seq_id = seq_group->get_running_sequences()[0]->get_id();

    ASSERT_EQ(orchestrator->get_linear_attention_block_table(seq_id).size(), 1u);

    const size_t committed = orchestrator->get_linear_attention_live_block(seq_id);
    seq_group->set_num_validated_tokens(N);

    auto out = scheduler.schedule(requests);
    ASSERT_TRUE(out.has_linear_attention_paging_data(seq_id));
    const auto& paging_data = out.get_linear_attention_paging_data(seq_id);

    ASSERT_EQ(paging_data.block_indices.size(), N + 2);
    EXPECT_EQ(static_cast<size_t>(paging_data.block_indices[0]), committed);
    std::set<int32_t> seen = {paging_data.block_indices[0]};
    for (size_t i = 1; i < paging_data.block_indices.size(); ++i) {
        EXPECT_NE(paging_data.block_indices[i], paging_data.block_indices[0])
            << "borrowed row " << i << " aliases the committed row";
        EXPECT_TRUE(seen.insert(paging_data.block_indices[i]).second) << "duplicate borrowed row " << i;
    }
    EXPECT_EQ(paging_data.cache_interval, 1);
    EXPECT_TRUE(paging_data.is_speculative);
    EXPECT_EQ(paging_data.num_processed_tokens_before, seq_group->get_num_processed_tokens());
    EXPECT_EQ(paging_data.num_processed_tokens_before, 4u);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id).size(), 1u);
    EXPECT_TRUE(la_block_manager.has_temporary_blocks(seq_id));

    scheduler.release_linear_attention_checkpoints(seq_id);
    EXPECT_FALSE(la_block_manager.has_temporary_blocks(seq_id));
    for (auto& seq : seq_group->get_sequences()) {
        scheduler.free_sequence(seq->get_id());
    }
}

// Admission grows the shared pool without changing per-sequence ownership.
TEST(TestScheduler, hybrid_non_prefix_linear_attention_borrowed_admission_reservation_grows_pool_not_owned_rows) {
    constexpr size_t N = 2;
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
    scheduler_config.num_linear_attention_blocks = 2;

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                       TEST_BLOCK_SIZE,
                                                       /*kv_num_layers=*/1,
                                                       /*la_num_layers=*/1);
    auto& la_block_manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
    ASSERT_EQ(la_block_manager.get_fixed_blocks_per_sequence(), 1u);
    ASSERT_EQ(la_block_manager.get_max_total_block_count(), 0u) << "this test is about unbounded growth";
    const size_t initial_pool = la_block_manager.get_total_block_count();
    ASSERT_LT(initial_pool, 1 + (1 + N));

    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    EXPECT_TRUE(scheduler.ensure_linear_attention_pool_blocks(1 + (1 + N)));
    EXPECT_EQ(la_block_manager.get_total_block_count(), 1 + (1 + N));
    EXPECT_FALSE(scheduler.ensure_linear_attention_pool_blocks(1 + (1 + N)));
    EXPECT_EQ(la_block_manager.get_fixed_blocks_per_sequence(), 1u);

    std::vector<SequenceGroup::Ptr> requests;
    auto seq_group = make_prompt_processed_sequence_group(scheduler, requests, {0, 1, 2, 3});
    const auto seq_id = seq_group->get_running_sequences()[0]->get_id();
    seq_group->set_num_validated_tokens(N);

    auto out = scheduler.schedule(requests);
    ASSERT_TRUE(out.has_linear_attention_paging_data(seq_id));
    const auto& paging_data = out.get_linear_attention_paging_data(seq_id);
    ASSERT_EQ(paging_data.block_indices.size(), N + 2);
    EXPECT_TRUE(paging_data.is_speculative);
    EXPECT_EQ(la_block_manager.get_num_blocks_in_use(), 1 + (1 + N));

    scheduler.release_linear_attention_checkpoints(seq_id);
    for (auto& seq : seq_group->get_sequences()) {
        scheduler.free_sequence(seq->get_id());
    }
}

// Mixed commit advances return every borrowed row.
TEST(TestScheduler, hybrid_non_prefix_linear_attention_borrowed_steady_state_returns_pool_rows_each_step) {
    constexpr size_t N = 3;
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                       TEST_BLOCK_SIZE,
                                                       /*kv_num_layers=*/1,
                                                       /*la_num_layers=*/1);
    auto& la_block_manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    scheduler.ensure_linear_attention_pool_blocks(1 + (1 + N));

    std::vector<SequenceGroup::Ptr> requests;
    auto seq_group = make_prompt_processed_sequence_group(scheduler, requests, {0, 1, 2, 3});
    auto sequence = seq_group->get_running_sequences()[0];
    const auto seq_id = sequence->get_id();

    ASSERT_EQ(orchestrator->get_linear_attention_block_table(seq_id).size(), 1u);
    const size_t committed_only_in_use = la_block_manager.get_num_blocks_in_use();
    ASSERT_EQ(committed_only_in_use, 1u);

    const std::vector<size_t> advances = {1, N + 1, 2, N + 1, 1};
    size_t processed = seq_group->get_num_processed_tokens();
    size_t prev_committed = orchestrator->get_linear_attention_live_block(seq_id);

    for (size_t step = 0; step < advances.size(); ++step) {
        seq_group->set_num_validated_tokens(N);

        auto out = run_one_speculative_step(scheduler, requests);
        ASSERT_TRUE(out.has_linear_attention_paging_data(seq_id));
        const auto& pd = out.get_linear_attention_paging_data(seq_id);
        ASSERT_TRUE(pd.is_speculative);
        ASSERT_EQ(pd.block_indices.size(), N + 2);

        EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id).size(), 1u)
            << "owned LA set changed size at step " << step;
        EXPECT_EQ(static_cast<size_t>(pd.block_indices[0]), prev_committed);
        EXPECT_EQ(la_block_manager.get_num_blocks_in_use(), 1 + (1 + N))
            << "unexpected pool occupancy while verifying at step " << step;
        std::set<int32_t> seen = {pd.block_indices[0]};
        for (size_t i = 1; i < pd.block_indices.size(); ++i) {
            EXPECT_TRUE(seen.insert(pd.block_indices[i]).second) << "duplicate row at step " << step;
        }

        const size_t advance = advances[step];
        const int32_t chosen = pd.block_indices[advance];
        scheduler.promote_linear_attention_checkpoint(seq_id, advance);

        EXPECT_EQ(la_block_manager.get_num_blocks_in_use(), committed_only_in_use)
            << "borrowed rows leaked at step " << step;

        seq_group->finish_iteration();
        processed += advance;
        seq_group->update_processed_tokens_num(processed);
        sequence->append_token(100 + static_cast<int64_t>(step), 0.9f);

        prev_committed = static_cast<size_t>(chosen);
        EXPECT_EQ(orchestrator->get_linear_attention_live_block(seq_id), prev_committed)
            << "committed row and block table diverged at step " << step;
    }

    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id).size(), 1u);

    for (auto& seq : seq_group->get_sequences()) {
        scheduler.free_sequence(seq->get_id());
    }
}

// Releasing borrowed rows leaves the committed row unchanged.
TEST(TestScheduler, linear_attention_borrowed_release_keeps_committed_row) {
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                       TEST_BLOCK_SIZE,
                                                       /*kv_num_layers=*/1,
                                                       /*la_num_layers=*/1);
    auto& la_block_manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);

    std::vector<uint64_t> tokens = {0, 1, 2, 3};
    SequenceGroup::Ptr seq_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    auto seq = seq_group->get_running_sequences()[0];
    orchestrator->allocate_tokens(seq, seq_group, 1, seq_group->get_prompt_len());
    const uint64_t seq_id = seq->get_id();
    const size_t committed = orchestrator->get_linear_attention_live_block(seq_id);

    const auto borrowed = orchestrator->reserve_linear_attention_temporary_blocks(seq_id, 4);
    ASSERT_EQ(borrowed.size(), 4u);
    EXPECT_TRUE(la_block_manager.has_temporary_blocks(seq_id));

    orchestrator->release_linear_attention_temporary_blocks(seq_id);
    EXPECT_FALSE(la_block_manager.has_temporary_blocks(seq_id));
    EXPECT_EQ(orchestrator->get_linear_attention_live_block(seq_id), committed);
    EXPECT_EQ(la_block_manager.get_num_blocks_in_use(), 1u);
    EXPECT_THROW(orchestrator->promote_linear_attention_temporary_block(seq_id, 0), ov::Exception);

    orchestrator->free_sequence(seq_id);
}

// Shared-pool shortages must defer before reservation (ADR 0004 I4).
namespace {
// Prompt-processes several sequences and leaves them ready for a speculative step.
std::vector<SequenceGroup::Ptr> make_prompt_processed_sequence_groups(Scheduler& scheduler,
                                                                     std::vector<SequenceGroup::Ptr>& requests,
                                                                     size_t num_groups,
                                                                     const std::vector<uint64_t>& tokens) {
    std::vector<SequenceGroup::Ptr> groups;
    for (size_t idx = 0; idx < num_groups; ++idx) {
        SequenceGroup::Ptr seq_group = std::make_shared<SequenceGroup>(
            idx,
            ov::Tensor(ov::element::i64, {tokens.size()}, const_cast<uint64_t*>(tokens.data())),
            utils::get_greedy_config());
        groups.push_back(seq_group);
        requests.push_back(seq_group);
    }

    std::ignore = scheduler.schedule(requests);
    for (const auto& seq_group : groups) {
        EXPECT_EQ(seq_group->get_num_scheduled_tokens(), tokens.size())
            << "prompt phase did not schedule request " << seq_group->get_request_id() << " in one shot";
        seq_group->finish_iteration();
        seq_group->get_running_sequences()[0]->append_token(42, 0.9f);
        seq_group->update_processed_tokens_num(tokens.size());
    }
    return groups;
}
}  // namespace

// With room for one speculative window, the second sequence defers until those rows are released.
TEST(TestScheduler, hybrid_non_prefix_linear_attention_borrowed_speculative_window_deferred_when_pool_is_short) {
    constexpr size_t N = 2;
    constexpr size_t WINDOW = N + 1;  // scheduled tokens == borrowed rows
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
    // Keep the LA pool as the only binding limit.
    scheduler_config.max_num_batched_tokens = 256;
    scheduler_config.num_kv_blocks = 256;
    // Two committed rows plus one borrowed window.
    scheduler_config.num_linear_attention_blocks = 2 + WINDOW;

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                       TEST_BLOCK_SIZE,
                                                       /*kv_num_layers=*/1,
                                                       /*la_num_layers=*/1);
    auto& la_block_manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
    ASSERT_EQ(la_block_manager.get_total_block_count(), 2 + WINDOW);
    ASSERT_EQ(la_block_manager.get_fixed_blocks_per_sequence(), 1u);

    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    std::vector<SequenceGroup::Ptr> requests;
    auto groups = make_prompt_processed_sequence_groups(scheduler, requests, 2, {0, 1, 2, 3});
    auto seq_group_a = groups[0];
    auto seq_group_b = groups[1];
    const auto seq_id_a = seq_group_a->get_running_sequences()[0]->get_id();
    const auto seq_id_b = seq_group_b->get_running_sequences()[0]->get_id();
    ASSERT_EQ(la_block_manager.get_num_blocks_in_use(), 2u);
    ASSERT_EQ(la_block_manager.num_free_blocks(), WINDOW);

    seq_group_a->set_num_validated_tokens(N);
    seq_group_b->set_num_validated_tokens(N);

    // The first sequence borrows the only window.
    Scheduler::Output out1;
    ASSERT_NO_THROW(out1 = scheduler.schedule(requests));
    EXPECT_EQ(seq_group_a->get_num_scheduled_tokens(), WINDOW);
    ASSERT_TRUE(out1.has_linear_attention_paging_data(seq_id_a));
    EXPECT_EQ(out1.get_linear_attention_paging_data(seq_id_a).block_indices.size(), N + 2);
    EXPECT_TRUE(la_block_manager.has_temporary_blocks(seq_id_a));

    EXPECT_EQ(seq_group_b->get_num_scheduled_tokens(), 0u);
    EXPECT_FALSE(out1.has_linear_attention_paging_data(seq_id_b));
    EXPECT_EQ(out1.m_scheduled_sequence_groups_ids, std::vector<uint64_t>({0}));
    EXPECT_EQ(out1.m_total_num_scheduled_tokens, WINDOW);
    EXPECT_FALSE(la_block_manager.has_temporary_blocks(seq_id_b));
    EXPECT_EQ(la_block_manager.get_num_blocks_in_use(), 2 + WINDOW);

    // finish_iteration clears m_num_validation_tokens, so re-arm both windows.
    seq_group_a->finish_iteration();
    seq_group_a->set_num_validated_tokens(N);
    seq_group_b->set_num_validated_tokens(N);

    // Both defer while the first sequence still holds its borrowed rows.
    Scheduler::Output out2;
    ASSERT_NO_THROW(out2 = scheduler.schedule(requests));
    EXPECT_EQ(out2.m_total_num_scheduled_tokens, 0u);
    EXPECT_TRUE(out2.m_scheduled_sequence_groups_ids.empty());
    EXPECT_EQ(seq_group_a->get_num_scheduled_tokens(), 0u);
    EXPECT_EQ(seq_group_b->get_num_scheduled_tokens(), 0u);
    EXPECT_FALSE(out2.has_linear_attention_paging_data(seq_id_a));
    EXPECT_FALSE(out2.has_linear_attention_paging_data(seq_id_b));

    // Freeing a sequence also releases its borrowed rows.
    seq_group_a->get_running_sequences()[0]->set_status(SequenceStatus::FINISHED);
    scheduler.free_sequence(seq_id_a);
    clear_finished_sequences(requests);
    ASSERT_EQ(requests.size(), 1u);
    ASSERT_EQ(la_block_manager.get_num_blocks_in_use(), 1u);

    Scheduler::Output out3;
    ASSERT_NO_THROW(out3 = scheduler.schedule(requests));
    EXPECT_EQ(seq_group_b->get_num_scheduled_tokens(), WINDOW);
    ASSERT_TRUE(out3.has_linear_attention_paging_data(seq_id_b));
    const auto& paging_b = out3.get_linear_attention_paging_data(seq_id_b);
    EXPECT_EQ(paging_b.block_indices.size(), N + 2);
    EXPECT_TRUE(paging_b.is_speculative);
    EXPECT_EQ(out3.m_scheduled_sequence_groups_ids, std::vector<uint64_t>({0}));

    scheduler.release_linear_attention_checkpoints(seq_id_b);
    seq_group_b->finish_iteration();
    for (auto& seq : seq_group_b->get_sequences()) {
        scheduler.free_sequence(seq->get_id());
    }
}

TEST(TestScheduler, hybrid_non_prefix_linear_attention_partial_group_reservation_rolls_back_captured_sequences) {
    constexpr size_t N = 2;
    constexpr size_t WINDOW = N + 1;
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
    scheduler_config.max_num_batched_tokens = 256;
    scheduler_config.num_kv_blocks = 256;
    // The shared committed row leaves room for exactly one of the two sequence reservations.
    scheduler_config.num_linear_attention_blocks = 1 + WINDOW;

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                       TEST_BLOCK_SIZE,
                                                       /*kv_num_layers=*/1,
                                                       /*la_num_layers=*/1);
    auto& la_block_manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);

    std::vector<SequenceGroup::Ptr> requests;
    auto seq_group = make_prompt_processed_sequence_group(scheduler, requests, {0, 1, 2, 3});
    auto parent = seq_group->get_running_sequences()[0];
    auto child = seq_group->fork_sequence(parent);
    scheduler.fork_sequence(parent->get_id(), child->get_id());
    ASSERT_EQ(seq_group->num_running_seqs(), 2u);
    ASSERT_EQ(la_block_manager.num_free_blocks(), WINDOW);

    seq_group->set_num_validated_tokens(N);
    const auto out = scheduler.schedule(requests);

    EXPECT_EQ(out.m_total_num_scheduled_tokens, 0u);
    EXPECT_EQ(seq_group->get_num_scheduled_tokens(), 0u);
    EXPECT_FALSE(out.has_linear_attention_paging_data(parent->get_id()));
    EXPECT_FALSE(out.has_linear_attention_paging_data(child->get_id()));
    EXPECT_FALSE(la_block_manager.has_temporary_blocks(parent->get_id()));
    EXPECT_FALSE(la_block_manager.has_temporary_blocks(child->get_id()));
    EXPECT_EQ(la_block_manager.get_num_sequences_with_temporary_blocks(), 0u);
    EXPECT_EQ(la_block_manager.num_free_blocks(), WINDOW);

    scheduler.free_sequence(child->get_id());
    scheduler.free_sequence(parent->get_id());
}

// The capacity predicate must reject every condition that would make reservation fail.
TEST(TestScheduler, linear_attention_can_reserve_temporary_blocks_agrees_with_reservation) {
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
    scheduler_config.num_linear_attention_blocks = 4;

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                       TEST_BLOCK_SIZE,
                                                       /*kv_num_layers=*/1,
                                                       /*la_num_layers=*/1);
    auto& la_block_manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);

    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    std::vector<SequenceGroup::Ptr> requests;
    auto seq_group = make_prompt_processed_sequence_group(scheduler, requests, {0, 1, 2, 3});
    const auto seq_id = seq_group->get_running_sequences()[0]->get_id();
    ASSERT_EQ(la_block_manager.get_num_blocks_in_use(), 1u);
    const size_t free_rows = la_block_manager.num_free_blocks();
    ASSERT_EQ(free_rows, 3u);

    constexpr uint64_t unknown_seq_id = 4242;
    EXPECT_FALSE(la_block_manager.can_reserve_temporary_blocks(unknown_seq_id, 1));
    EXPECT_FALSE(orchestrator->can_reserve_linear_attention_temporary_blocks(unknown_seq_id, 1));
    EXPECT_FALSE(scheduler.can_reserve_linear_attention_checkpoints(unknown_seq_id, 1));
    EXPECT_FALSE(la_block_manager.has_temporary_blocks(unknown_seq_id));
    EXPECT_EQ(la_block_manager.get_num_sequences_with_temporary_blocks(), 0u);

    EXPECT_TRUE(la_block_manager.can_reserve_temporary_blocks(seq_id, free_rows));
    EXPECT_TRUE(scheduler.can_reserve_linear_attention_checkpoints(seq_id, free_rows));
    EXPECT_FALSE(la_block_manager.can_reserve_temporary_blocks(seq_id, free_rows + 1));
    EXPECT_FALSE(scheduler.can_reserve_linear_attention_checkpoints(seq_id, free_rows + 1));
    EXPECT_FALSE(orchestrator->can_reserve_linear_attention_temporary_blocks(seq_id, 0));
    EXPECT_THROW(std::ignore = orchestrator->reserve_linear_attention_temporary_blocks(seq_id, 0), ov::Exception);

    std::vector<int> borrowed;
    ASSERT_NO_THROW(borrowed = la_block_manager.reserve_temporary_blocks(seq_id, free_rows));
    EXPECT_EQ(borrowed.size(), free_rows);

    EXPECT_FALSE(la_block_manager.can_reserve_temporary_blocks(seq_id, 1));
    EXPECT_FALSE(scheduler.can_reserve_linear_attention_checkpoints(seq_id, 1));
    EXPECT_THROW(std::ignore = la_block_manager.reserve_temporary_blocks(seq_id, 1), ov::Exception);

    la_block_manager.release_temporary_blocks(seq_id);
    EXPECT_TRUE(la_block_manager.can_reserve_temporary_blocks(seq_id, free_rows));

    EXPECT_FALSE(la_block_manager.can_reserve_temporary_blocks(seq_id, free_rows + 1));
    EXPECT_THROW(std::ignore = la_block_manager.reserve_temporary_blocks(seq_id, free_rows + 1), ov::Exception);
    EXPECT_FALSE(la_block_manager.has_temporary_blocks(seq_id));
    EXPECT_EQ(la_block_manager.get_num_sequences_with_temporary_blocks(), 0u);

    for (auto& seq : seq_group->get_sequences()) {
        scheduler.free_sequence(seq->get_id());
    }

    // Temporary rows require the shared single-layer block table.
    std::vector<uint64_t> tokens = {0, 1, 2, 3};
    SequenceGroup::Ptr multi_layer_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
        utils::get_greedy_config());
    auto multi_layer_sequence = multi_layer_group->get_running_sequences()[0];
    BlockManager multi_layer_manager(/*num_blocks=*/8,
                                    /*enable_prefix_caching=*/false,
                                    /*block_size=*/1,
                                    /*num_layers=*/2,
                                    /*fixed_blocks_per_sequence=*/1);
    multi_layer_manager.allocate_tokens(multi_layer_sequence,
                                        multi_layer_group,
                                        1,
                                        multi_layer_group->get_prompt_len());
    const uint64_t multi_layer_seq_id = multi_layer_sequence->get_id();
    ASSERT_TRUE(multi_layer_manager.has_block_table(multi_layer_seq_id));
    EXPECT_FALSE(multi_layer_manager.can_reserve_temporary_blocks(multi_layer_seq_id, 1));
    EXPECT_THROW(std::ignore = multi_layer_manager.reserve_temporary_blocks(multi_layer_seq_id, 1), ov::Exception);
    multi_layer_manager.free_sequence(multi_layer_seq_id);
}

// num_linear_attention_blocks is a hard ceiling when explicitly configured.

// Admission pre-sizing clamps to the configured ceiling; scheduling then defers excess windows.
TEST(TestScheduler, hybrid_non_prefix_linear_attention_borrowed_admission_reservation_clamped_to_configured_pool_budget) {
    constexpr size_t N = 2;
    constexpr size_t WINDOW = N + 1;
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
    scheduler_config.max_num_batched_tokens = 256;
    scheduler_config.num_kv_blocks = 256;
    // Two committed rows plus one borrowed window.
    scheduler_config.num_linear_attention_blocks = 2 + WINDOW;

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                       TEST_BLOCK_SIZE,
                                                       /*kv_num_layers=*/1,
                                                       /*la_num_layers=*/1,
                                                       /*cap_la_pool=*/true);
    auto& la_block_manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
    ASSERT_EQ(la_block_manager.get_max_total_block_count(), 2 + WINDOW);
    ASSERT_EQ(la_block_manager.get_total_block_count(), 2 + WINDOW);

    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    const size_t worst_case_pool = 2 * (1 + WINDOW);
    ASSERT_GT(worst_case_pool, 2 + WINDOW);
    ASSERT_NO_THROW(std::ignore = scheduler.ensure_linear_attention_pool_blocks(worst_case_pool));
    EXPECT_EQ(la_block_manager.get_total_block_count(), 2 + WINDOW)
        << "admission pre-sizing grew the pool past its configured budget";
    EXPECT_FALSE(scheduler.ensure_linear_attention_pool_blocks(worst_case_pool));

    std::vector<SequenceGroup::Ptr> requests;
    auto groups = make_prompt_processed_sequence_groups(scheduler, requests, 2, {0, 1, 2, 3});
    const auto seq_id_a = groups[0]->get_running_sequences()[0]->get_id();
    const auto seq_id_b = groups[1]->get_running_sequences()[0]->get_id();
    EXPECT_EQ(la_block_manager.get_total_block_count(), 2 + WINDOW);

    groups[0]->set_num_validated_tokens(N);
    groups[1]->set_num_validated_tokens(N);

    Scheduler::Output out;
    ASSERT_NO_THROW(out = scheduler.schedule(requests));
    EXPECT_EQ(groups[0]->get_num_scheduled_tokens(), WINDOW);
    EXPECT_EQ(groups[1]->get_num_scheduled_tokens(), 0u);
    EXPECT_TRUE(la_block_manager.has_temporary_blocks(seq_id_a));
    EXPECT_FALSE(la_block_manager.has_temporary_blocks(seq_id_b));
    EXPECT_EQ(la_block_manager.get_total_block_count(), 2 + WINDOW)
        << "the scheduling step grew the pool past its configured budget";

    scheduler.release_linear_attention_checkpoints(seq_id_a);
    for (const auto& seq_group : groups) {
        seq_group->finish_iteration();
        for (auto& seq : seq_group->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

// Every pool growth path must honor the configured ceiling.
TEST(TestScheduler, linear_attention_pool_budget_bounds_every_growth_path) {
    constexpr size_t BUDGET = 4;
    BlockManager capped(/*num_blocks=*/2,
                        /*enable_prefix_caching=*/false,
                        /*block_size=*/1,
                        /*num_layers=*/1,
                        /*fixed_blocks_per_sequence=*/1,
                        /*restore_latest_prefix_block_only=*/false,
                        /*max_total_blocks=*/BUDGET);
    ASSERT_EQ(capped.get_max_total_block_count(), BUDGET);

    EXPECT_TRUE(capped.can_increase_block_count_to(BUDGET));
    EXPECT_FALSE(capped.can_increase_block_count_to(BUDGET + 1));

    EXPECT_TRUE(capped.increase_block_count_up_to(BUDGET + 10));
    EXPECT_EQ(capped.get_total_block_count(), BUDGET);
    EXPECT_FALSE(capped.increase_block_count_up_to(BUDGET + 10));
    EXPECT_EQ(capped.get_total_block_count(), BUDGET);

    // The return value terminates the scheduler's cache-growth loop.
    EXPECT_FALSE(capped.grow_capacity_by_tokens(64));
    EXPECT_EQ(capped.get_total_block_count(), BUDGET);
    capped.ensure_sequence_token_capacity({{64, 4}});
    EXPECT_EQ(capped.get_total_block_count(), BUDGET);

    // The exact-size API cannot silently clamp a caller-computed target.
    EXPECT_THROW(capped.increase_block_count(BUDGET + 1), ov::Exception);
    EXPECT_EQ(capped.get_total_block_count(), BUDGET);

    EXPECT_THROW(BlockManager(/*num_blocks=*/8,
                              /*enable_prefix_caching=*/false,
                              /*block_size=*/1,
                              /*num_layers=*/1,
                              /*fixed_blocks_per_sequence=*/1,
                              /*restore_latest_prefix_block_only=*/false,
                              /*max_total_blocks=*/4),
                 ov::Exception);

    // Zero means no ceiling.
    BlockManager uncapped(/*num_blocks=*/2, /*enable_prefix_caching=*/false, /*block_size=*/1);
    EXPECT_EQ(uncapped.get_max_total_block_count(), 0u);
    EXPECT_TRUE(uncapped.can_increase_block_count_to(1u << 20));
    EXPECT_TRUE(uncapped.increase_block_count_up_to(64));
    EXPECT_EQ(uncapped.get_total_block_count(), 64u);
}

// Pool growth changes the footprint high-water mark without increasing occupancy.
TEST(TestScheduler, linear_attention_pool_blocks_high_water_tracks_growth_that_occupancy_metrics_miss) {
    constexpr size_t N = 2;
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
    scheduler_config.num_linear_attention_blocks = 2;

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                       TEST_BLOCK_SIZE,
                                                       /*kv_num_layers=*/1,
                                                       /*la_num_layers=*/1);
    auto& la_block_manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);

    std::vector<SequenceGroup::Ptr> requests;
    auto seq_group = make_prompt_processed_sequence_group(scheduler, requests, {0, 1, 2, 3});
    const auto seq_id = seq_group->get_running_sequences()[0]->get_id();

    orchestrator->sample_linear_attention_pool_blocks_high_water();
    const size_t pool_before = scheduler.get_linear_attention_pool_blocks_high_water();
    const size_t in_use_before = la_block_manager.get_num_blocks_in_use();
    const float usage_before = la_block_manager.get_used_percentage();
    ASSERT_EQ(pool_before, 2u);
    ASSERT_EQ(in_use_before, 1u);

    ASSERT_TRUE(scheduler.ensure_linear_attention_pool_blocks(2 + 4 * (1 + N)));
    orchestrator->sample_linear_attention_pool_blocks_high_water();
    const size_t peak_pool_blocks = scheduler.get_linear_attention_pool_blocks_high_water();

    EXPECT_EQ(peak_pool_blocks, 2 + 4 * (1 + N)) << "pool-size high-water missed the growth";
    EXPECT_GT(peak_pool_blocks, pool_before);
    EXPECT_EQ(la_block_manager.get_num_blocks_in_use(), in_use_before);
    EXPECT_LT(la_block_manager.get_used_percentage(), usage_before);

    // A high-water mark does not fall when rows are returned.
    seq_group->set_num_validated_tokens(N);
    std::ignore = scheduler.schedule(requests);
    scheduler.release_linear_attention_checkpoints(seq_id);
    orchestrator->sample_linear_attention_pool_blocks_high_water();
    EXPECT_EQ(scheduler.get_linear_attention_pool_blocks_high_water(), 2 + 4 * (1 + N));

    seq_group->finish_iteration();
    for (auto& seq : seq_group->get_sequences()) {
        scheduler.free_sequence(seq->get_id());
    }
}

// Admission requires S_live committed rows plus one speculative window.

// A window may fit by itself while the committed rows make the full requirement exceed the ceiling.
TEST(TestScheduler, hybrid_non_prefix_linear_attention_borrow_pool_budget_below_live_plus_window_asserts) {
    constexpr size_t N = 2;
    constexpr size_t WINDOW = 1 + N;   // 3
    constexpr size_t S_LIVE = 6;       // concurrently verifying sequences
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
    scheduler_config.max_num_seqs = S_LIVE;
    // One row short of the S_live + W floor.
    scheduler_config.num_linear_attention_blocks = S_LIVE + WINDOW - 1;

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                       TEST_BLOCK_SIZE,
                                                       /*kv_num_layers=*/1,
                                                       /*la_num_layers=*/1,
                                                       /*cap_la_pool=*/true);
    auto& la_block_manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
    ASSERT_EQ(la_block_manager.get_max_total_block_count(), S_LIVE + WINDOW - 1);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);

    // The window-only bound still passes.
    ASSERT_GE(la_block_manager.get_max_total_block_count(), WINDOW);

    EXPECT_THROW(scheduler.check_linear_attention_borrow_pool_floor(S_LIVE, WINDOW), ov::Exception);
    try {
        scheduler.check_linear_attention_borrow_pool_floor(S_LIVE, WINDOW);
        ADD_FAILURE() << "expected the borrow floor assert to fire";
    } catch (const ov::Exception& ex) {
        const std::string message(ex.what());
        EXPECT_NE(message.find(std::to_string(S_LIVE + WINDOW) + " rows"), std::string::npos) << message;
        EXPECT_NE(message.find("num_linear_attention_blocks"), std::string::npos) << message;
        // dynamic_split_fuse can make submitted request count exceed max_num_seqs.
        EXPECT_NE(message.find("concurrent requests"), std::string::npos) << message;
        EXPECT_NE(message.find("num_assistant_tokens"), std::string::npos) << message;
    }
}

// The exact S_live + W floor is accepted.
TEST(TestScheduler, hybrid_non_prefix_linear_attention_borrow_pool_budget_at_live_plus_window_is_accepted) {
    constexpr size_t N = 2;
    constexpr size_t WINDOW = 1 + N;
    constexpr size_t S_LIVE = 6;
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
    scheduler_config.max_num_seqs = S_LIVE;
    scheduler_config.num_linear_attention_blocks = S_LIVE + WINDOW;

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                       TEST_BLOCK_SIZE,
                                                       /*kv_num_layers=*/1,
                                                       /*la_num_layers=*/1,
                                                       /*cap_la_pool=*/true);
    ASSERT_EQ(orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE).get_max_total_block_count(),
              S_LIVE + WINDOW);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);

    EXPECT_NO_THROW(scheduler.check_linear_attention_borrow_pool_floor(S_LIVE, WINDOW));
    // dynamic_split_fuse does not clamp live sequences to max_num_seqs.
    EXPECT_THROW(scheduler.check_linear_attention_borrow_pool_floor(S_LIVE + 4, WINDOW), ov::Exception);
}

// An uncapped pool can grow to satisfy the admission floor.
TEST(TestScheduler, linear_attention_borrow_pool_floor_silent_while_pool_is_uncapped) {
    constexpr size_t WINDOW = 3;
    constexpr size_t S_LIVE = 6;
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
    scheduler_config.max_num_seqs = S_LIVE;
    scheduler_config.num_linear_attention_blocks = 1;

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                       TEST_BLOCK_SIZE,
                                                       /*kv_num_layers=*/1,
                                                       /*la_num_layers=*/1,
                                                       /*cap_la_pool=*/false);
    auto& la_block_manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
    ASSERT_EQ(la_block_manager.get_max_total_block_count(), 0u) << "this test is about the growing-pool case";
    ASSERT_LT(la_block_manager.get_total_block_count(), S_LIVE + WINDOW);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);

    EXPECT_NO_THROW(scheduler.check_linear_attention_borrow_pool_floor(S_LIVE, WINDOW));
    EXPECT_TRUE(scheduler.ensure_linear_attention_pool_blocks(S_LIVE + WINDOW));
    EXPECT_GE(la_block_manager.get_total_block_count(), S_LIVE + WINDOW);
}

// Non-verifying sequences also own committed rows and count toward S_live.
TEST(TestScheduler, hybrid_non_prefix_linear_attention_borrow_pool_floor_counts_non_verifying_live_sequences) {
    constexpr size_t N = 2;
    constexpr size_t WINDOW = 1 + N;          // 3
    constexpr size_t S_VERIFYING = 2;         // speculative sequences
    constexpr size_t S_PLAIN = 4;             // non-speculative sequences, one committed row each
    constexpr size_t S_LIVE = S_VERIFYING + S_PLAIN;  // 6
    constexpr size_t CEILING = 6;
    static_assert(S_VERIFYING + WINDOW <= CEILING, "the old verifying-only bound must accept this ceiling");
    static_assert(S_LIVE + WINDOW > CEILING, "the live-count bound must reject this ceiling");

    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
    scheduler_config.max_num_seqs = S_LIVE;
    scheduler_config.num_linear_attention_blocks = CEILING;

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                       TEST_BLOCK_SIZE,
                                                       /*kv_num_layers=*/1,
                                                       /*la_num_layers=*/1,
                                                       /*cap_la_pool=*/true);
    auto& la_block_manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
    ASSERT_EQ(la_block_manager.get_max_total_block_count(), CEILING);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);

    // Verifying-only arithmetic passes, but live-sequence arithmetic does not.
    EXPECT_NO_THROW(scheduler.check_linear_attention_borrow_pool_floor(S_VERIFYING, WINDOW));

    EXPECT_THROW(scheduler.check_linear_attention_borrow_pool_floor(S_LIVE, WINDOW), ov::Exception);
    try {
        scheduler.check_linear_attention_borrow_pool_floor(S_LIVE, WINDOW);
        ADD_FAILURE() << "expected the borrow floor assert to fire on the live-sequence count";
    } catch (const ov::Exception& ex) {
        const std::string message(ex.what());
        EXPECT_NE(message.find(std::to_string(S_LIVE) + " committed recurrent-state rows"), std::string::npos)
            << message;
        EXPECT_NE(message.find("one per concurrently live sequence"), std::string::npos) << message;
        EXPECT_NE(message.find(std::to_string(S_LIVE + WINDOW) + " rows"), std::string::npos) << message;
        EXPECT_NE(message.find("caps the whole pool at " + std::to_string(CEILING) + " rows"), std::string::npos)
            << message;
        EXPECT_NE(message.find("num_linear_attention_blocks"), std::string::npos) << message;
    }
}

// When every live sequence verifies, the old and new capacity expressions are identical.
TEST(TestScheduler, hybrid_non_prefix_linear_attention_borrow_pool_floor_homogeneous_boundary_unchanged) {
    constexpr size_t N = 2;
    constexpr size_t WINDOW = 1 + N;  // 3
    constexpr size_t S_LIVE = 6;      // every live sequence verifies
    constexpr size_t S_VERIFYING = S_LIVE;
    static_assert(S_LIVE + WINDOW == S_VERIFYING + WINDOW, "homogeneous floor must not move");
    static_assert(S_LIVE + S_VERIFYING * WINDOW == S_VERIFYING * (1 + WINDOW),
                  "homogeneous growth target must not move");

    {   // One row below the floor: rejected, as before.
        SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
        scheduler_config.max_num_seqs = S_LIVE;
        scheduler_config.num_linear_attention_blocks = S_LIVE + WINDOW - 1;
        auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                           TEST_BLOCK_SIZE,
                                                           /*kv_num_layers=*/1,
                                                           /*la_num_layers=*/1,
                                                           /*cap_la_pool=*/true);
        ASSERT_EQ(orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE).get_max_total_block_count(),
                  S_LIVE + WINDOW - 1);
        Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
        EXPECT_THROW(scheduler.check_linear_attention_borrow_pool_floor(S_LIVE, WINDOW), ov::Exception);
    }
    {   // Exactly at the floor: accepted, as before.
        SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
        scheduler_config.max_num_seqs = S_LIVE;
        scheduler_config.num_linear_attention_blocks = S_LIVE + WINDOW;
        auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                           TEST_BLOCK_SIZE,
                                                           /*kv_num_layers=*/1,
                                                           /*la_num_layers=*/1,
                                                           /*cap_la_pool=*/true);
        ASSERT_EQ(orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE).get_max_total_block_count(),
                  S_LIVE + WINDOW);
        Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
        EXPECT_NO_THROW(scheduler.check_linear_attention_borrow_pool_floor(S_LIVE, WINDOW));
    }
    {   // The new and old growth targets also request the same number of rows.
        SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
        scheduler_config.max_num_seqs = S_LIVE;
        scheduler_config.num_linear_attention_blocks = 1;
        auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                           TEST_BLOCK_SIZE,
                                                           /*kv_num_layers=*/1,
                                                           /*la_num_layers=*/1,
                                                           /*cap_la_pool=*/false);
        auto& la_block_manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
        Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
        EXPECT_TRUE(scheduler.ensure_linear_attention_pool_blocks(S_LIVE + S_VERIFYING * WINDOW));
        EXPECT_GE(la_block_manager.get_total_block_count(), S_VERIFYING * (1 + WINDOW));
        EXPECT_FALSE(scheduler.ensure_linear_attention_pool_blocks(S_VERIFYING * (1 + WINDOW)));
    }
}
