// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <numeric>
#include "openvino/runtime/core.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"
#include "sequence_group.hpp"
#include "continuous_batching/scheduler.hpp"
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

std::shared_ptr<CacheOrchestrator> init_cache_orchestrator(SchedulerConfig scheduler_config, size_t block_size = TEST_BLOCK_SIZE, size_t num_layers = 1) {
    ov::Core core = ov::Core();
    ov::InferRequest request = core.compile_model(get_dummy_model(core, TEST_NUM_DECODER_LAYERS)).create_infer_request();
    auto cache_manager = std::make_shared<KVCacheManager>(request);
    auto block_manager = std::make_shared<BlockManager>(scheduler_config.num_kv_blocks, scheduler_config.enable_prefix_caching, block_size, num_layers);
    auto orchestrator = std::make_shared<CacheOrchestrator>();
    std::vector<size_t> all_layers(num_layers);
    std::iota(all_layers.begin(), all_layers.end(), 0);
    orchestrator->register_cache_type(CacheType::KV_CACHE, cache_manager, block_manager, all_layers);
    return orchestrator;
}

std::shared_ptr<CacheOrchestrator> init_hybrid_cache_orchestrator(SchedulerConfig scheduler_config,
                                                                   size_t kv_block_size = TEST_BLOCK_SIZE,
                                                                   size_t kv_num_layers = 1,
                                                                   size_t la_num_layers = 1) {
    ov::Core core = ov::Core();
    ov::InferRequest request = core.compile_model(get_dummy_hybrid_model(core, kv_num_layers, la_num_layers)).create_infer_request();

    auto kv_cache_manager = std::make_shared<KVCacheManager>(request);
    auto kv_block_manager = std::make_shared<BlockManager>(scheduler_config.num_kv_blocks,
                                                           scheduler_config.enable_prefix_caching,
                                                           kv_block_size,
                                                           kv_num_layers);

    auto la_cache_manager = std::make_shared<LinearAttentionCacheManager>(request);
    std::shared_ptr<BlockManager> la_block_manager;
    if (scheduler_config.enable_prefix_caching) {
        la_block_manager = std::make_shared<BlockManager>(scheduler_config.num_linear_attention_blocks,
                                                          true,
                                                          scheduler_config.cache_interval,
                                                          la_cache_manager->get_num_layers());
    } else {
        const size_t num_la_blocks = scheduler_config.num_linear_attention_blocks > 0
                                         ? scheduler_config.num_linear_attention_blocks
                                         : (scheduler_config.num_kv_blocks > 0 ? scheduler_config.max_num_seqs : 0);
        la_block_manager = std::make_shared<BlockManager>(num_la_blocks,
                                                          false,
                                                          1,
                                                          la_cache_manager->get_num_layers(),
                                                          1);
    }

    auto orchestrator = std::make_shared<CacheOrchestrator>();
    std::vector<size_t> kv_layer_ids(kv_num_layers);
    std::iota(kv_layer_ids.begin(), kv_layer_ids.end(), 0);
    orchestrator->register_cache_type(CacheType::KV_CACHE, kv_cache_manager, kv_block_manager, kv_layer_ids);

    std::vector<size_t> la_layer_ids(la_cache_manager->get_num_layers());
    std::iota(la_layer_ids.begin(), la_layer_ids.end(), kv_num_layers);
    orchestrator->register_cache_type(CacheType::LINEAR_ATTENTION_CACHE,
                                      la_cache_manager,
                                      la_block_manager,
                                      la_layer_ids);
    return orchestrator;
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
        EXPECT_EQ(out1.m_block_tables[idx0][0].size(), 2);
        EXPECT_EQ(out1.m_block_tables[idx1][0].size(), 2);
        EXPECT_EQ(out1.m_block_tables[idx2][0].size(), 2);
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
        EXPECT_EQ(out3.m_block_tables[idx0][0].size(), 3);
        EXPECT_EQ(out3.m_block_tables[idx1][0].size(), 3);
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
        EXPECT_EQ(out4.m_block_tables[idx2][0].size(), 2);
        EXPECT_FALSE(out4.m_block_tables[idx2][0][0]->is_free());
        EXPECT_EQ(out4.m_block_tables[idx2][0][0]->get_index(), 0);
        EXPECT_FALSE(out4.m_block_tables[idx2][0][1]->is_free());
        EXPECT_EQ(out4.m_block_tables[idx2][0][1]->get_index(), 1);

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

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    Scheduler scheduler = Scheduler(orchestrator, scheduler_config);
    auto prompt_out = scheduler.schedule(requests);

    EXPECT_TRUE(prompt_out.m_linear_attention_paging_data.count(seq_id1));
    EXPECT_TRUE(prompt_out.m_linear_attention_paging_data.count(seq_id2));
    EXPECT_EQ(prompt_out.m_linear_attention_paging_data.at(seq_id1).block_indices.size(), 2);
    EXPECT_EQ(prompt_out.m_linear_attention_paging_data.at(seq_id1).block_indices[0], prompt_out.m_linear_attention_paging_data.at(seq_id1).block_indices[1]);
    EXPECT_EQ(prompt_out.m_linear_attention_paging_data.at(seq_id2).block_indices.size(), 2);
    EXPECT_EQ(prompt_out.m_linear_attention_paging_data.at(seq_id2).block_indices[0], prompt_out.m_linear_attention_paging_data.at(seq_id2).block_indices[1]);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id1).size(), 1);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id2).size(), 1);

    for (auto& req : requests) {
        auto running = req->get_running_sequences();
        running[0]->append_token(42, 0.9f);
        req->finish_iteration();
    }

    auto gen_out = scheduler.schedule(requests);
    EXPECT_TRUE(gen_out.m_linear_attention_paging_data.count(seq_id1));
    EXPECT_TRUE(gen_out.m_linear_attention_paging_data.count(seq_id2));
    EXPECT_EQ(gen_out.m_linear_attention_paging_data.at(seq_id1).block_indices.size(), 2);
    EXPECT_EQ(gen_out.m_linear_attention_paging_data.at(seq_id1).block_indices[0], gen_out.m_linear_attention_paging_data.at(seq_id1).block_indices[1]);
    EXPECT_EQ(gen_out.m_linear_attention_paging_data.at(seq_id2).block_indices.size(), 2);
    EXPECT_EQ(gen_out.m_linear_attention_paging_data.at(seq_id2).block_indices[0], gen_out.m_linear_attention_paging_data.at(seq_id2).block_indices[1]);
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

    ASSERT_TRUE(out.m_linear_attention_paging_data.count(seq_id));
    const auto& paging_data = out.m_linear_attention_paging_data.at(seq_id);
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
    EXPECT_TRUE(out.m_linear_attention_paging_data.count(seq_id1) || out.m_linear_attention_paging_data.count(seq_id2));
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

    EXPECT_EQ(orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE)->get_total_number_of_kv_blocks(), 0);
    std::ignore = scheduler.schedule(requests);
    EXPECT_EQ(orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE)->get_total_number_of_kv_blocks(), 2);

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, DISABLED_hybrid_runtime_arrival_beyond_initial_fixed_capacity_schedules_after_growth) {
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
    EXPECT_EQ(orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE)->get_total_number_of_kv_blocks(), 1);

    auto running = seq_group1->get_running_sequences();
    running[0]->append_token(42, 0.9f);
    seq_group1->finish_iteration();

    SequenceGroup::Ptr seq_group2 = std::make_shared<SequenceGroup>(1,
                                                                     ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                     utils::get_greedy_config());
    auto seq_id2 = seq_group2->get_running_sequences()[0]->get_id();
    requests.push_back(seq_group2);

    auto out = scheduler.schedule(requests);
    EXPECT_TRUE(out.m_linear_attention_paging_data.count(seq_id1));
    EXPECT_TRUE(out.m_linear_attention_paging_data.count(seq_id2));
}

TEST(TestScheduler, hybrid_prefix_caching_prefill_requires_read_and_interval_write_blocks) {
    // Target contract (cache_interval=128):
    // prefill requires 1 read block + ceil((processed % interval + scheduled) / interval) write blocks,
    // with the zero-state read reusing the first write block.
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 512;
    scheduler_config.num_kv_blocks = 128;
    scheduler_config.num_linear_attention_blocks = 32;
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 8;

    // Prompt length 260 => write blocks = ceil((0 + 260) / 128) = 3, plus one read block => 4 total.
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

    ASSERT_TRUE(out.m_linear_attention_paging_data.count(seq_id));
    const auto& paging_data = out.m_linear_attention_paging_data.at(seq_id);
    ASSERT_EQ(paging_data.block_indices.size(), 4);
    EXPECT_EQ(paging_data.past_length, 0);
    EXPECT_EQ(paging_data.cache_interval, DEFAULT_LINEAR_ATTENTION_CACHE_INTERVAL);
    EXPECT_EQ(paging_data.block_indices[0], paging_data.block_indices[1]);
    EXPECT_EQ(std::set<int32_t>(paging_data.block_indices.begin(), paging_data.block_indices.end()).size(), 3);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id).size(), 3);

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, hybrid_prefix_caching_prefill_uses_scheduler_config_cache_interval) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 128;
    scheduler_config.num_kv_blocks = 64;
    scheduler_config.num_linear_attention_blocks = 16;
    scheduler_config.cache_interval = 64;
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

    ASSERT_TRUE(out.m_linear_attention_paging_data.count(seq_id));
    const auto& paging_data = out.m_linear_attention_paging_data.at(seq_id);
    ASSERT_EQ(paging_data.block_indices.size(), 3);
    EXPECT_EQ(paging_data.cache_interval, 64);
    EXPECT_EQ(paging_data.block_indices[0], paging_data.block_indices[1]);
    EXPECT_NE(paging_data.block_indices[1], paging_data.block_indices[2]);
    EXPECT_EQ(orchestrator->get_block_size(CacheType::LINEAR_ATTENTION_CACHE), 64);
    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id).size(), 2);

    for (auto& req : requests) {
        for (auto& seq : req->get_sequences()) {
            scheduler.free_sequence(seq->get_id());
        }
    }
}

TEST(TestScheduler, scheduler_config_zero_cache_interval_requires_disabled_prefix_caching) {
    SchedulerConfig scheduler_config;
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.cache_interval = 0;

    EXPECT_ANY_THROW(scheduler_config.validate());

    scheduler_config.enable_prefix_caching = false;
    EXPECT_NO_THROW(scheduler_config.validate());
    EXPECT_EQ(scheduler_config.cache_interval, 0);
}

TEST(TestScheduler, scheduler_config_custom_cache_interval_requires_linear_attention_model) {
    ov::Core core;
    ov::InferRequest request = core.compile_model(get_dummy_model(core, TEST_NUM_DECODER_LAYERS)).create_infer_request();
    auto get_available_memory = [](const std::string&, size_t) {
        return std::numeric_limits<size_t>::max();
    };

    SchedulerConfig default_config;
    default_config.num_kv_blocks = 64;
    EXPECT_NO_THROW(CacheOrchestrator::create(request, default_config, get_available_memory));

    SchedulerConfig custom_interval_config;
    custom_interval_config.num_kv_blocks = 64;
    custom_interval_config.cache_interval = 64;
    EXPECT_ANY_THROW(CacheOrchestrator::create(request, custom_interval_config, get_available_memory));
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
    for (size_t token = tokens.size(); token < 127; ++token) {
        running_sequence->append_token(token, 0.9f);
    }

    // processed=127, scheduled=1 stores the 128-token checkpoint in the current block.
    seq_group->update_processed_tokens_num(127);
    auto out = scheduler.schedule(requests);

    ASSERT_TRUE(out.m_linear_attention_paging_data.count(seq_id));
    ASSERT_EQ(out.m_linear_attention_paging_data.at(seq_id).block_indices.size(), 2);
    const auto read_idx = out.m_linear_attention_paging_data.at(seq_id).block_indices[0];
    const auto write_idx = out.m_linear_attention_paging_data.at(seq_id).block_indices[1];
    EXPECT_EQ(read_idx, write_idx);
    EXPECT_EQ(out.m_linear_attention_paging_data.at(seq_id).cache_interval, DEFAULT_LINEAR_ATTENTION_CACHE_INTERVAL);

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
    for (size_t token = tokens.size(); token < DEFAULT_LINEAR_ATTENTION_CACHE_INTERVAL; ++token) {
        running_sequence->append_token(token, 0.9f);
    }

    seq_group->update_processed_tokens_num(DEFAULT_LINEAR_ATTENTION_CACHE_INTERVAL);
    auto out = scheduler.schedule(requests);

    ASSERT_TRUE(out.m_linear_attention_paging_data.count(seq_id));
    ASSERT_EQ(out.m_linear_attention_paging_data.at(seq_id).block_indices.size(), 2);
    const auto read_idx = out.m_linear_attention_paging_data.at(seq_id).block_indices[0];
    const auto write_idx = out.m_linear_attention_paging_data.at(seq_id).block_indices[1];
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
    for (size_t token = tokens.size(); token < 127; ++token) {
        running_sequence->append_token(token, 0.9f);
    }

    // processed=126, scheduled=1 does not cross cache_interval=128 boundary.
    seq_group->update_processed_tokens_num(126);
    auto out = scheduler.schedule(requests);

    ASSERT_TRUE(out.m_linear_attention_paging_data.count(seq_id));
    ASSERT_EQ(out.m_linear_attention_paging_data.at(seq_id).block_indices.size(), 2);
    const auto read_idx = out.m_linear_attention_paging_data.at(seq_id).block_indices[0];
    const auto write_idx = out.m_linear_attention_paging_data.at(seq_id).block_indices[1];
    EXPECT_EQ(read_idx, write_idx);
    EXPECT_EQ(out.m_linear_attention_paging_data.at(seq_id).cache_interval, DEFAULT_LINEAR_ATTENTION_CACHE_INTERVAL);

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
    EXPECT_EQ(out1.m_block_tables[idx0][0].size(), 2);
    EXPECT_EQ(out1.m_block_tables[idx1][0].size(), 2);
    EXPECT_FALSE(out1.m_block_tables[idx0][0][0]->is_free());
    EXPECT_EQ(out1.m_block_tables[idx0][0][0]->get_index(), 0);
    EXPECT_FALSE(out1.m_block_tables[idx0][0][1]->is_free());
    EXPECT_EQ(out1.m_block_tables[idx0][0][1]->get_index(), 1);
    EXPECT_FALSE(out1.m_block_tables[idx1][0][0]->is_free());
    EXPECT_EQ(out1.m_block_tables[idx1][0][0]->get_index(), 2);
    EXPECT_FALSE(out1.m_block_tables[idx1][0][1]->is_free());
    EXPECT_EQ(out1.m_block_tables[idx1][0][1]->get_index(), 3);
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
    EXPECT_EQ(out2.m_block_tables[idx0][0].size(), 3);
    EXPECT_FALSE(out2.m_block_tables[idx0][0][0]->is_free());
    EXPECT_EQ(out2.m_block_tables[idx0][0][0]->get_index(), 0);
    EXPECT_FALSE(out2.m_block_tables[idx0][0][1]->is_free());
    EXPECT_EQ(out2.m_block_tables[idx0][0][1]->get_index(), 1);
    EXPECT_FALSE(out2.m_block_tables[idx0][0][2]->is_free());
    EXPECT_EQ(out2.m_block_tables[idx0][0][2]->get_index(), 4);

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
    auto block_table1 = scheduler.get_block_tables(*(*sequence_group1)[0])[0];
    auto block_table2 = scheduler.get_block_tables(*(*sequence_group2)[0])[0];
    EXPECT_EQ(block_table1.size(), 4);
    EXPECT_EQ(block_table1[0]->get_index(), 0);
    EXPECT_EQ(block_table1[1]->get_index(), 1);
    EXPECT_EQ(block_table1[2]->get_index(), 2);
    EXPECT_EQ(block_table1[3]->get_index(), 5);
    EXPECT_EQ(block_table2.size(), 2);
    EXPECT_EQ(block_table2[0]->get_index(), 3);
    EXPECT_EQ(block_table2[1]->get_index(), 4);

    EXPECT_EQ(out2.m_total_num_scheduled_tokens, 1);
    EXPECT_EQ(out2.m_block_tables[idx0][0][0]->get_index(), 0);
    EXPECT_EQ(out2.m_block_tables[idx0][0][1]->get_index(), 1);
    EXPECT_EQ(out2.m_block_tables[idx0][0][2]->get_index(), 2);
    EXPECT_EQ(out2.m_block_tables[idx0][0][3]->get_index(), 5);

    // finish first sequence
    requests[0]->get_running_sequences()[0]->set_status(SequenceStatus::FINISHED);
    scheduler.free_sequence(idx0);
    clear_finished_sequences(requests);
    // KV blocks 0,1,2,5 are free now

    // sequence_group2 should be scheduled
    auto out3 = scheduler.schedule(requests);

    // last token should be recomputed
    EXPECT_EQ(out3.m_total_num_scheduled_tokens, 1);
    EXPECT_EQ(out3.m_block_tables[idx1][0][0]->get_index(), 3);
    EXPECT_EQ(out3.m_block_tables[idx1][0][1]->get_index(), 4);
    EXPECT_EQ(out3.m_block_tables[idx1][0][2]->get_index(), 0);

    block_table2 = scheduler.get_block_tables(*(*sequence_group2)[0])[0];
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
        EXPECT_EQ(scheduler.get_block_tables(*seqs[0])[0].size(), 2);
        EXPECT_EQ(scheduler.get_block_tables(*seqs[1])[0].size(), 2);
        EXPECT_EQ(scheduler.get_block_tables(*seqs[2])[0].size(), 2);
        EXPECT_EQ(scheduler.get_block_tables(*seqs[3])[0].size(), 2);
        EXPECT_EQ(scheduler.get_block_tables(*seqs[4])[0].size(), 2);

        // append another 20 tokens to greedy group, this should result in usage of all free blocks and
        // another partial preemption of beam search group
        for (size_t i = 0; i < 20; i++) {
            out = scheduler.schedule(new_requests);
            sequence_group_greedy->get_sequences()[0]->append_token(token, 0.5);
            sequence_group_greedy->finish_iteration();
        }

        EXPECT_EQ(sequence_group->get_num_processed_tokens(), 4);
        seqs = sequence_group->get_sequences();
        EXPECT_EQ(scheduler.get_block_tables(*seqs[0])[0].size(), 1);
        EXPECT_EQ(scheduler.get_block_tables(*seqs[1])[0].size(), 1);
        EXPECT_EQ(scheduler.get_block_tables(*seqs[2])[0].size(), 1);
        EXPECT_EQ(scheduler.get_block_tables(*seqs[3])[0].size(), 1);
        EXPECT_EQ(scheduler.get_block_tables(*seqs[4])[0].size(), 1);

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
        auto block_tables_for_all_layers = scheduler.get_block_tables(*(*sequence_group1)[0]);
        auto block_table1 = block_tables_for_all_layers[0];
        EXPECT_EQ(block_table1.size(), 4);
        EXPECT_EQ(block_table1[0]->get_index(), 0);
        EXPECT_EQ(block_table1[1]->get_index(), 1);
        EXPECT_EQ(block_table1[2]->get_index(), 2);
        EXPECT_EQ(block_table1[3]->get_index(), 5);
        EXPECT_EQ(out2.m_block_tables[idx0][0].size(), 4);
        EXPECT_EQ(out2.m_block_tables[idx0][0][0]->get_index(), 0);
        EXPECT_EQ(out2.m_block_tables[idx0][0][1]->get_index(), 1);
        EXPECT_EQ(out2.m_block_tables[idx0][0][2]->get_index(), 2);
        EXPECT_EQ(out2.m_block_tables[idx0][0][3]->get_index(), 5);

        std::vector<uint64_t> ref_ids = {0};
        EXPECT_EQ(out2.m_scheduled_sequence_groups_ids, ref_ids);
        EXPECT_EQ(out2.m_total_num_scheduled_tokens, 1);

        if (scheduler_config.dynamic_split_fuse) {
            // for dynamic_split_fuse sequence_group2 is preemted partially, part of prompt is left
            EXPECT_TRUE(scheduler.has_block_table(idx1));
            auto block_table2 = scheduler.get_block_tables(*(*sequence_group2)[0])[0];
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

        EXPECT_EQ(out3.m_block_tables[idx1][0][0]->get_index(), 3);
        EXPECT_EQ(out3.m_block_tables[idx1][0][1]->get_index(), 4);
        EXPECT_EQ(out3.m_block_tables[idx1][0][2]->get_index(), 0);

        auto block_table2 = scheduler.get_block_tables(*(*sequence_group2)[0])[0];
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
    auto block_table1 = scheduler.get_block_tables(*(*sequence_group1)[0]);
    ASSERT_EQ(block_table1[0].size(), 4);
    ASSERT_EQ(block_table1[0][0]->get_index(), 0);
    ASSERT_EQ(block_table1[0][1]->get_index(), 1);
    ASSERT_EQ(block_table1[0][2]->get_index(), 2);
    ASSERT_EQ(block_table1[0][3]->get_index(), 3);
    ASSERT_EQ(out2.m_block_tables[idx0][0].size(), 4);
    ASSERT_EQ(out2.m_block_tables[idx0][0][0]->get_index(), 0);
    ASSERT_EQ(out2.m_block_tables[idx0][0][1]->get_index(), 1);
    ASSERT_EQ(out2.m_block_tables[idx0][0][2]->get_index(), 2);
    ASSERT_EQ(out2.m_block_tables[idx0][0][3]->get_index(), 3);

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

    ASSERT_EQ(out3.m_block_tables[idx1][0][0]->get_index(), 4);
    ASSERT_EQ(out3.m_block_tables[idx1][0][1]->get_index(), 5);
    ASSERT_EQ(out3.m_block_tables[idx1][0][2]->get_index(), 0);

    auto block_table2 = scheduler.get_block_tables(*(*sequence_group2)[0]);
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
    auto block_table1 = scheduler.get_block_tables(*(*sequence_group1)[0]);
    ASSERT_EQ(block_table1[0].size(), 4);
    ASSERT_EQ(block_table1[0][0]->get_index(), 0);
    ASSERT_EQ(block_table1[0][1]->get_index(), 1);
    ASSERT_EQ(block_table1[0][2]->get_index(), 2);
    ASSERT_EQ(block_table1[0][3]->get_index(), 3);
    ASSERT_EQ(out2.m_block_tables[idx0][0].size(), 4);
    ASSERT_EQ(out2.m_block_tables[idx0][0][0]->get_index(), 0);
    ASSERT_EQ(out2.m_block_tables[idx0][0][1]->get_index(), 1);
    ASSERT_EQ(out2.m_block_tables[idx0][0][2]->get_index(), 2);
    ASSERT_EQ(out2.m_block_tables[idx0][0][3]->get_index(), 3);

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

    ASSERT_EQ(out3.m_block_tables[idx1][0][0]->get_index(), 4);
    ASSERT_EQ(out3.m_block_tables[idx1][0][1]->get_index(), 5);
    ASSERT_EQ(out3.m_block_tables[idx1][0][2]->get_index(), 0);

    auto block_table2 = scheduler.get_block_tables(*(*sequence_group2)[0]);
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
    auto block_table1 = _get_indices(scheduler.get_block_tables(*(*sequence_group1)[0])[0]);
    auto block_table2 = _get_indices(scheduler.get_block_tables(*(*sequence_group2)[0])[0]);

    const std::vector<size_t> ref_block_table1{0, 1, 2};
    EXPECT_EQ(block_table1, ref_block_table1);

    const std::vector<size_t> ref_block_table2{3, 4, 5};
    EXPECT_EQ(block_table2, ref_block_table2);

    // Next generation in 1-st sequence group should lead to preemption of 2-nd, but tokens from it were evicted already
    // Should ensure that the 2-nd sequence can only be preempted completely
    out = _schedule_one_mock_generation_token_for_each_sequence_group(scheduler, requests);

    block_table1 = _get_indices(scheduler.get_block_tables(*(*sequence_group1)[0])[0]);

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
    block_table2 = _get_indices(scheduler.get_block_tables(*(*sequence_group2)[0])[0]);
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
