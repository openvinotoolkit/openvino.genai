// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <algorithm>
#include <numeric>
#include <set>
#include "openvino/runtime/core.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"
#include "sequence_group.hpp"
#include "continuous_batching/scheduler.hpp"
#include "continuous_batching/pipeline_impl.hpp"
#include "continuous_batching/cache/cache_orchestrator.hpp"
#include "helper.hpp"
#include "scheduler_test_utils.hpp"
#include "utils.hpp"

using namespace ov::genai;

void clear_finished_sequences(std::vector<SequenceGroup::Ptr>& requests);

class CBPublicationTest : public testing::Test, public ContinuousBatchingPipeline {
protected:
    class PipelineTestInstance : public ContinuousBatchingPipeline::ContinuousBatchingImpl {
    public:
        void publish_completed_cache_blocks(const std::shared_ptr<Scheduler>& scheduler,
                                            const std::vector<SequenceGroup::Ptr>& requests,
                                            const Scheduler::Output& scheduler_output) {
            m_scheduler = scheduler;
            m_requests = requests;
            _publish_completed_cache_blocks(scheduler_output);
        }

        void commit_linear_attention_checkpoint_transactions(const std::shared_ptr<Scheduler>& scheduler,
                                                             const std::vector<SequenceGroup::Ptr>& requests,
                                                             Scheduler::Output& scheduler_output,
                                                             const SamplerOutput& sampler_output) {
            m_scheduler = scheduler;
            m_requests = requests;
            _commit_linear_attention_checkpoint_transactions(scheduler_output, sampler_output);
        }
    };
};

TEST_F(CBPublicationTest, kv_prefix_caching_republishes_accepted_completed_cow_through_pipeline_hook) {
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 16;
    scheduler_config.num_kv_blocks = 12;
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.dynamic_split_fuse = false;
    scheduler_config.max_num_seqs = 4;

    auto orchestrator = init_cache_orchestrator(scheduler_config);
    auto scheduler = std::make_shared<Scheduler>(orchestrator, scheduler_config);

    std::vector<uint64_t> source_tokens = {0, 1, 2, 3, 4, 5};
    SequenceGroup::Ptr source_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {source_tokens.size()}, source_tokens.data()),
        utils::get_greedy_config());
    const uint64_t source_seq_id = source_group->get_running_sequences()[0]->get_id();
    std::vector<SequenceGroup::Ptr> source_requests = {source_group};
    std::ignore = scheduler->schedule(source_requests);
    source_group->finish_iteration();
    scheduler->free_sequence(source_seq_id);

    std::vector<uint64_t> completed_tokens = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<uint64_t> divergent_tokens = {0, 1, 2, 3, 4, 5, 20, 21};
    SequenceGroup::Ptr completed_group = std::make_shared<SequenceGroup>(
        1,
        ov::Tensor(ov::element::i64, {completed_tokens.size()}, completed_tokens.data()),
        utils::get_greedy_config());
    SequenceGroup::Ptr divergent_group = std::make_shared<SequenceGroup>(
        2,
        ov::Tensor(ov::element::i64, {divergent_tokens.size()}, divergent_tokens.data()),
        utils::get_greedy_config());
    scheduler->restore_cached_blocks(completed_group);
    scheduler->restore_cached_blocks(divergent_group);
    ASSERT_EQ(completed_group->get_num_processed_tokens(), source_tokens.size());
    ASSERT_EQ(divergent_group->get_num_processed_tokens(), source_tokens.size());

    scheduler->set_expected_num_scheduled_tokens(completed_group->get_request_id(), 2);
    std::vector<SequenceGroup::Ptr> requests = {completed_group};
    const Scheduler::Output output = scheduler->schedule(requests);
    ASSERT_EQ(output.m_total_num_scheduled_tokens, 2);
    ASSERT_EQ(output.get_kv_paged_attention_data(completed_group->get_running_sequences()[0]->get_id())
                  .num_processed_tokens_before,
              source_tokens.size());
    completed_group->finish_iteration();
    PipelineTestInstance pipeline;
    pipeline.publish_completed_cache_blocks(scheduler, requests, output);

    scheduler->free_sequence(completed_group->get_running_sequences()[0]->get_id());
    scheduler->free_sequence(divergent_group->get_running_sequences()[0]->get_id());

    SequenceGroup::Ptr restored_group = std::make_shared<SequenceGroup>(
        3,
        ov::Tensor(ov::element::i64, {completed_tokens.size()}, completed_tokens.data()),
        utils::get_greedy_config());
    const BlockManager::PrefixRestorePlan restore_plan =
        orchestrator->get_block_manager(CacheType::KV_CACHE).get_prefix_restore_plan(restored_group);
    EXPECT_EQ(restore_plan.cache_token_position, completed_tokens.size());
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

TEST(TestScheduler, hybrid_non_prefix_linear_attention_fork_cows_latest_rows_before_publication) {
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
    scheduler_config.num_linear_attention_blocks = 4;

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                       TEST_BLOCK_SIZE,
                                                       /*kv_num_layers=*/1,
                                                       /*la_num_layers=*/1);
    Scheduler scheduler(orchestrator, scheduler_config);
    std::vector<SequenceGroup::Ptr> requests;
    auto sequence_group = make_prompt_processed_sequence_group(scheduler, requests, {0, 1, 2, 3});
    auto parent = sequence_group->get_running_sequences().at(0);
    auto child = sequence_group->fork_sequence(parent);
    scheduler.fork_sequence(parent->get_id(), child->get_id());

    const size_t shared_latest_row = orchestrator->get_linear_attention_latest_row(parent->get_id());
    ASSERT_EQ(orchestrator->get_linear_attention_latest_row(child->get_id()), shared_latest_row);
    ASSERT_TRUE(orchestrator->is_linear_attention_latest_row_shared(parent->get_id()));
    ASSERT_TRUE(orchestrator->is_linear_attention_latest_row_shared(child->get_id()));
    const size_t free_rows_before =
        orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE).num_free_blocks();

    const Scheduler::Output output = scheduler.schedule(requests);

    const size_t parent_latest_row = orchestrator->get_linear_attention_latest_row(parent->get_id());
    const size_t child_latest_row = orchestrator->get_linear_attention_latest_row(child->get_id());
    EXPECT_NE(parent_latest_row, child_latest_row);
    EXPECT_FALSE(orchestrator->is_linear_attention_latest_row_shared(parent->get_id()));
    EXPECT_FALSE(orchestrator->is_linear_attention_latest_row_shared(child->get_id()));
    ASSERT_TRUE(output.has_linear_attention_paging_data(parent->get_id()));
    ASSERT_TRUE(output.has_linear_attention_paging_data(child->get_id()));
    EXPECT_EQ(output.get_linear_attention_paging_data(parent->get_id()).block_indices,
              (std::vector<int32_t>{static_cast<int32_t>(parent_latest_row),
                                    static_cast<int32_t>(parent_latest_row)}));
    EXPECT_EQ(output.get_linear_attention_paging_data(child->get_id()).block_indices,
              (std::vector<int32_t>{static_cast<int32_t>(child_latest_row),
                                    static_cast<int32_t>(child_latest_row)}));
    EXPECT_EQ(orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE).num_free_blocks(),
              free_rows_before - 1);

    scheduler.free_sequence(child->get_id());
    scheduler.free_sequence(parent->get_id());
}

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

    const auto borrowed = la_block_manager.reserve_temporary_blocks(seq_id, 2);
    ASSERT_EQ(borrowed.size(), 2u);
    ASSERT_TRUE(la_block_manager.has_temporary_blocks(seq_id));

    EXPECT_THROW(std::ignore = scheduler.schedule(requests), ov::Exception);
    EXPECT_TRUE(la_block_manager.has_temporary_blocks(seq_id))
        << "plain paging silently released scratch instead of reporting the violated invariant";

    scheduler.release_linear_attention_checkpoints(seq_id);
    EXPECT_FALSE(la_block_manager.has_temporary_blocks(seq_id));
    scheduler.free_sequence(seq_id);
}

// The paging window contains one latest row and N+1 distinct borrowed rows.
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

    const size_t latest_row = orchestrator->get_linear_attention_latest_row(seq_id);
    seq_group->set_num_validated_tokens(N);

    auto out = scheduler.schedule(requests);
    ASSERT_TRUE(out.has_linear_attention_paging_data(seq_id));
    const auto& paging_data = out.get_linear_attention_paging_data(seq_id);

    ASSERT_EQ(paging_data.block_indices.size(), N + 2);
    EXPECT_EQ(static_cast<size_t>(paging_data.block_indices[0]), latest_row);
    std::set<int32_t> seen = {paging_data.block_indices[0]};
    for (size_t i = 1; i < paging_data.block_indices.size(); ++i) {
        EXPECT_NE(paging_data.block_indices[i], paging_data.block_indices[0])
            << "borrowed row " << i << " aliases the latest row";
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

TEST_F(CBPublicationTest, real_sampler_acceptance_promotes_linear_attention_checkpoint_before_terminal_free) {
    struct AcceptanceCase {
        std::vector<int64_t> candidates;
        size_t expected_depth;
        bool stop_on_second_token;
    };
    const std::vector<AcceptanceCase> cases{
        {{4, 2, 3}, 1, false},
        {{1, 4, 3}, 2, false},
        {{1, 2, 3}, 4, false},
        {{1, 2, 3}, 2, true},
    };
    const std::vector<float> logits{
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
        0, 0, 0, 0, 1.f,
    };

    for (const AcceptanceCase& acceptance_case : cases) {
        SCOPED_TRACE(acceptance_case.expected_depth);
        SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
        auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config,
                                                           TEST_BLOCK_SIZE,
                                                           /*kv_num_layers=*/1,
                                                           /*la_num_layers=*/1);
        auto scheduler = std::make_shared<Scheduler>(orchestrator, scheduler_config);
        scheduler->ensure_linear_attention_pool_blocks(5);

        GenerationConfig generation_config = utils::get_greedy_config();
        if (acceptance_case.stop_on_second_token) {
            generation_config.stop_token_ids = {2};
        }
        const std::vector<int64_t> prompt{0, 1, 2, 3};
        auto sequence_group = std::make_shared<SequenceGroup>(0, prompt, generation_config);
        std::vector<SequenceGroup::Ptr> requests{sequence_group};
        Scheduler::Output prefill_output = scheduler->schedule(requests);
        sequence_group->finish_iteration();
        PipelineTestInstance pipeline;
        pipeline.publish_completed_cache_blocks(scheduler, requests, prefill_output);
        Sequence::Ptr sequence = sequence_group->get_running_sequences().front();
        sequence->append_token(0, 1.f);
        sequence_group->update_processed_tokens_num(prompt.size());
        for (const int64_t token_id : acceptance_case.candidates) {
            sequence->append_token(token_id, 1.f);
        }
        sequence_group->set_num_validated_tokens(acceptance_case.candidates.size());

        Scheduler::Output scheduler_output = scheduler->schedule(requests);
        const uint64_t sequence_id = sequence->get_id();
        const auto& paging_data = scheduler_output.get_linear_attention_paging_data(sequence_id);
        ASSERT_TRUE(paging_data.is_speculative);
        ov::Tensor validation_logits(ov::element::f32, ov::Shape{4, 1, 5});
        std::copy(logits.begin(), logits.end(), validation_logits.data<float>());
        Sampler sampler;
        const SamplerOutput sampler_output = sampler.sample(requests, validation_logits, true, false, true);

        ASSERT_EQ(sampler_output.acceptance_by_sequence.count(sequence_id), 1u);
        ASSERT_EQ(sampler_output.acceptance_by_sequence.at(sequence_id).accepted_depth,
                  acceptance_case.expected_depth);
        EXPECT_EQ(sequence_group->get_num_processed_tokens(), prompt.size());
        auto& kv_block_manager = orchestrator->get_block_manager(CacheType::KV_CACHE);
        const size_t allocated_kv_blocks = kv_block_manager.get_block_tables(sequence_id).front().size();
        const size_t free_kv_blocks_before = kv_block_manager.num_free_blocks();
        const size_t promoted_row = static_cast<size_t>(paging_data.block_indices.at(acceptance_case.expected_depth));
        pipeline.commit_linear_attention_checkpoint_transactions(scheduler, requests, scheduler_output, sampler_output);

        auto& la_block_manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
        const size_t accepted_endpoint = prompt.size() + acceptance_case.expected_depth;
        const size_t expected_kv_blocks = 1 + (accepted_endpoint - 1) / TEST_BLOCK_SIZE;
        EXPECT_EQ(orchestrator->get_linear_attention_latest_row(sequence_id), promoted_row);
        EXPECT_EQ(la_block_manager.get_linear_attention_live_state(sequence_id).endpoint,
                  sequence_group->get_num_processed_tokens());
        EXPECT_EQ(sequence_group->get_num_processed_tokens(), accepted_endpoint);
        EXPECT_FALSE(la_block_manager.has_temporary_blocks(sequence_id));
        const auto& kv_block_tables = orchestrator->get_kv_block_tables(sequence_id);
        ASSERT_FALSE(kv_block_tables.empty());
        EXPECT_EQ(kv_block_tables.front().size(), expected_kv_blocks);
        EXPECT_EQ(kv_block_manager.num_free_blocks(),
              free_kv_blocks_before + allocated_kv_blocks - expected_kv_blocks);
        EXPECT_EQ(sequence->has_finished(), acceptance_case.stop_on_second_token);
        if (acceptance_case.stop_on_second_token) {
            EXPECT_EQ(sequence->get_finish_reason(), GenerationFinishReason::STOP);
        }
        scheduler->free_sequence(sequence_id);
    }
}

TEST_F(CBPublicationTest, invalid_second_kv_tail_target_leaves_all_tables_and_free_rows_unchanged) {
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    auto scheduler = std::make_shared<Scheduler>(orchestrator, scheduler_config);
    auto first_group = std::make_shared<SequenceGroup>(
        0, std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7}, utils::get_greedy_config());
    auto second_group = std::make_shared<SequenceGroup>(
        1, std::vector<int64_t>{8, 9, 10, 11, 12, 13, 14, 15}, utils::get_greedy_config());
    std::vector<SequenceGroup::Ptr> requests{first_group, second_group};
    std::ignore = scheduler->schedule(requests);

    const uint64_t first_id = first_group->get_sequences().front()->get_id();
    const uint64_t second_id = second_group->get_sequences().front()->get_id();
    auto& kv_block_manager = orchestrator->get_block_manager(CacheType::KV_CACHE);
    const auto first_table_before = kv_block_manager.get_block_tables(first_id);
    const auto second_table_before = kv_block_manager.get_block_tables(second_id);
    const size_t free_blocks_before = kv_block_manager.num_free_blocks();
    ASSERT_EQ(first_table_before.front().size(), 2u);
    ASSERT_EQ(second_table_before.front().size(), 2u);

    const std::vector<BlockManager::TailReleaseTarget> targets{
        {first_id, TEST_BLOCK_SIZE},
        {second_id, 2 * TEST_BLOCK_SIZE + 1},
    };
    EXPECT_THROW(std::ignore = scheduler->prepare_kv_tail_releases(targets), ov::Exception);

    EXPECT_EQ(kv_block_manager.get_block_tables(first_id), first_table_before);
    EXPECT_EQ(kv_block_manager.get_block_tables(second_id), second_table_before);
    EXPECT_EQ(kv_block_manager.num_free_blocks(), free_blocks_before);
    scheduler->free_sequence(first_id);
    scheduler->free_sequence(second_id);
}

TEST_F(CBPublicationTest, missing_second_acceptance_does_not_promote_first_linear_attention_checkpoint) {
    constexpr size_t accepted_depth = 1;
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    auto scheduler = std::make_shared<Scheduler>(orchestrator, scheduler_config);
    scheduler->ensure_linear_attention_pool_blocks(6);
    auto first_group = std::make_shared<SequenceGroup>(
        0, std::vector<int64_t>{0, 1, 2, 3}, utils::get_greedy_config());
    auto second_group = std::make_shared<SequenceGroup>(
        1, std::vector<int64_t>{4, 5, 6, 7}, utils::get_greedy_config());
    std::vector<SequenceGroup::Ptr> requests{first_group, second_group};
    std::ignore = scheduler->schedule(requests);
    first_group->finish_iteration();
    second_group->finish_iteration();
    for (const auto& sequence_group : requests) {
        sequence_group->get_running_sequences().front()->append_token(42, 0.f);
        sequence_group->update_processed_tokens_num(4);
        sequence_group->set_num_validated_tokens(accepted_depth);
    }

    Scheduler::Output output = scheduler->schedule(requests);
    const uint64_t first_id = first_group->get_running_sequences().front()->get_id();
    const uint64_t second_id = second_group->get_running_sequences().front()->get_id();
    auto& block_manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
    const auto first_state_before = block_manager.get_linear_attention_live_state(first_id);
    const auto second_state_before = block_manager.get_linear_attention_live_state(second_id);
    const size_t free_rows_before = block_manager.num_free_blocks();
    ASSERT_TRUE(block_manager.has_temporary_blocks(first_id));
    ASSERT_TRUE(block_manager.has_temporary_blocks(second_id));

    SamplerOutput sampler_output;
    sampler_output.acceptance_by_sequence.emplace(
        first_id, SamplerOutput::AcceptanceResult{accepted_depth, 4 + accepted_depth});
    PipelineTestInstance pipeline;
    EXPECT_THROW(
        pipeline.commit_linear_attention_checkpoint_transactions(scheduler, requests, output, sampler_output),
        ov::Exception);

    const auto& first_state_after = block_manager.get_linear_attention_live_state(first_id);
    const auto& second_state_after = block_manager.get_linear_attention_live_state(second_id);
    EXPECT_EQ(first_state_after.endpoint, first_state_before.endpoint);
    EXPECT_EQ(first_state_after.generation, first_state_before.generation);
    EXPECT_EQ(first_state_after.rows, first_state_before.rows);
    EXPECT_EQ(second_state_after.endpoint, second_state_before.endpoint);
    EXPECT_EQ(second_state_after.generation, second_state_before.generation);
    EXPECT_EQ(second_state_after.rows, second_state_before.rows);
    EXPECT_TRUE(block_manager.has_temporary_blocks(first_id));
    EXPECT_TRUE(block_manager.has_temporary_blocks(second_id));
    EXPECT_EQ(block_manager.num_free_blocks(), free_rows_before);

    const auto first_promoted_row = output.get_linear_attention_paging_data(first_id).block_indices.at(1);
    const auto second_promoted_row = output.get_linear_attention_paging_data(second_id).block_indices.at(2);
    sampler_output.acceptance_by_sequence.emplace(second_id, SamplerOutput::AcceptanceResult{2, 6});
    pipeline.commit_linear_attention_checkpoint_transactions(scheduler, requests, output, sampler_output);
    EXPECT_EQ(block_manager.get_linear_attention_live_state(first_id).endpoint, first_state_before.endpoint + 1);
    EXPECT_EQ(block_manager.get_linear_attention_live_state(second_id).endpoint, second_state_before.endpoint + 2);
    EXPECT_EQ(block_manager.get_linear_attention_live_state(first_id).generation, first_state_before.generation + 1);
    EXPECT_EQ(block_manager.get_linear_attention_live_state(second_id).generation, second_state_before.generation + 1);
    EXPECT_EQ(orchestrator->get_linear_attention_latest_row(first_id), first_promoted_row);
    EXPECT_EQ(orchestrator->get_linear_attention_latest_row(second_id), second_promoted_row);
    EXPECT_TRUE(output.m_linear_attention_scratch_leases.empty());
    EXPECT_EQ(block_manager.num_free_blocks(), free_rows_before + 4);
    EXPECT_FALSE(block_manager.has_temporary_blocks(first_id));
    EXPECT_FALSE(block_manager.has_temporary_blocks(second_id));
    scheduler->free_sequence(first_id);
    scheduler->free_sequence(second_id);
}

TEST_F(CBPublicationTest, invalid_second_slot_does_not_promote_first_linear_attention_checkpoint) {
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    auto scheduler = std::make_shared<Scheduler>(orchestrator, scheduler_config);
    scheduler->ensure_linear_attention_pool_blocks(6);
    auto first_group = std::make_shared<SequenceGroup>(
        0, std::vector<int64_t>{0, 1, 2, 3}, utils::get_greedy_config());
    auto second_group = std::make_shared<SequenceGroup>(
        1, std::vector<int64_t>{4, 5, 6, 7}, utils::get_greedy_config());
    std::vector<SequenceGroup::Ptr> requests{first_group, second_group};
    std::ignore = scheduler->schedule(requests);
    first_group->finish_iteration();
    second_group->finish_iteration();
    for (const auto& sequence_group : requests) {
        sequence_group->get_running_sequences().front()->append_token(42, 0.f);
        sequence_group->update_processed_tokens_num(4);
        sequence_group->set_num_validated_tokens(1);
    }

    Scheduler::Output output = scheduler->schedule(requests);
    const uint64_t first_id = first_group->get_running_sequences().front()->get_id();
    const uint64_t second_id = second_group->get_running_sequences().front()->get_id();
    auto& block_manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
    const auto first_state_before = block_manager.get_linear_attention_live_state(first_id);
    const auto second_state_before = block_manager.get_linear_attention_live_state(second_id);
    const size_t free_rows_before = block_manager.num_free_blocks();
    SamplerOutput sampler_output;
    sampler_output.acceptance_by_sequence.emplace(first_id, SamplerOutput::AcceptanceResult{1, 5});
    sampler_output.acceptance_by_sequence.emplace(second_id, SamplerOutput::AcceptanceResult{3, 7});

    PipelineTestInstance pipeline;
    EXPECT_THROW(
        pipeline.commit_linear_attention_checkpoint_transactions(scheduler, requests, output, sampler_output),
        ov::Exception);

    const auto& first_state_after = block_manager.get_linear_attention_live_state(first_id);
    const auto& second_state_after = block_manager.get_linear_attention_live_state(second_id);
    EXPECT_EQ(first_state_after.endpoint, first_state_before.endpoint);
    EXPECT_EQ(first_state_after.generation, first_state_before.generation);
    EXPECT_EQ(first_state_after.rows, first_state_before.rows);
    EXPECT_EQ(second_state_after.endpoint, second_state_before.endpoint);
    EXPECT_EQ(second_state_after.generation, second_state_before.generation);
    EXPECT_EQ(second_state_after.rows, second_state_before.rows);
    EXPECT_TRUE(block_manager.has_temporary_blocks(first_id));
    EXPECT_TRUE(block_manager.has_temporary_blocks(second_id));
    EXPECT_EQ(block_manager.num_free_blocks(), free_rows_before);

    output.m_linear_attention_scratch_leases.clear();
    EXPECT_FALSE(block_manager.has_temporary_blocks(first_id));
    EXPECT_FALSE(block_manager.has_temporary_blocks(second_id));
    scheduler->free_sequence(first_id);
    scheduler->free_sequence(second_id);
}

TEST_F(CBPublicationTest, abandoned_moved_prepared_promotion_preserves_scratch_until_lease_cleanup) {
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config, TEST_BLOCK_SIZE, 1, 2);
    auto scheduler = std::make_shared<Scheduler>(orchestrator, scheduler_config);
    scheduler->ensure_linear_attention_pool_blocks(3);
    auto sequence_group = std::make_shared<SequenceGroup>(
        0, std::vector<int64_t>{0, 1, 2, 3}, utils::get_greedy_config());
    std::vector<SequenceGroup::Ptr> requests{sequence_group};
    std::ignore = scheduler->schedule(requests);
    sequence_group->finish_iteration();
    sequence_group->get_running_sequences().front()->append_token(42, 0.f);
    sequence_group->update_processed_tokens_num(4);
    sequence_group->set_num_validated_tokens(1);
    Scheduler::Output output = scheduler->schedule(requests);
    const uint64_t sequence_id = sequence_group->get_running_sequences().front()->get_id();
    auto& block_manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
    const auto state_before = block_manager.get_linear_attention_live_state(sequence_id);
    const size_t free_rows_before = block_manager.num_free_blocks();
    auto* lease = output.m_linear_attention_scratch_leases.at(sequence_id).get();
    const std::vector<CacheOrchestrator::LinearAttentionScratchLease*> leases{lease};
    const std::vector<BlockManager::TemporaryPromotionRequest> promotion_requests{
        lease->promotion_request(1)};
    auto mismatched_requests = promotion_requests;
    ++mismatched_requests.front().expected_generation;
    EXPECT_THROW(CacheOrchestrator::LinearAttentionScratchLease::prepare_promotions(leases, mismatched_requests),
                 ov::Exception);
    {
        auto prepared = CacheOrchestrator::LinearAttentionScratchLease::prepare_promotions(
            leases, promotion_requests);
        auto moved = std::move(prepared);
        EXPECT_THROW(prepared.apply(), ov::Exception);
    }

    const auto& state_after = block_manager.get_linear_attention_live_state(sequence_id);
    EXPECT_EQ(state_after.endpoint, state_before.endpoint);
    EXPECT_EQ(state_after.generation, state_before.generation);
    EXPECT_EQ(state_after.rows, state_before.rows);
    EXPECT_TRUE(block_manager.has_temporary_blocks(sequence_id));
    EXPECT_EQ(block_manager.num_free_blocks(), free_rows_before);
    {
        auto prepared = CacheOrchestrator::LinearAttentionScratchLease::prepare_promotions(
            leases, promotion_requests);
        const auto& promoted_indices = prepared.apply();
        ASSERT_EQ(promoted_indices.size(), 1u);
        const auto& committed_state = block_manager.get_linear_attention_live_state(sequence_id);
        EXPECT_EQ(committed_state.endpoint, state_before.endpoint + 1);
        EXPECT_EQ(committed_state.generation, state_before.generation + 1);
        ASSERT_EQ(committed_state.rows.size(), 1u);
        for (const auto& row : committed_state.rows) {
            EXPECT_EQ(row->get_index(), promoted_indices.front());
            EXPECT_EQ(row->get_references_count(), 2u);
        }
        EXPECT_THROW(prepared.apply(), ov::Exception);
        lease->mark_committed();
    }
    output.m_linear_attention_scratch_leases.clear();
    EXPECT_FALSE(block_manager.has_temporary_blocks(sequence_id));
    EXPECT_EQ(block_manager.num_free_blocks(), free_rows_before + 2);
    scheduler->free_sequence(sequence_id);
}

TEST_F(CBPublicationTest, non_prefix_prefill_then_decode_advances_linear_attention_live_endpoint) {
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    auto scheduler = std::make_shared<Scheduler>(orchestrator, scheduler_config);
    std::vector<SequenceGroup::Ptr> requests;
    auto sequence_group = std::make_shared<SequenceGroup>(
        0, std::vector<int64_t>{0, 1, 2, 3}, utils::get_greedy_config());
    requests.push_back(sequence_group);
    const uint64_t sequence_id = sequence_group->get_running_sequences().front()->get_id();
    PipelineTestInstance pipeline;

    Scheduler::Output prefill_output = scheduler->schedule(requests);
    sequence_group->finish_iteration();
    pipeline.publish_completed_cache_blocks(scheduler, requests, prefill_output);
    EXPECT_EQ(orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE)
                  .get_linear_attention_live_state(sequence_id).endpoint,
              4u);

    sequence_group->get_running_sequences().front()->append_token(42, 0.f);
    Scheduler::Output decode_output = scheduler->schedule(requests);
    sequence_group->finish_iteration();
    pipeline.publish_completed_cache_blocks(scheduler, requests, decode_output);
    EXPECT_EQ(orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE)
                  .get_linear_attention_live_state(sequence_id).endpoint,
              5u);
    scheduler->free_sequence(sequence_id);
}

TEST_F(CBPublicationTest, prefix_prefill_append_updates_live_endpoint_and_row) {
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
    scheduler_config.enable_prefix_caching = true;
    scheduler_config.cache_interval_multiplier = 1;
    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    auto scheduler = std::make_shared<Scheduler>(orchestrator, scheduler_config);
    std::vector<SequenceGroup::Ptr> requests;
    auto sequence_group = std::make_shared<SequenceGroup>(
        0, std::vector<int64_t>{0, 1, 2, 3, 4, 5}, utils::get_greedy_config());
    requests.push_back(sequence_group);
    const uint64_t sequence_id = sequence_group->get_running_sequences().front()->get_id();

    Scheduler::Output output = scheduler->schedule(requests);
    sequence_group->finish_iteration();
    PipelineTestInstance pipeline;
    pipeline.publish_completed_cache_blocks(scheduler, requests, output);

    const auto& block_manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
    const auto& live_state = block_manager.get_linear_attention_live_state(sequence_id);
    EXPECT_EQ(live_state.endpoint, 6u);
    ASSERT_EQ(live_state.rows.size(), 1u);
    EXPECT_EQ(live_state.rows.front()->get_index(), block_manager.get_block_tables(sequence_id).at(0).at(1)->get_index());
    scheduler->free_sequence(sequence_id);
}

TEST_F(CBPublicationTest, multinomial_speculative_commit_preserves_processed_depth_fallback) {
    constexpr size_t accepted_depth = 2;
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    auto scheduler = std::make_shared<Scheduler>(orchestrator, scheduler_config);
    scheduler->ensure_linear_attention_pool_blocks(1 + (1 + accepted_depth));
    auto sampling_config = utils::get_multinomial_config();
    auto sequence_group = std::make_shared<SequenceGroup>(
        0, std::vector<int64_t>{0, 1, 2, 3}, sampling_config);
    std::vector<SequenceGroup::Ptr> requests{sequence_group};
    std::ignore = scheduler->schedule(requests);
    sequence_group->finish_iteration();
    sequence_group->get_running_sequences().front()->append_token(42, 0.f);
    sequence_group->update_processed_tokens_num(4);
    sequence_group->set_num_validated_tokens(accepted_depth);
    const uint64_t sequence_id = sequence_group->get_running_sequences().front()->get_id();
    Scheduler::Output output = scheduler->schedule(requests);
    const size_t promoted_row = static_cast<size_t>(
        output.get_linear_attention_paging_data(sequence_id).block_indices.at(accepted_depth));
    sequence_group->update_processed_tokens_num(4 + accepted_depth);

    PipelineTestInstance pipeline;
    pipeline.commit_linear_attention_checkpoint_transactions(scheduler, requests, output, SamplerOutput{});
    EXPECT_EQ(orchestrator->get_linear_attention_latest_row(sequence_id), promoted_row);
    scheduler->free_sequence(sequence_id);
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
    size_t previous_latest_row = orchestrator->get_linear_attention_latest_row(seq_id);

    for (size_t step = 0; step < advances.size(); ++step) {
        seq_group->set_num_validated_tokens(N);

        auto out = scheduler.schedule(requests);
        ASSERT_TRUE(out.has_linear_attention_paging_data(seq_id));
        const auto& pd = out.get_linear_attention_paging_data(seq_id);
        ASSERT_TRUE(pd.is_speculative);
        ASSERT_EQ(pd.block_indices.size(), N + 2);

        EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id).size(), 1u)
            << "owned LA set changed size at step " << step;
        EXPECT_EQ(static_cast<size_t>(pd.block_indices[0]), previous_latest_row);
        EXPECT_EQ(la_block_manager.get_num_blocks_in_use(), 1 + (1 + N))
            << "unexpected pool occupancy while verifying at step " << step;
        std::set<int32_t> seen = {pd.block_indices[0]};
        for (size_t i = 1; i < pd.block_indices.size(); ++i) {
            EXPECT_TRUE(seen.insert(pd.block_indices[i]).second) << "duplicate row at step " << step;
        }

        const size_t advance = advances[step];
        const int32_t chosen = pd.block_indices[advance];
        const auto& live_state = la_block_manager.get_linear_attention_live_state(seq_id);
        la_block_manager.promote_temporary_block(seq_id, advance, live_state.endpoint, live_state.generation);

        EXPECT_EQ(la_block_manager.get_num_blocks_in_use(), committed_only_in_use)
            << "borrowed rows leaked at step " << step;

        seq_group->finish_iteration();
        processed += advance;
        seq_group->update_processed_tokens_num(processed);
        sequence->append_token(100 + static_cast<int64_t>(step), 0.9f);

        previous_latest_row = static_cast<size_t>(chosen);
        EXPECT_EQ(orchestrator->get_linear_attention_latest_row(seq_id), previous_latest_row)
            << "latest row and block table diverged at step " << step;
    }

    EXPECT_EQ(orchestrator->get_linear_attention_block_table(seq_id).size(), 1u);

    for (auto& seq : seq_group->get_sequences()) {
        scheduler.free_sequence(seq->get_id());
    }
}

// Releasing borrowed rows leaves the latest row unchanged.
TEST(TestScheduler, linear_attention_borrowed_release_keeps_latest_row) {
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
    const size_t latest_row = orchestrator->get_linear_attention_latest_row(seq_id);

    const auto borrowed = la_block_manager.reserve_temporary_blocks(seq_id, 4);
    ASSERT_EQ(borrowed.size(), 4u);
    EXPECT_TRUE(la_block_manager.has_temporary_blocks(seq_id));

    orchestrator->release_linear_attention_temporary_blocks(seq_id);
    EXPECT_FALSE(la_block_manager.has_temporary_blocks(seq_id));
    EXPECT_EQ(orchestrator->get_linear_attention_latest_row(seq_id), latest_row);
    EXPECT_EQ(la_block_manager.get_num_blocks_in_use(), 1u);
    const auto& live_state = la_block_manager.get_linear_attention_live_state(seq_id);
    EXPECT_THROW(la_block_manager.promote_temporary_block(seq_id, 0, live_state.endpoint, live_state.generation),
                 ov::Exception);

    orchestrator->free_sequence(seq_id);
}

// Shared-pool shortages must defer before reservation
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

TEST(TestScheduler, hybrid_linear_attention_defers_when_kv_is_full_and_no_victim_is_available) {
    SchedulerConfig scheduler_config = make_speculative_linear_attention_scheduler_config();
    scheduler_config.num_kv_blocks = 1;
    scheduler_config.num_linear_attention_blocks = 4;

    auto orchestrator = init_hybrid_cache_orchestrator(scheduler_config);
    auto& la_block_manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
    Scheduler scheduler(orchestrator, scheduler_config);
    std::vector<SequenceGroup::Ptr> requests;
    auto groups = make_prompt_processed_sequence_groups(scheduler, requests, 1, {0, 1, 2, 3});
    const auto& group = groups.front();
    const uint64_t seq_id = group->get_running_sequences().front()->get_id();
    group->set_num_validated_tokens(2);

    ASSERT_EQ(orchestrator->get_block_manager(CacheType::KV_CACHE).num_free_blocks(), 0u);
    ASSERT_TRUE(orchestrator->can_reserve_linear_attention_temporary_blocks(seq_id, 3));
    Scheduler::Output output;
    ASSERT_NO_THROW(output = scheduler.schedule(requests));
    EXPECT_TRUE(output.m_scheduled_sequence_groups_ids.empty());
    EXPECT_EQ(group->get_num_scheduled_tokens(), 0u);
    EXPECT_FALSE(la_block_manager.has_temporary_blocks(seq_id));
    EXPECT_EQ(la_block_manager.num_free_blocks(), 3u);

    scheduler.free_sequence(seq_id);
}

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

    // An active lease excludes sequence teardown until its owner explicitly aborts.
    seq_group_a->get_running_sequences()[0]->set_status(SequenceStatus::FINISHED);
    EXPECT_THROW(scheduler.free_sequence(seq_id_a), ov::Exception);
    ASSERT_EQ(out1.m_linear_attention_scratch_leases.count(seq_id_a), 1u);
    out1.m_linear_attention_scratch_leases.erase(seq_id_a);
    EXPECT_FALSE(la_block_manager.has_temporary_blocks(seq_id_a));
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
    EXPECT_THROW(std::ignore = la_block_manager.reserve_temporary_blocks(seq_id, 0), ov::Exception);

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

    // Every temporary slot owns one synchronized row across all layers.
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
    EXPECT_TRUE(multi_layer_manager.can_reserve_temporary_blocks(multi_layer_seq_id, 1));
    const std::vector<int> multi_layer_rows =
        multi_layer_manager.reserve_temporary_blocks(multi_layer_seq_id, 1);
    ASSERT_EQ(multi_layer_rows.size(), 1u);
    EXPECT_TRUE(multi_layer_manager.has_temporary_blocks(multi_layer_seq_id));
    multi_layer_manager.release_temporary_blocks(multi_layer_seq_id);
    EXPECT_FALSE(multi_layer_manager.has_temporary_blocks(multi_layer_seq_id));
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
