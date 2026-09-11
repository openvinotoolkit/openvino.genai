#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cstdlib>
#include <limits>
#include <new>

#include "continuous_batching/cache/block_manager.hpp"
#include "continuous_batching/pipeline_impl.hpp"

namespace {

thread_local size_t allocations_remaining = std::numeric_limits<size_t>::max();

class FailAllocation {
public:
    explicit FailAllocation(size_t successful_allocations) {
        allocations_remaining = successful_allocations;
    }
    ~FailAllocation() {
        allocations_remaining = std::numeric_limits<size_t>::max();
    }
    FailAllocation(const FailAllocation&) = delete;
    FailAllocation& operator=(const FailAllocation&) = delete;
};

}  // namespace

void* operator new(std::size_t size) {
    if (allocations_remaining != std::numeric_limits<size_t>::max()) {
        if (allocations_remaining == 0) {
            throw std::bad_alloc();
        }
        --allocations_remaining;
    }
    if (void* allocation = std::malloc(size == 0 ? 1 : size)) {
        return allocation;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
    return ::operator new(size);
}

void operator delete(void* allocation) noexcept {
    std::free(allocation);
}

void operator delete[](void* allocation) noexcept {
    ::operator delete(allocation);
}

void operator delete(void* allocation, std::size_t) noexcept {
    ::operator delete(allocation);
}

void operator delete[](void* allocation, std::size_t) noexcept {
    ::operator delete(allocation);
}

namespace {

using ov::genai::BlockManager;
using ov::genai::CacheOrchestrator;
using ov::genai::CacheType;
using ov::genai::SequenceGroup;

class RowOnlyCacheManager : public ov::genai::ICacheManager {
public:
    using CopyMap = std::map<size_t, std::list<size_t>>;
    MOCK_METHOD(void, allocate_cache_if_needed, (size_t), (override));
    MOCK_METHOD(void, copy_blocks, (const CopyMap&), (override));
    MOCK_METHOD(void, clear, (), (override));
    MOCK_METHOD(size_t, get_num_layers, (), (const, override));
    MOCK_METHOD(size_t, get_num_cache_tensors, (), (const, override));
    MOCK_METHOD(size_t, get_block_size, (), (const, override));
    MOCK_METHOD(std::string, get_device, (), (const, override));
    MOCK_METHOD(size_t, get_block_size_in_bytes, (), (const, override));
    MOCK_METHOD(size_t, get_num_allocated_blocks, (), (const, override));
};

class PreparedCacheAllocationFailure : public testing::TestWithParam<size_t>,
                                       public ov::genai::ContinuousBatchingPipeline {
protected:
    class Pipeline : public ContinuousBatchingImpl {
    public:
        Pipeline(const std::shared_ptr<ov::genai::Scheduler>& scheduler,
                 const std::vector<SequenceGroup::Ptr>& groups) {
            m_scheduler = scheduler;
            m_requests = groups;
        }

        void commit(ov::genai::Scheduler::Output& output, const ov::genai::SamplerOutput& sampling) {
            _commit_linear_attention_checkpoint_transactions(output, sampling);
        }
    };

    std::shared_ptr<CacheOrchestrator> make_orchestrator() {
        auto orchestrator = std::make_shared<CacheOrchestrator>();
        for (const CacheType type : {CacheType::KV_CACHE, CacheType::LINEAR_ATTENTION_CACHE}) {
            orchestrator->register_cache_type(
                type,
                std::make_unique<testing::NiceMock<RowOnlyCacheManager>>(),
                std::make_unique<BlockManager>(24, false, 4, GetParam(),
                                               type == CacheType::LINEAR_ATTENTION_CACHE ? 1 : 0),
                GetParam() > 1);
        }
        return orchestrator;
    }

    SequenceGroup::Ptr make_group(uint64_t request_id) {
        return std::make_shared<SequenceGroup>(
            request_id, std::vector<int64_t>{1, 2, 3, 4}, ov::genai::GenerationConfig{});
    }

    void allocate_live(BlockManager& manager, const SequenceGroup::Ptr& group) {
        const auto sequence = group->get_sequences().front();
        manager.allocate_tokens(sequence, group, 4, 4);
        ov::genai::BlocksPerLayer rows;
        for (size_t layer = 0; layer < GetParam(); ++layer) {
            rows.push_back(manager.get_block_table(sequence->get_id(), layer).front());
        }
        manager.set_linear_attention_live_state(sequence->get_id(), 4, rows);
    }
};

TEST_P(PreparedCacheAllocationFailure, ScratchReservationRollsBackEveryAllocationFailure) {
    auto orchestrator = make_orchestrator();
    auto& manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
    auto group = make_group(0);
    const uint64_t sequence_id = group->get_sequences().front()->get_id();
    allocate_live(manager, group);
    const auto original = manager.get_linear_attention_live_state(sequence_id);
    const size_t free_before = manager.num_free_blocks();
    bool completed = false;
    size_t failures = 0;
    for (size_t allocation_index = 0; allocation_index < 256; ++allocation_index) {
        SCOPED_TRACE(allocation_index);
        try {
            FailAllocation failure(allocation_index);
            auto lease = std::make_unique<CacheOrchestrator::LinearAttentionScratchLease>(
                orchestrator->prepare_linear_attention_scratch(sequence_id, 4));
            completed = true;
        } catch (const std::bad_alloc&) {
            ++failures;
        }
        const auto& current = manager.get_linear_attention_live_state(sequence_id);
        EXPECT_EQ(current.endpoint, original.endpoint);
        EXPECT_EQ(current.generation, original.generation);
        EXPECT_EQ(current.rows, original.rows);
        EXPECT_FALSE(manager.has_temporary_blocks(sequence_id));
        EXPECT_EQ(manager.num_free_blocks(), free_before);
        for (const auto& row : original.rows) {
            EXPECT_EQ(row->get_references_count(), 2u);
        }
        if (completed) {
            break;
        }
    }
    EXPECT_TRUE(completed);
    EXPECT_GT(failures, 0u);
    manager.free_sequence(sequence_id);
    EXPECT_EQ(manager.num_free_blocks(), manager.get_total_block_count());
}

TEST_P(PreparedCacheAllocationFailure, CombinedPreparationIsAtomicAndApplyDoesNotAllocate) {
    for (const size_t accepted_depth : {1u, 2u, 4u}) {
        SCOPED_TRACE(accepted_depth);
        BlockManager kv_manager(24, false, 4, GetParam());
        BlockManager la_manager(12, false, 4, GetParam(), 1);
        const std::vector<SequenceGroup::Ptr> groups{make_group(0), make_group(1)};
        std::vector<BlockManager::TailReleaseTarget> targets;
        std::vector<BlockManager::TemporaryPromotionRequest> promotions;
        std::vector<BlockManager::LinearAttentionLiveState> original_states;
        for (const auto& group : groups) {
            const auto sequence = group->get_sequences().front();
            const uint64_t sequence_id = sequence->get_id();
            kv_manager.allocate_tokens(sequence, group, 16, 4);
            allocate_live(la_manager, group);
            original_states.push_back(la_manager.get_linear_attention_live_state(sequence_id));
            const auto scratch = la_manager.prepare_temporary_blocks(sequence_id, 4);
            targets.push_back({sequence_id, 4 + accepted_depth});
            promotions.push_back({sequence_id, accepted_depth, scratch.endpoint, scratch.generation});
        }
        const size_t kv_free_before = kv_manager.num_free_blocks();
        const size_t la_free_before = la_manager.num_free_blocks();
        bool completed = false;
        size_t failures = 0;
        for (size_t allocation_index = 0; allocation_index < 256; ++allocation_index) {
            SCOPED_TRACE(allocation_index);
            try {
                FailAllocation failure(allocation_index);
                auto kv_prepared = kv_manager.prepare_tail_releases(targets);
                auto la_prepared = la_manager.prepare_temporary_promotions(promotions);
                completed = true;
            } catch (const std::bad_alloc&) {
                ++failures;
            }
            EXPECT_EQ(kv_manager.num_free_blocks(), kv_free_before);
            EXPECT_EQ(la_manager.num_free_blocks(), la_free_before);
            for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
                const uint64_t sequence_id = groups[group_index]->get_sequences().front()->get_id();
                const auto& current = la_manager.get_linear_attention_live_state(sequence_id);
                EXPECT_EQ(current.rows, original_states[group_index].rows);
                EXPECT_EQ(current.endpoint, original_states[group_index].endpoint);
                EXPECT_EQ(current.generation, original_states[group_index].generation);
                EXPECT_TRUE(la_manager.has_temporary_blocks(sequence_id));
                for (size_t layer = 0; layer < GetParam(); ++layer) {
                    EXPECT_EQ(kv_manager.get_block_table(sequence_id, layer).size(), 4u);
                    for (const auto& block : kv_manager.get_block_table(sequence_id, layer)) {
                        EXPECT_EQ(block->get_references_count(), 1u);
                    }
                    EXPECT_EQ(current.rows[layer]->get_references_count(), 2u);
                }
            }
            if (completed) {
                break;
            }
        }
        ASSERT_TRUE(completed);
        EXPECT_GT(failures, 0u);
        {
            auto kv_prepared = kv_manager.prepare_tail_releases(targets);
            auto la_prepared = la_manager.prepare_temporary_promotions(promotions);
            FailAllocation failure(0);
            kv_prepared.apply();
            std::ignore = la_prepared.apply();
        }
        EXPECT_EQ(kv_manager.num_free_blocks(), kv_free_before + 4);
        EXPECT_EQ(la_manager.num_free_blocks(), la_free_before + 8);
        for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
            const uint64_t sequence_id = groups[group_index]->get_sequences().front()->get_id();
            const auto& current = la_manager.get_linear_attention_live_state(sequence_id);
            EXPECT_EQ(current.endpoint, 4 + accepted_depth);
            EXPECT_EQ(current.generation, original_states[group_index].generation + 1);
            EXPECT_FALSE(la_manager.has_temporary_blocks(sequence_id));
            for (size_t layer = 0; layer < GetParam(); ++layer) {
                EXPECT_EQ(kv_manager.get_block_table(sequence_id, layer).size(), 2u);
                EXPECT_EQ(current.rows[layer]->get_references_count(), 2u);
            }
            kv_manager.free_sequence(sequence_id);
            la_manager.free_sequence(sequence_id);
        }
        EXPECT_EQ(kv_manager.num_free_blocks(), kv_manager.get_total_block_count());
        EXPECT_EQ(la_manager.num_free_blocks(), la_manager.get_total_block_count());
    }
}

TEST_P(PreparedCacheAllocationFailure, PipelinePreparationPreservesBothCountersAtEveryAllocationFailure) {
    auto orchestrator = make_orchestrator();
    auto& kv_manager = orchestrator->get_block_manager(CacheType::KV_CACHE);
    auto& la_manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
    const std::vector<SequenceGroup::Ptr> groups{make_group(0), make_group(1)};
    ov::genai::SchedulerConfig config;
    config.num_kv_blocks = 24;
    config.num_linear_attention_blocks = 24;
    auto scheduler = std::make_shared<ov::genai::Scheduler>(orchestrator, config);
    Pipeline pipeline(scheduler, groups);
    ov::genai::Scheduler::Output output;
    ov::genai::SamplerOutput sampling;
    std::vector<BlockManager::LinearAttentionLiveState> original_states;
    std::vector<int> selected_rows;
    for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
        const auto& group = groups[group_index];
        const auto sequence = group->get_sequences().front();
        const uint64_t sequence_id = sequence->get_id();
        kv_manager.allocate_tokens(sequence, group, 16, 4);
        allocate_live(la_manager, group);
        original_states.push_back(la_manager.get_linear_attention_live_state(sequence_id));
        group->update_processed_tokens_num(4);
        group->schedule_tokens(4);
        output.m_scheduled_sequence_groups_ids.push_back(group_index);
        output.m_kv_paged_attention_data[sequence_id].num_processed_tokens_before = 4;
        auto& paging = output.m_linear_attention_paging_data[sequence_id];
        paging.num_processed_tokens_before = 4;
        paging.is_speculative = true;
        auto lease = std::make_unique<CacheOrchestrator::LinearAttentionScratchLease>(
            orchestrator->prepare_linear_attention_scratch(sequence_id, 4));
        selected_rows.push_back(lease->block_indices()[1]);
        output.m_linear_attention_scratch_leases.emplace(sequence_id, std::move(lease));
        sampling.acceptance_by_sequence.emplace(sequence_id, ov::genai::SamplerOutput::AcceptanceResult{2, 6});
    }
    const size_t kv_free_before = kv_manager.num_free_blocks();
    const size_t la_free_before = la_manager.num_free_blocks();
    bool completed = false;
    size_t failures = 0;
    for (size_t allocation_index = 0; allocation_index < 256; ++allocation_index) {
        SCOPED_TRACE(allocation_index);
        try {
            FailAllocation failure(allocation_index);
            pipeline.commit(output, sampling);
            completed = true;
        } catch (const std::bad_alloc&) {
            ++failures;
        }
        for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
            const auto& group = groups[group_index];
            const uint64_t sequence_id = group->get_sequences().front()->get_id();
            const auto& current = la_manager.get_linear_attention_live_state(sequence_id);
            EXPECT_EQ(group->get_num_processed_tokens(), completed ? 6u : 4u);
            EXPECT_EQ(group->get_num_scheduled_tokens(), completed ? 0u : 4u);
            EXPECT_EQ(current.endpoint, completed ? 6u : 4u);
            EXPECT_EQ(current.generation, original_states[group_index].generation + (completed ? 1 : 0));
            EXPECT_EQ(la_manager.has_temporary_blocks(sequence_id), !completed);
            if (!completed) {
                EXPECT_EQ(current.rows, original_states[group_index].rows);
                EXPECT_EQ(output.m_linear_attention_scratch_leases.at(sequence_id)->state(),
                          CacheOrchestrator::LinearAttentionScratchLease::State::ACTIVE);
            } else {
                EXPECT_EQ(current.rows.front()->get_index(), selected_rows[group_index]);
            }
            for (size_t layer = 0; layer < GetParam(); ++layer) {
                EXPECT_EQ(kv_manager.get_block_table(sequence_id, layer).size(), completed ? 2u : 4u);
                EXPECT_EQ(current.rows[layer]->get_references_count(), 2u);
            }
        }
        EXPECT_EQ(kv_manager.num_free_blocks(), kv_free_before + (completed ? 4 : 0));
        EXPECT_EQ(la_manager.num_free_blocks(), la_free_before + (completed ? 8 : 0));
        EXPECT_EQ(output.m_linear_attention_scratch_leases.size(), completed ? 0u : 2u);
        if (completed) {
            break;
        }
    }
    EXPECT_TRUE(completed);
    EXPECT_GT(failures, 0u);
    output.m_linear_attention_scratch_leases.clear();
    for (const auto& group : groups) {
        const uint64_t sequence_id = group->get_sequences().front()->get_id();
        kv_manager.free_sequence(sequence_id);
        la_manager.free_sequence(sequence_id);
    }
    EXPECT_EQ(kv_manager.num_free_blocks(), kv_manager.get_total_block_count());
    EXPECT_EQ(la_manager.num_free_blocks(), la_manager.get_total_block_count());
}

INSTANTIATE_TEST_SUITE_P(Layers, PreparedCacheAllocationFailure, testing::Values(1u, 2u));

}  // namespace
