#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cstdlib>
#include <limits>
#include <new>

#include "continuous_batching/cache/block_manager.hpp"
#include "continuous_batching/pipeline_impl.hpp"
#include "speculative_decoding/continuous_batching/pipeline_impl.hpp"
#include "speculative_decoding/continuous_batching/mtp_strategy.hpp"
#include "prompt_lookup/continuous_batching_for_prompt_lookup.hpp"

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

    class DraftPipeline : public ContinuousBatchingForSpeculativeDecodingImpl {
    public:
        DraftPipeline(const std::shared_ptr<ov::genai::Scheduler>& scheduler, const SequenceGroup::Ptr& group) {
            m_scheduler = scheduler;
            m_requests = {group};
            m_sampler = std::make_shared<ov::genai::Sampler>();
        }
    };

    class FailurePropagationStrategy : public SpeculativeDecodingImpl {
    public:
        FailurePropagationStrategy(const std::shared_ptr<DraftPipeline>& main,
                                   const std::shared_ptr<DraftPipeline>& draft,
                                   uint64_t request_id,
                                   const ov::genai::GenerationHandle& draft_handle) {
            m_main_pipeline = main;
            m_draft_pipeline = draft;
            m_draft_generations.emplace(request_id, draft_handle);
        }

        size_t draft_handle_count() const {
            return m_draft_generations.size();
        }
    };

    class MtpPipeline : public ContinuousBatchingForMtpDecodingImpl {
    public:
        using ContinuousBatchingForMtpDecodingImpl::validate_prefix_cache_support;

        MtpPipeline(const std::shared_ptr<ov::genai::Scheduler>& scheduler,
                    const std::vector<SequenceGroup::Ptr>& groups) {
            m_scheduler = scheduler;
            m_awaiting_requests = groups;
        }
    };

    class MtpAdmissionChild : public MtpPipeline {
    public:
        explicit MtpAdmissionChild(const std::shared_ptr<ov::genai::Scheduler>& scheduler)
            : MtpPipeline(scheduler, {}) {}

        ov::genai::GenerationHandle add_request(
            uint64_t request_id,
            const ov::Tensor& embeddings,
            const ov::genai::GenerationConfig& config,
            std::optional<ov::Tensor> token_type_ids = std::nullopt,
            std::optional<ov::Tensor> prompt_ids = std::nullopt,
            std::optional<std::unordered_map<std::string, ov::Tensor>> extra_inputs = std::nullopt) override {
            ov::Tensor tokens(ov::element::i64, {1, embeddings.get_shape()[1]});
            const size_t hidden_size = embeddings.get_shape()[2];
            for (size_t token = 0; token < tokens.get_size(); ++token) {
                tokens.data<int64_t>()[token] =
                    static_cast<int64_t>(embeddings.data<const float>()[token * hidden_size]);
            }
            return ContinuousBatchingImpl::add_request(request_id, tokens, config);
        }

        size_t awaiting_count() const {
            return m_awaiting_requests.size();
        }

        size_t processed(uint64_t request_id) const {
            for (const auto& group : m_awaiting_requests) {
                if (group->get_request_id() == request_id) {
                    return group->get_num_processed_tokens();
                }
            }
            return 0;
        }
    };

    class MtpAdmissionStrategy : public MtpDecodingImpl {
    public:
        MtpAdmissionStrategy(const std::shared_ptr<MtpAdmissionChild>& main,
                             const std::shared_ptr<MtpAdmissionChild>& draft)
            : main_child(main),
              draft_child(draft) {
            m_main_pipeline = main;
            m_draft_pipeline = draft;
        }

        size_t draft_handle_count() const {
            return m_draft_generations.size();
        }

        void discard(uint64_t request_id) {
            main_child->discard_awaiting_request(request_id);
            draft_child->discard_awaiting_request(request_id);
            m_draft_generations.erase(request_id);
        }

        bool failed_after_main_restore = false;
        bool failed_with_both_queued = false;

    protected:
        void align_request_pair_processed_prefix(uint64_t request_id) override {
            try {
                MtpDecodingImpl::align_request_pair_processed_prefix(request_id);
            } catch (...) {
                failed_after_main_restore |= main_child->processed(request_id) > 0;
                failed_with_both_queued |= main_child->awaiting_count() == 2 && draft_child->awaiting_count() == 2;
                throw;
            }
        }

    private:
        std::shared_ptr<MtpAdmissionChild> main_child;
        std::shared_ptr<MtpAdmissionChild> draft_child;
    };

    std::shared_ptr<CacheOrchestrator> make_orchestrator() {
        auto orchestrator = std::make_shared<CacheOrchestrator>();
        for (const CacheType type : {CacheType::KV_CACHE, CacheType::LINEAR_ATTENTION_CACHE}) {
            orchestrator->register_cache_type(
                type,
                std::make_unique<testing::NiceMock<RowOnlyCacheManager>>(),
                std::make_unique<BlockManager>(24,
                                               false,
                                               4,
                                               GetParam(),
                                               type == CacheType::LINEAR_ATTENTION_CACHE ? 1 : 0),
                GetParam() > 1);
        }
        return orchestrator;
    }

    SequenceGroup::Ptr make_group(uint64_t request_id) {
        return std::make_shared<SequenceGroup>(request_id,
                                               std::vector<int64_t>{1, 2, 3, 4},
                                               ov::genai::GenerationConfig{});
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

TEST_P(PreparedCacheAllocationFailure, EmbeddingPrefixVerificationRequiresStrategySupport) {
    std::unique_ptr<ContinuousBatchingImpl> lookup = std::make_unique<ContinuousBatchingForPromptLookupImpl>();
    EXPECT_TRUE(lookup->supports_embedding_prefix_verification());
    ContinuousBatchingForMtpDecodingImpl mtp;
    EXPECT_TRUE(mtp.supports_embedding_prefix_verification());
    ContinuousBatchingForSpeculativeDecodingImpl independent_draft;
    EXPECT_FALSE(independent_draft.supports_embedding_prefix_verification());
}

TEST_P(PreparedCacheAllocationFailure, DraftFailurePropagatesToMainPipeline) {
    auto main_scheduler = std::make_shared<ov::genai::Scheduler>(make_orchestrator(), ov::genai::SchedulerConfig{});
    auto draft_scheduler = std::make_shared<ov::genai::Scheduler>(make_orchestrator(), ov::genai::SchedulerConfig{});
    auto main_group = make_group(0);
    auto draft_group = make_group(0);
    auto main_pipeline = std::make_shared<DraftPipeline>(main_scheduler, main_group);
    auto draft_pipeline = std::make_shared<DraftPipeline>(draft_scheduler, draft_group);
    auto main_handle = std::make_shared<ov::genai::GenerationHandleImpl>(main_group->get_generation_stream(),
                                                                         main_group->get_sampling_parameters());
    auto draft_handle = std::make_shared<ov::genai::GenerationHandleImpl>(draft_group->get_generation_stream(),
                                                                          draft_group->get_sampling_parameters());
    FailurePropagationStrategy strategy(main_pipeline, draft_pipeline, 0, draft_handle);

    draft_pipeline->fail_pipeline(std::make_exception_ptr(std::logic_error("draft failure")));

    for (size_t attempt = 0; attempt < 2; ++attempt) {
        EXPECT_THROW(
            try { strategy.step(); } catch (const std::logic_error& error) {
                EXPECT_STREQ(error.what(), "draft failure");
                throw;
            },
            std::logic_error);
    }
    EXPECT_EQ(main_handle->get_status(), ov::genai::GenerationStatus::FAILED);
    EXPECT_TRUE(main_handle->can_read());
    for (size_t attempt = 0; attempt < 2; ++attempt) {
        EXPECT_THROW(
            try { main_handle->read(); } catch (const std::logic_error& error) {
                EXPECT_STREQ(error.what(), "draft failure");
                throw;
            },
            std::logic_error);
    }
    EXPECT_EQ(strategy.draft_handle_count(), 0u);
    EXPECT_TRUE(main_pipeline->is_requests_empty());
    EXPECT_TRUE(draft_pipeline->is_requests_empty());
}

TEST_P(PreparedCacheAllocationFailure, MtpAdmissionRollbackReleasesOnlyFailedRequest) {
    auto orchestrator = make_orchestrator();
    auto scheduler = std::make_shared<ov::genai::Scheduler>(orchestrator, ov::genai::SchedulerConfig{});
    auto failed = make_group(0);
    auto neighbor = make_group(1);
    for (const CacheType type : {CacheType::KV_CACHE, CacheType::LINEAR_ATTENTION_CACHE}) {
        auto& manager = orchestrator->get_block_manager(type);
        allocate_live(manager, failed);
        allocate_live(manager, neighbor);
    }
    MtpPipeline pipeline(scheduler, {failed, neighbor});
    pipeline.discard_awaiting_request(0);
    EXPECT_EQ(pipeline.get_awaiting_requests(), std::vector<SequenceGroup::Ptr>{neighbor});
    for (const CacheType type : {CacheType::KV_CACHE, CacheType::LINEAR_ATTENTION_CACHE}) {
        auto& manager = orchestrator->get_block_manager(type);
        EXPECT_EQ(manager.num_free_blocks(), 23u);
    }
    pipeline.discard_awaiting_request(0);
    pipeline.discard_awaiting_request(1);
    EXPECT_TRUE(pipeline.get_awaiting_requests().empty());
    for (const CacheType type : {CacheType::KV_CACHE, CacheType::LINEAR_ATTENTION_CACHE}) {
        EXPECT_EQ(orchestrator->get_block_manager(type).num_free_blocks(), 24u);
    }
}

TEST_P(PreparedCacheAllocationFailure, MtpPrefixGuardRejectsOnlyLinearAttentionDraft) {
    for (const bool has_linear_attention : {false, true}) {
        auto cache = std::make_shared<CacheOrchestrator>();
        cache->register_cache_type(CacheType::KV_CACHE,
            std::make_unique<testing::NiceMock<RowOnlyCacheManager>>(),
            std::make_unique<BlockManager>(24, true, 4, GetParam()), GetParam() > 1);
        if (has_linear_attention) {
            cache->register_cache_type(CacheType::LINEAR_ATTENTION_CACHE,
                std::make_unique<testing::NiceMock<RowOnlyCacheManager>>(),
                std::make_unique<BlockManager>(24, true, 4, GetParam(), 0, true), GetParam() > 1);
        }
        for (const bool prefix_enabled : {false, true}) {
            ov::genai::SchedulerConfig config;
            config.enable_prefix_caching = prefix_enabled;
            auto scheduler = std::make_shared<ov::genai::Scheduler>(cache, config);
            MtpPipeline pipeline(scheduler, {});
            EXPECT_NO_THROW(pipeline.validate_prefix_cache_support(config, true));
            if (has_linear_attention && prefix_enabled) {
                try {
                    pipeline.validate_prefix_cache_support(config, false);
                    FAIL() << "Expected prefix-enabled linear-attention MTP draft rejection";
                } catch (const ov::Exception& error) {
                    EXPECT_NE(std::string(error.what()).find(
                        "Prefix-enabled MTP draft pipelines with linear-attention state are not supported"),
                        std::string::npos);
                }
            } else {
                EXPECT_NO_THROW(pipeline.validate_prefix_cache_support(config, false));
            }
        }
    }
}

TEST_P(PreparedCacheAllocationFailure, MtpPairedAdmissionRollsBackEveryAllocationFailure) {
    ov::genai::GenerationConfig config;
    config.max_new_tokens = 8;
    config.num_assistant_tokens = 4;
    config.ignore_eos = true;
    config.set_eos_token_id(0);
    ov::Tensor embeddings(ov::element::f32, {1, 13, 1});
    ov::Tensor prompt_ids(ov::element::i64, {1, 13});
    for (size_t token = 0; token < 13; ++token) {
        embeddings.data<float>()[token] = static_cast<float>(token + 1);
        prompt_ids.data<int64_t>()[token] = static_cast<int64_t>(token + 1);
    }

    for (const size_t draft_endpoint : {4u, 8u}) {
        SCOPED_TRACE(draft_endpoint);
        auto main_cache = std::make_shared<CacheOrchestrator>();
        auto draft_cache = std::make_shared<CacheOrchestrator>();
        auto parent = std::make_shared<SequenceGroup>(
            10, std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}, config);
        auto shifted = std::make_shared<SequenceGroup>(
            10, std::vector<int64_t>{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}, config);
        parent->get_sequences().front()->set_prefix_cache_policy(13);
        shifted->get_sequences().front()->set_prefix_cache_policy(12, [parent](size_t length, size_t block_size) {
            return parent->get_sequences().front()->get_hash(length + 1, block_size);
        });
        parent->schedule_tokens(8);
        shifted->schedule_tokens(draft_endpoint);
        std::vector<BlockManager*> managers;
        for (const CacheType type : {CacheType::KV_CACHE, CacheType::LINEAR_ATTENTION_CACHE}) {
            auto manager = std::make_unique<BlockManager>(24, true, 4, GetParam(), 0,
                                                         type == CacheType::LINEAR_ATTENTION_CACHE, 24);
            manager->append_slots(parent);
            manager->publish_completed_blocks(parent->get_sequences().front(), 0, 8);
            manager->free_sequence(parent->get_sequences().front()->get_id());
            managers.push_back(manager.get());
            main_cache->register_cache_type(type, std::make_unique<testing::NiceMock<RowOnlyCacheManager>>(),
                                           std::move(manager), GetParam() > 1);
        }
        auto draft_manager = std::make_unique<BlockManager>(24, true, 4, GetParam());
        draft_manager->append_slots(shifted);
        draft_manager->publish_completed_blocks(shifted->get_sequences().front(), 0, draft_endpoint);
        draft_manager->free_sequence(shifted->get_sequences().front()->get_id());
        managers.push_back(draft_manager.get());
        draft_cache->register_cache_type(CacheType::KV_CACHE,
            std::make_unique<testing::NiceMock<RowOnlyCacheManager>>(), std::move(draft_manager), GetParam() > 1);

        ov::genai::SchedulerConfig scheduler_config;
        scheduler_config.enable_prefix_caching = true;
        auto main_child = std::make_shared<MtpAdmissionChild>(
            std::make_shared<ov::genai::Scheduler>(main_cache, scheduler_config));
        auto draft_child = std::make_shared<MtpAdmissionChild>(
            std::make_shared<ov::genai::Scheduler>(draft_cache, scheduler_config));
        MtpAdmissionStrategy strategy(main_child, draft_child);
        const auto neighbor_handle = strategy.add_request(99, embeddings, config, std::nullopt, prompt_ids);
        const auto main_neighbor = main_child->get_awaiting_requests().front();
        const auto draft_neighbor = draft_child->get_awaiting_requests().front();
        ASSERT_EQ(main_neighbor->get_num_processed_tokens(), draft_endpoint);
        ASSERT_EQ(draft_neighbor->get_num_processed_tokens(), draft_endpoint);
        const uint64_t main_neighbor_id = main_neighbor->get_sequences().front()->get_id();
        const uint64_t draft_neighbor_id = draft_neighbor->get_sequences().front()->get_id();
        const auto neighbor_live = managers[1]->get_linear_attention_live_state(main_neighbor_id);
        std::vector<size_t> free_before;
        std::vector<std::pair<ov::genai::CacheBlock::Ptr, size_t>> references_before;
        for (size_t manager_index = 0; manager_index < managers.size(); ++manager_index) {
            auto& manager = *managers[manager_index];
            free_before.push_back(manager.num_free_blocks());
            const auto& tables = manager.get_block_tables(manager_index == 2 ? draft_neighbor_id : main_neighbor_id);
            for (const auto& layer : tables) {
                for (const auto& row : layer) {
                    references_before.emplace_back(row, row->get_references_count());
                }
            }
        }

        bool completed = false;
        size_t failures = 0;
        for (size_t allocation_index = 0; allocation_index < 2048; ++allocation_index) {
            SCOPED_TRACE(allocation_index);
            ov::genai::GenerationHandle handle;
            try {
                FailAllocation failure(allocation_index);
                handle = strategy.add_request(0, embeddings, config, std::nullopt, prompt_ids);
                completed = true;
            } catch (const std::bad_alloc&) {
                ++failures;
            }
            if (completed) {
                EXPECT_EQ(main_child->processed(0), draft_endpoint);
                EXPECT_EQ(draft_child->processed(0), draft_endpoint);
                EXPECT_EQ(strategy.draft_handle_count(), 2u);
                strategy.discard(0);
            }
            EXPECT_EQ(main_child->get_awaiting_requests(), std::vector<SequenceGroup::Ptr>{main_neighbor});
            EXPECT_EQ(draft_child->get_awaiting_requests(), std::vector<SequenceGroup::Ptr>{draft_neighbor});
            EXPECT_EQ(strategy.draft_handle_count(), 1u);
            EXPECT_FALSE(main_neighbor->handle_stopped());
            EXPECT_FALSE(draft_neighbor->handle_stopped());
            EXPECT_EQ(main_neighbor->get_num_processed_tokens(), draft_endpoint);
            EXPECT_EQ(draft_neighbor->get_num_processed_tokens(), draft_endpoint);
            const auto live = managers[1]->get_linear_attention_live_state(main_neighbor_id);
            EXPECT_EQ(live.rows, neighbor_live.rows);
            EXPECT_EQ(live.endpoint, neighbor_live.endpoint);
            EXPECT_EQ(live.generation, neighbor_live.generation);
            for (size_t manager_index = 0; manager_index < managers.size(); ++manager_index) {
                EXPECT_EQ(managers[manager_index]->num_free_blocks(), free_before[manager_index]);
                EXPECT_EQ(managers[manager_index]->get_num_sequences_with_temporary_blocks(), 0u);
                EXPECT_EQ(managers[manager_index]->get_num_linear_attention_headroom_blocks(), 0u);
            }
            for (const auto& [row, references] : references_before) {
                EXPECT_EQ(row->get_references_count(), references);
                EXPECT_TRUE(row->has_published_hash());
            }
            if (completed) {
                break;
            }
        }
        ASSERT_TRUE(completed);
        EXPECT_GT(failures, 0u);
        EXPECT_TRUE(strategy.failed_with_both_queued);
        EXPECT_TRUE(strategy.failed_after_main_restore);
        RecordProperty("shared_endpoint_" + std::to_string(draft_endpoint) + "_failures", failures);
        strategy.discard(99);
        for (const auto* manager : managers) {
            EXPECT_EQ(manager->num_free_blocks(), 24u);
        }

        completed = false;
        failures = 0;
        for (size_t allocation_index = 0; allocation_index < 2048; ++allocation_index) {
            SCOPED_TRACE(allocation_index);
            ov::genai::GenerationHandle handle;
            try {
                FailAllocation failure(allocation_index);
                handle = strategy.add_request(0, embeddings, config, std::nullopt, prompt_ids);
                completed = true;
            } catch (const std::bad_alloc&) {
                ++failures;
            }
            if (completed) {
                EXPECT_EQ(main_child->processed(0), draft_endpoint);
                EXPECT_EQ(draft_child->processed(0), draft_endpoint);
                strategy.discard(0);
            }
            ASSERT_EQ(main_child->awaiting_count(), 0u);
            ASSERT_EQ(draft_child->awaiting_count(), 0u);
            EXPECT_EQ(strategy.draft_handle_count(), 0u);
            for (auto* manager : managers) {
                EXPECT_EQ(manager->num_free_blocks(), 24u);
                EXPECT_EQ(manager->get_num_sequences_with_temporary_blocks(), 0u);
                EXPECT_EQ(manager->get_num_linear_attention_headroom_blocks(), 0u);
            }
            for (const auto& [row, references] : references_before) {
                EXPECT_EQ(row->get_references_count(), 0u);
                EXPECT_TRUE(row->has_published_hash());
            }
            if (completed) {
                break;
            }
        }
        ASSERT_TRUE(completed);
        EXPECT_GT(failures, 0u);
        RecordProperty("unshared_endpoint_" + std::to_string(draft_endpoint) + "_failures", failures);
    }
}

TEST_P(PreparedCacheAllocationFailure, PromptOnlyPolicyPublishesCompletedPromptAndKeepsGeneratedRowsPrivate) {
    BlockManager manager(12, true, 4, GetParam());
    auto group = make_group(0);
    auto sequence = group->get_sequences().front();
    sequence->set_prefix_cache_policy(group->get_prompt_len());
    group->schedule_tokens(4);
    manager.append_slots(group);
    EXPECT_FALSE(manager.get_block_tables(sequence->get_id()).front().front()->has_published_hash());
    group->finish_iteration();
    manager.publish_completed_blocks(sequence, 0, 4);
    EXPECT_TRUE(manager.get_block_tables(sequence->get_id()).front().front()->has_published_hash());
    for (size_t token = 0; token < 4; ++token) {
        sequence->append_token(9, 0.1f);
    }
    group->schedule_tokens(4);
    manager.append_slots(group);
    group->finish_iteration();
    manager.publish_completed_blocks(sequence, 4, 8);
    EXPECT_FALSE(manager.get_block_tables(sequence->get_id()).front().back()->has_published_hash());
    manager.free_sequence(sequence->get_id());
    auto consumer = std::make_shared<SequenceGroup>(
        1, std::vector<int64_t>{1, 2, 3, 4, 9, 9, 9, 9, 5}, ov::genai::GenerationConfig{});
    const auto plan = manager.get_prefix_restore_plan(consumer);
    EXPECT_EQ(plan.cache_token_position, 4u);
    EXPECT_EQ(manager.num_free_blocks(), 12u);
}

TEST_P(PreparedCacheAllocationFailure, ShiftedDraftIdentityIncludesParentFirstToken) {
    auto first_parent = std::make_shared<SequenceGroup>(
        0, std::vector<int64_t>{1, 2, 3, 4, 5}, ov::genai::GenerationConfig{});
    auto second_parent = std::make_shared<SequenceGroup>(
        1, std::vector<int64_t>{8, 2, 3, 4, 5}, ov::genai::GenerationConfig{});
    auto first_draft = std::make_shared<SequenceGroup>(
        2, std::vector<int64_t>{2, 3, 4, 5}, ov::genai::GenerationConfig{});
    auto second_draft = std::make_shared<SequenceGroup>(
        3, std::vector<int64_t>{2, 3, 4, 5}, ov::genai::GenerationConfig{});
    const auto first_sequence = first_draft->get_sequences().front();
    const auto second_sequence = second_draft->get_sequences().front();
    first_sequence->set_prefix_cache_policy(4, [first_parent](size_t length, size_t block_size) {
        return first_parent->get_sequences().front()->get_hash(length + 1, block_size);
    });
    second_sequence->set_prefix_cache_policy(4, [second_parent](size_t length, size_t block_size) {
        return second_parent->get_sequences().front()->get_hash(length + 1, block_size);
    });
    EXPECT_EQ(first_sequence->get_hash(4, 4), first_parent->get_sequences().front()->get_hash(5, 4));
    EXPECT_NE(first_sequence->get_hash(4, 4), second_sequence->get_hash(4, 4));
    EXPECT_THROW(first_sequence->get_hash(5, 4), ov::Exception);
}

TEST_P(PreparedCacheAllocationFailure, PromptOnlyEmbeddingIdentityIncludesCompleteTokenIds) {
    ov::Tensor embeddings(ov::element::f32, {1, 4, 8});
    std::fill_n(embeddings.data<float>(), embeddings.get_size(), 1.f);
    ov::Tensor first_ids(ov::element::i64, {1, 4});
    ov::Tensor second_ids(ov::element::i64, {1, 4});
    std::fill_n(first_ids.data<int64_t>(), 4, 1);
    std::fill_n(second_ids.data<int64_t>(), 4, 1);
    second_ids.data<int64_t>()[0] = 2;
    auto first = std::make_shared<SequenceGroup>(0, embeddings, ov::genai::GenerationConfig{},
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, first_ids);
    auto second = std::make_shared<SequenceGroup>(1, embeddings, ov::genai::GenerationConfig{},
        std::nullopt, std::nullopt, std::nullopt, std::nullopt, second_ids);
    first->get_sequences().front()->set_prefix_cache_policy(4);
    second->get_sequences().front()->set_prefix_cache_policy(4);
    EXPECT_NE(first->get_sequences().front()->get_hash(4, 4), second->get_sequences().front()->get_hash(4, 4));
}

TEST_P(PreparedCacheAllocationFailure, PrefixPoolGrowthAccountsForOccupiedPromptCheckpoints) {
    auto orchestrator = std::make_shared<CacheOrchestrator>();
    auto manager = std::make_unique<BlockManager>(8, true, 4, GetParam(), 0, true);
    auto group = std::make_shared<SequenceGroup>(
        0, std::vector<int64_t>(32, 1), ov::genai::GenerationConfig{});
    group->schedule_tokens(32);
    manager->append_slots(group);
    group->finish_iteration();
    const uint64_t seq_id = group->get_sequences().front()->get_id();
    manager->advance_linear_attention_live_state(seq_id, 32);
    orchestrator->register_cache_type(CacheType::LINEAR_ATTENTION_CACHE,
        std::make_unique<testing::NiceMock<RowOnlyCacheManager>>(), std::move(manager), GetParam() > 1);
    EXPECT_TRUE(orchestrator->ensure_linear_attention_pool_blocks(7));
    EXPECT_FALSE(orchestrator->ensure_linear_attention_pool_blocks(7));
    auto& blocks = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
    EXPECT_TRUE(blocks.can_prepare_temporary_blocks(seq_id, 5));
    const auto live = blocks.get_linear_attention_live_state(seq_id);
    const auto scratch = blocks.prepare_temporary_blocks(seq_id, 5);
    EXPECT_EQ(scratch.block_indices.size(), 5u);
    auto promotion = blocks.prepare_temporary_promotions({{seq_id, 1, 32, live.generation}});
    EXPECT_EQ(promotion.apply().size(), 1u);
    EXPECT_GT(blocks.get_num_linear_attention_headroom_blocks(), 0u);
    EXPECT_FALSE(orchestrator->ensure_linear_attention_pool_blocks(7));
    EXPECT_TRUE(blocks.can_prepare_temporary_blocks(seq_id, 5));
    orchestrator->free_sequence(seq_id);
}

TEST_P(PreparedCacheAllocationFailure, PrefixPromotionPreservesPublishedBaseAndAppliesWithoutAllocation) {
    BlockManager manager(12, true, 4, GetParam(), 0, true, 12);
    auto group = make_group(0);
    group->schedule_tokens(4);
    manager.append_slots(group);
    group->finish_iteration();
    const auto sequence = group->get_sequences().front();
    const uint64_t seq_id = sequence->get_id();
    manager.advance_linear_attention_live_state(seq_id, 4);
    const auto original = manager.get_linear_attention_live_state(seq_id);
    const auto scratch = manager.prepare_temporary_blocks(seq_id, 5);
    const std::vector<BlockManager::TemporaryPromotionRequest> requests{{seq_id, 5, 4, original.generation}};
    bool completed = false;
    size_t failures = 0;
    for (size_t allocation_index = 0; allocation_index < 512; ++allocation_index) {
        SCOPED_TRACE(allocation_index);
        try {
            FailAllocation failure(allocation_index);
            auto prepared = manager.prepare_temporary_promotions(requests);
            allocations_remaining = 0;
            const auto& indices = prepared.apply();
            completed = indices.front() == static_cast<size_t>(scratch.block_indices.back());
        } catch (const std::bad_alloc&) {
            ++failures;
        }
        if (completed) {
            break;
        }
        EXPECT_EQ(manager.get_linear_attention_live_state(seq_id).rows, original.rows);
        EXPECT_EQ(manager.get_linear_attention_live_state(seq_id).endpoint, 4u);
        EXPECT_TRUE(manager.has_temporary_blocks(seq_id));
        EXPECT_EQ(manager.get_block_tables(seq_id).front().size(), 1u);
    }
    ASSERT_TRUE(completed);
    EXPECT_GT(failures, 0u);
    EXPECT_EQ(manager.get_linear_attention_live_state(seq_id).endpoint, 9u);
    EXPECT_EQ(manager.get_block_table_logical_start(seq_id), 1u);
    EXPECT_EQ(manager.get_block_tables(seq_id).front().size(), 2u);
    EXPECT_EQ(manager.get_block_tables(seq_id).front().front()->get_index(), scratch.block_indices[3]);
    EXPECT_FALSE(manager.has_temporary_blocks(seq_id));
    for (const auto& row : original.rows) {
        EXPECT_TRUE(row->has_published_hash());
        EXPECT_EQ(row->get_references_count(), 0);
    }
    auto consumer = std::make_shared<SequenceGroup>(
        1, std::vector<int64_t>{1, 2, 3, 4, 5}, ov::genai::GenerationConfig{});
    ASSERT_TRUE(manager.restore_cached_blocks(consumer));
    EXPECT_EQ(consumer->get_num_processed_tokens(), 4u);
    manager.free_sequence(consumer->get_sequences().front()->get_id());
    manager.free_sequence(seq_id);
    EXPECT_EQ(manager.num_free_blocks(), 12u);
}

TEST_P(PreparedCacheAllocationFailure, HybridDraftRejectionRestoresBeforeRecomputation) {
    for (const bool prefix_caching : {false, true}) {
        SCOPED_TRACE(prefix_caching);
        auto orchestrator = std::make_shared<CacheOrchestrator>();
        for (const CacheType type : {CacheType::KV_CACHE, CacheType::LINEAR_ATTENTION_CACHE}) {
            const bool linear_attention = type == CacheType::LINEAR_ATTENTION_CACHE;
            orchestrator->register_cache_type(type, std::make_unique<testing::NiceMock<RowOnlyCacheManager>>(),
                std::make_unique<BlockManager>(16, prefix_caching, 4, GetParam(),
                    linear_attention && !prefix_caching ? 1 : 0, linear_attention && prefix_caching), GetParam() > 1);
        }
        ov::genai::SchedulerConfig config;
        config.enable_prefix_caching = prefix_caching;
        auto scheduler = std::make_shared<ov::genai::Scheduler>(orchestrator, config, false, 1, false);
        ov::genai::GenerationConfig generation_config;
        generation_config.num_assistant_tokens = 4;
        auto group = std::make_shared<SequenceGroup>(
            0, std::vector<int64_t>{1, 2, 3, 4, 5}, generation_config);
        DraftPipeline pipeline(scheduler, group);
        const ov::genai::GeneratedSequences initial{{0, ov::genai::GeneratedSequence(
            {10, 11, 12, 13}, {0.1f, 0.1f, 0.1f, 0.1f})}};
        EXPECT_EQ(pipeline.update_request(0, initial, true).inserted_tokens_cnt, 4u);
        group->schedule_tokens(8);
        orchestrator->append_slots(group);
        group->finish_iteration();
        const uint64_t seq_id = group->get_sequences().front()->get_id();
        auto& manager = orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE);
        manager.advance_linear_attention_live_state(seq_id, 8);
        const ov::genai::GeneratedSequences accepted{{0, ov::genai::GeneratedSequence(
            {10, 99}, {0.1f, 0.1f})}};
        const auto update = pipeline.update_request(0, accepted, true);
        EXPECT_EQ(update.removed_tokens_cnt, 3u);
        EXPECT_EQ(update.inserted_tokens_cnt, 1u);
        EXPECT_EQ(group->get_num_processed_tokens(), prefix_caching ? 4u : 0u);
        EXPECT_EQ(group->get_num_available_tokens_for_batching(), prefix_caching ? 2u : 6u);
        EXPECT_EQ(manager.has_block_table(seq_id), prefix_caching);
        if (prefix_caching) {
            EXPECT_EQ(manager.get_linear_attention_live_state(seq_id).endpoint, 4u);
        }
        orchestrator->free_sequence(seq_id);
        EXPECT_EQ(manager.num_free_blocks(), 16u);
    }
}

TEST_P(PreparedCacheAllocationFailure, AbandonedAndStaleRestorePreparationPreserveOwnership) {
    BlockManager manager(1, true, 4, GetParam(), 0, true, 1);
    auto producer = make_group(0);
    producer->schedule_tokens(4);
    manager.append_slots(producer);
    const auto rows = manager.get_block_tables(producer->get_sequences().front()->get_id());
    manager.free_sequence(producer->get_sequences().front()->get_id());
    auto consumer = std::make_shared<SequenceGroup>(
        1, std::vector<int64_t>{1, 2, 3, 4, 5}, ov::genai::GenerationConfig{});
    const uint64_t seq_id = consumer->get_sequences().front()->get_id();
    const auto plan = manager.get_prefix_restore_plan(consumer);
    ASSERT_FALSE(plan.empty());
    {
        auto prepared = manager.prepare_prefix_restore(consumer, plan);
        ASSERT_TRUE(prepared.has_value());
    }
    EXPECT_FALSE(manager.has_block_table(seq_id));
    EXPECT_EQ(manager.num_free_blocks(), 1u);
    for (const auto& layer : rows) {
        EXPECT_EQ(layer.front()->get_references_count(), 0);
    }
    auto competitor = std::make_shared<SequenceGroup>(
        2, std::vector<int64_t>{6, 7, 8, 9}, ov::genai::GenerationConfig{});
    competitor->schedule_tokens(4);
    manager.append_slots(competitor);
    EXPECT_FALSE(manager.prepare_prefix_restore(consumer, plan).has_value());
    EXPECT_FALSE(manager.has_block_table(seq_id));
    EXPECT_EQ(consumer->get_num_processed_tokens(), 0u);
    manager.free_sequence(competitor->get_sequences().front()->get_id());
    EXPECT_EQ(manager.num_free_blocks(), 1u);
}

TEST_P(PreparedCacheAllocationFailure, HybridRestorePreparationIsAtomic) {
    auto orchestrator = std::make_shared<CacheOrchestrator>();
    auto producer = make_group(0);
    producer->schedule_tokens(4);
    for (const CacheType type : {CacheType::KV_CACHE, CacheType::LINEAR_ATTENTION_CACHE}) {
        auto manager = std::make_unique<BlockManager>(8, true, 4, GetParam(), 0,
                                                      type == CacheType::LINEAR_ATTENTION_CACHE);
        manager->append_slots(producer);
        manager->free_sequence(producer->get_sequences().front()->get_id());
        orchestrator->register_cache_type(type, std::make_unique<testing::NiceMock<RowOnlyCacheManager>>(),
                                           std::move(manager), GetParam() > 1);
    }
    auto consumer = std::make_shared<SequenceGroup>(
        1, std::vector<int64_t>{1, 2, 3, 4, 5}, ov::genai::GenerationConfig{});
    const uint64_t seq_id = consumer->get_sequences().front()->get_id();
    bool completed = false;
    size_t failures = 0;
    for (size_t allocation_index = 0; allocation_index < 512; ++allocation_index) {
        SCOPED_TRACE(allocation_index);
        try {
            FailAllocation failure(allocation_index);
            orchestrator->restore_cached_blocks(consumer);
            completed = true;
        } catch (const std::bad_alloc&) {
            ++failures;
        }
        EXPECT_EQ(consumer->get_num_processed_tokens(), completed ? 4u : 0u);
        for (const CacheType type : {CacheType::KV_CACHE, CacheType::LINEAR_ATTENTION_CACHE}) {
            auto& manager = orchestrator->get_block_manager(type);
            EXPECT_EQ(manager.has_block_table(seq_id), completed);
            EXPECT_EQ(manager.num_free_blocks(), completed ? 7u : 8u);
        }
        if (completed) {
            break;
        }
    }
    ASSERT_TRUE(completed);
    EXPECT_GT(failures, 0u);
    EXPECT_EQ(orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE)
                  .get_linear_attention_live_state(seq_id).endpoint, 4u);
    orchestrator->free_sequence(seq_id);
}

TEST_P(PreparedCacheAllocationFailure, PublicationPreparationPreservesWholeSetOnEveryAllocationFailure) {
    BlockManager manager(8, true, 4, GetParam());
    auto source = std::make_shared<SequenceGroup>(
        0, std::vector<int64_t>{1, 2, 3, 4, 5, 6}, ov::genai::GenerationConfig{});
    source->schedule_tokens(6);
    manager.append_slots(source);
    source->finish_iteration();
    auto consumer = std::make_shared<SequenceGroup>(
        1, std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}, ov::genai::GenerationConfig{});
    ASSERT_TRUE(manager.restore_cached_blocks(consumer));
    consumer->schedule_tokens(2);
    manager.append_slots(consumer);
    consumer->finish_iteration();
    consumer->schedule_tokens(4);
    manager.append_slots(consumer);
    consumer->finish_iteration();
    consumer->update_processed_tokens_num(11);
    manager.free_empty_physical_blocks(consumer);
    consumer->schedule_tokens(1);
    manager.append_slots(consumer);
    consumer->finish_iteration();
    auto sequence = consumer->get_sequences().front();
    const auto rows = manager.get_block_tables(sequence->get_id());
    for (const auto& layer : rows) {
        ASSERT_EQ(layer.size(), 3u);
        ASSERT_FALSE(layer[1]->has_published_hash());
        ASSERT_FALSE(layer.back()->has_published_hash());
    }
    bool completed = false;
    size_t failures = 0;
    for (size_t allocation_index = 0; allocation_index < 256; ++allocation_index) {
        SCOPED_TRACE(allocation_index);
        try {
            FailAllocation failure(allocation_index);
            manager.publish_completed_blocks(sequence, 6, 12);
            completed = true;
        } catch (const std::bad_alloc&) {
            ++failures;
        }
        for (const auto& layer : rows) {
            EXPECT_EQ(layer[1]->has_published_hash(), completed);
            EXPECT_EQ(layer.back()->has_published_hash(), completed);
        }
        auto restored = std::make_shared<SequenceGroup>(
            2, consumer->get_prompt_ids(), ov::genai::GenerationConfig{});
        ASSERT_TRUE(manager.restore_cached_blocks(restored));
        EXPECT_EQ(restored->get_num_processed_tokens(), completed ? 12u : 6u);
        manager.free_sequence(restored->get_sequences().front()->get_id());
        if (completed) {
            break;
        }
    }
    EXPECT_TRUE(completed);
    EXPECT_GT(failures, 0u);
    manager.free_sequence(source->get_sequences().front()->get_id());
    manager.free_sequence(sequence->get_id());
}

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

TEST_P(PreparedCacheAllocationFailure, ScratchEvictionPreparationPreservesRegistryOnEveryAllocationFailure) {
    BlockManager manager(5, true, 4, GetParam(), 0, true, 5);
    auto producer = make_group(0);
    const uint64_t producer_id = producer->get_sequences().front()->get_id();
    producer->schedule_tokens(4);
    manager.append_slots(producer);
    producer->finish_iteration();
    const auto cached = manager.get_block_tables(producer_id);
    const auto hash = cached.front().front()->get_hash();
    manager.free_sequence(producer_id);
    auto active = std::make_shared<SequenceGroup>(
        1, std::vector<int64_t>{5, 6, 7, 8}, ov::genai::GenerationConfig{});
    const uint64_t active_id = active->get_sequences().front()->get_id();
    active->schedule_tokens(4);
    manager.append_slots(active);
    active->finish_iteration();
    const auto active_rows = manager.get_block_tables(active_id);
    bool completed = false;
    size_t failures = 0;
    for (size_t allocation_index = 0; allocation_index < 256; ++allocation_index) {
        SCOPED_TRACE(allocation_index);
        try {
            FailAllocation failure(allocation_index);
            std::ignore = manager.reserve_temporary_blocks(active_id, 4);
            completed = true;
        } catch (const std::bad_alloc&) {
            ++failures;
        }
        EXPECT_EQ(manager.get_block_tables(active_id), active_rows);
        EXPECT_EQ(manager.get_total_block_count(), 5u);
        EXPECT_EQ(manager.num_free_blocks(), completed ? 0u : 4u);
        EXPECT_EQ(manager.has_temporary_blocks(active_id), completed);
        for (size_t layer = 0; layer < GetParam(); ++layer) {
            EXPECT_EQ(cached[layer].front()->has_published_hash(), !completed);
            EXPECT_EQ(cached[layer].front()->get_references_count(), completed ? 1u : 0u);
            if (!completed) {
                EXPECT_EQ(cached[layer].front()->get_hash(), hash);
            }
        }
        auto consumer = std::make_shared<SequenceGroup>(
            2, std::vector<int64_t>{1, 2, 3, 4, 9}, ov::genai::GenerationConfig{});
        const uint64_t consumer_id = consumer->get_sequences().front()->get_id();
        manager.restore_cached_blocks(consumer);
        EXPECT_EQ(consumer->get_num_processed_tokens(), completed ? 0u : 4u);
        if (!completed) {
            EXPECT_EQ(manager.get_block_tables(consumer_id), cached);
            manager.free_sequence(consumer_id);
        } else {
            EXPECT_FALSE(manager.has_block_table(consumer_id));
        }
        if (completed) {
            FailAllocation failure(0);
            manager.release_temporary_blocks(active_id);
            break;
        }
    }
    EXPECT_TRUE(completed);
    EXPECT_GT(failures, 0u);
    manager.free_sequence(active_id);
    EXPECT_EQ(manager.num_free_blocks(), manager.get_total_block_count());
}

INSTANTIATE_TEST_SUITE_P(Layers, PreparedCacheAllocationFailure, testing::Values(1u, 2u));

}  // namespace
