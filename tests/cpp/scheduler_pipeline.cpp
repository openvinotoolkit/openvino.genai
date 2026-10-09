// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <exception>
#include <future>
#include <numeric>
#include <set>
#include <typeinfo>
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

namespace {

void expect_same_exception(const std::exception_ptr& actual, const std::exception_ptr& expected) {
    ASSERT_NE(actual, nullptr);
    ASSERT_NE(expected, nullptr);
    try {
        std::rethrow_exception(expected);
    } catch (const std::exception& expected_error) {
        try {
            std::rethrow_exception(actual);
        } catch (const std::exception& actual_error) {
            EXPECT_EQ(typeid(actual_error), typeid(expected_error));
            EXPECT_STREQ(actual_error.what(), expected_error.what());
        } catch (...) {
            FAIL() << "Actual failure is not a std::exception";
        }
    } catch (...) {
        FAIL() << "Expected failure is not a std::exception";
    }
}

}  // namespace

class CBFailureBoundaryTest : public testing::Test, public ContinuousBatchingPipeline {
protected:
    class HandleReleaseGuard {
    public:
        explicit HandleReleaseGuard(const std::vector<GenerationHandle>& handles) : m_handles(handles) {}

        ~HandleReleaseGuard() {
            for (const GenerationHandle& handle : m_handles) {
                handle->stop();
            }
        }

    private:
        const std::vector<GenerationHandle>& m_handles;
    };

    class PipelineTestInstance : public ContinuousBatchingPipeline::ContinuousBatchingImpl {
    public:
        PipelineTestInstance() {
            SchedulerConfig scheduler_config;
            scheduler_config.num_kv_blocks = 8;
            scheduler_config.max_num_batched_tokens = 8;
            scheduler_config.max_num_seqs = 4;
            m_scheduler = std::make_shared<ContinuousBatchingScheduler>(init_cache_orchestrator(scheduler_config, 4, 1), scheduler_config);
            m_sampler = std::make_shared<Sampler>();
        }

        GenerationHandle add_test_request(uint64_t request_id, bool active) {
            GenerationConfig config = utils::get_greedy_config();
            config.max_new_tokens = 4;
            const auto request = std::make_shared<SequenceGroup>(request_id, std::vector<int64_t>{1}, config);
            GenerationHandle handle =
                std::make_shared<GenerationHandleImpl>(request->get_generation_stream(), config);
            if (active) {
                m_requests.push_back(request);
            } else {
                m_awaiting_requests.push_back(request);
            }
            std::vector<SequenceGroup::Ptr> scheduled_requests{request};
            m_scheduler->schedule(scheduled_requests);
            m_sequence_ids.emplace(request_id, request->get_sequences().front()->get_id());
            m_sampler->create_logit_processor(request_id, config, request->get_prompt_ids());
            m_seq_group_id_to_cache_eviction_algo_map.emplace(request->get_sequences().front()->get_id(),
                                                               CacheEvictionAlgorithm{});
            return handle;
        }

        bool is_clean() {
            return m_requests.empty() && m_awaiting_requests.empty() &&
                   m_seq_group_id_to_cache_eviction_algo_map.empty();
        }

        bool has_block_table(uint64_t request_id) {
            return m_scheduler->has_block_table(m_sequence_ids.at(request_id));
        }

        bool has_sampler_context(uint64_t request_id) {
            try {
                m_sampler->get_logit_processor(request_id);
                return true;
            } catch (...) {
                return false;
            }
        }

        void fail_next_step() {
            m_step_failure = std::make_exception_ptr(std::runtime_error("injected step failure"));
        }

        const std::exception_ptr& get_step_failure() const {
            return m_step_failure;
        }

    protected:
        void _pull_awaiting_requests() override {
            if (m_step_failure) {
                std::rethrow_exception(m_step_failure);
            }
            ContinuousBatchingImpl::_pull_awaiting_requests();
        }

    private:
        std::exception_ptr m_step_failure;
        std::map<uint64_t, uint64_t> m_sequence_ids;
    };
};

class CBNotificationOrderingTest : public testing::Test, public ContinuousBatchingPipeline {
protected:
    class PipelineTestInstance : public ContinuousBatchingPipeline::ContinuousBatchingImpl {
    public:
        explicit PipelineTestInstance(size_t max_num_batched_tokens = 8) {
            SchedulerConfig scheduler_config;
            scheduler_config.num_kv_blocks = 8;
            scheduler_config.max_num_batched_tokens = max_num_batched_tokens;
            scheduler_config.max_num_seqs = 1;
            m_scheduler = std::make_shared<ContinuousBatchingScheduler>(init_cache_orchestrator(scheduler_config, 4, 1), scheduler_config);
            m_sampler = std::make_shared<Sampler>();

            ov::ParameterVector parameters;
            const auto add_input = [&parameters](const std::string& name,
                                                 const ov::element::Type& type,
                                                 const ov::PartialShape& shape) {
                auto parameter = std::make_shared<ov::op::v0::Parameter>(type, shape);
                parameter->output(0).get_tensor().set_names({name});
                parameters.push_back(parameter);
            };
            add_input("input_ids", ov::element::i64, ov::PartialShape::dynamic(1));
            add_input("position_ids", ov::element::i64, ov::PartialShape::dynamic(1));
            add_input("past_lens", ov::element::i32, ov::PartialShape::dynamic(1));
            add_input("subsequence_begins", ov::element::i32, ov::PartialShape::dynamic(1));
            add_input("block_indices", ov::element::i32, ov::PartialShape::dynamic(1));
            add_input("block_indices_begins", ov::element::i32, ov::PartialShape::dynamic(1));
            add_input("max_context_len", ov::element::i32, ov::PartialShape{});
            const auto logits = ov::op::v0::Constant::create(ov::element::f32,
                                                             ov::Shape{1, 1, 8},
                                                             std::vector<float>{0, 0, 0, 0, 0, 0, 0, 1});
            logits->output(0).get_tensor().set_names({"logits"});
            const auto model = std::make_shared<ov::Model>(ov::OutputVector{logits}, parameters);
            ov::InferRequest request = ov::Core().compile_model(model, "CPU").create_infer_request();
            m_model_runner = std::make_shared<ModelRunner>(request, 4, 1);
        }

        ~PipelineTestInstance() override {
            drop_requests();
        }

        GenerationHandle add_test_request(uint64_t request_id,
                                          const std::vector<int64_t>& prompt,
                                          GenerationConfig config) {
            ov::Tensor input_ids(ov::element::i64, ov::Shape{prompt.size()});
            std::copy(prompt.begin(), prompt.end(), input_ids.data<int64_t>());
            const auto request = std::make_shared<SequenceGroup>(request_id, input_ids, config);
            m_requests.push_back(request);
            m_observed_request = request;
            return std::make_shared<GenerationHandleImpl>(request->get_generation_stream(), config);
        }

        void fail_at_candidate_commit() {
            m_candidate_failure = std::make_exception_ptr(std::runtime_error("injected candidate commit failure"));
        }

        bool output_was_visible_at_candidate_commit() const {
            return m_output_was_visible_at_candidate_commit;
        }

        size_t processed_tokens_at_candidate_commit() const {
            return m_processed_tokens_at_candidate_commit;
        }

        void discard_appended_candidates() {
            m_observed_request->get_sequences().front()->remove_last_tokens(2);
        }

        const std::exception_ptr& get_candidate_failure() const {
            return m_candidate_failure;
        }

    protected:
        void generate_candidates_for_prompt_lookup() override {
            m_output_was_visible_at_candidate_commit = m_observed_request->get_generation_stream()->can_read();
            m_processed_tokens_at_candidate_commit = m_observed_request->get_num_processed_tokens();
            if (m_candidate_failure) {
                std::rethrow_exception(m_candidate_failure);
            }
            for (const Sequence::Ptr& sequence : m_observed_request->get_running_sequences()) {
                sequence->append_token(6, 0.0f);
                sequence->append_token(-1, 0.0f);
            }
        }

    private:
        SequenceGroup::Ptr m_observed_request;
        std::exception_ptr m_candidate_failure;
        bool m_output_was_visible_at_candidate_commit = false;
        size_t m_processed_tokens_at_candidate_commit = 0;
    };
};

class CBLinearAttentionFailureGateTest : public testing::Test, public ContinuousBatchingPipeline {
protected:
    class PipelineTestInstance : public ContinuousBatchingPipeline::ContinuousBatchingImpl {
    public:
        PipelineTestInstance() {
            SchedulerConfig scheduler_config;
            scheduler_config.num_kv_blocks = 16;
            scheduler_config.num_linear_attention_blocks = 6;
            scheduler_config.max_num_batched_tokens = 16;
            scheduler_config.max_num_seqs = 1;
            scheduler_config.dynamic_split_fuse = false;

            m_orchestrator = init_hybrid_cache_orchestrator(scheduler_config, 4, 1, 1, true);
            m_scheduler = std::make_shared<ContinuousBatchingScheduler>(m_orchestrator, scheduler_config);
            m_sampler = std::make_shared<Sampler>();
            m_is_validation_mode_enabled = true;

            ov::Core core;
            const std::shared_ptr<ov::Model> cache_model = get_dummy_hybrid_model(core, 1, 1);
            ov::ParameterVector parameters = cache_model->get_parameters();
            const auto add_input = [&parameters](const std::string& name,
                                                 const ov::element::Type& type,
                                                 const ov::PartialShape& shape) {
                auto parameter = std::make_shared<ov::op::v0::Parameter>(type, shape);
                parameter->output(0).get_tensor().set_names({name});
                parameters.push_back(parameter);
            };
            add_input("input_ids", ov::element::i64, ov::PartialShape::dynamic(1));
            add_input("position_ids", ov::element::i64, ov::PartialShape::dynamic(1));
            add_input("past_lens", ov::element::i32, ov::PartialShape::dynamic(1));
            add_input("subsequence_begins", ov::element::i32, ov::PartialShape::dynamic(1));
            add_input("block_indices", ov::element::i32, ov::PartialShape::dynamic(1));
            add_input("block_indices_begins", ov::element::i32, ov::PartialShape::dynamic(1));
            add_input("max_context_len", ov::element::i32, ov::PartialShape{});
            add_input("linear_attention_block_indices", ov::element::i32, ov::PartialShape::dynamic(1));
            add_input("linear_attention_block_indices_begins", ov::element::i32, ov::PartialShape::dynamic(1));
            add_input("linear_attention_past_lens", ov::element::i32, ov::PartialShape::dynamic(1));
            add_input("linear_attention_cache_interval", ov::element::i32, ov::PartialShape::dynamic(1));
            const auto logits = ov::op::v0::Constant::create(ov::element::f32,
                                                             ov::Shape{3, 1, 8},
                                                             std::vector<float>{0, 0, 0, 0, 0, 0, 0, 1,
                                                                                0, 0, 0, 0, 0, 0, 0, 1,
                                                                                0, 0, 0, 0, 0, 0, 0, 1});
            logits->output(0).get_tensor().set_names({"logits"});
            ov::OutputVector outputs = cache_model->outputs();
            for (size_t output_index = 0; output_index < outputs.size(); ++output_index) {
                if (outputs[output_index].get_names().empty()) {
                    outputs[output_index].get_tensor().set_names({"cache_output." + std::to_string(output_index)});
                }
            }
            outputs.push_back(logits);
            const auto model = std::make_shared<ov::Model>(outputs, parameters);
            ov::InferRequest request = core.compile_model(model, "CPU").create_infer_request();
            m_model_runner = std::make_shared<ModelRunner>(request, 4, 1);
        }

        ~PipelineTestInstance() override {
            drop_requests();
        }

        GenerationHandle add_active_request(uint64_t request_id) {
            GenerationConfig config = utils::get_greedy_config();
            config.max_new_tokens = 8;
            config.num_assistant_tokens = 2;
            const std::vector<int64_t> prompt{1, 2, 3, 4};
            auto request = std::make_shared<SequenceGroup>(request_id, prompt, config);
            m_requests.push_back(request);
            m_sampler->create_logit_processor(request_id, config, request->get_prompt_ids());
            m_sequence_id = request->get_sequences().front()->get_id();
            return std::make_shared<GenerationHandleImpl>(request->get_generation_stream(), config);
        }

        GenerationHandle add_awaiting_request(uint64_t request_id) {
            GenerationConfig config = utils::get_greedy_config();
            config.max_new_tokens = 8;
            auto request = std::make_shared<SequenceGroup>(request_id, std::vector<int64_t>{5}, config);
            m_awaiting_requests.push_back(request);
            m_sampler->create_logit_processor(request_id, config, request->get_prompt_ids());
            return std::make_shared<GenerationHandleImpl>(request->get_generation_stream(), config);
        }

        void prepare_validation_candidates() {
            SequenceGroup::Ptr request = m_requests.front();
            request->get_running_sequences().front()->append_token(7, 0.0f);
            request->get_running_sequences().front()->append_token(7, 0.0f);
            request->set_num_validated_tokens(2);
            m_corrupt_next_transaction = true;
        }

        bool sampler_acceptance_observed() const {
            return m_sampler_acceptance_observed;
        }

        const std::exception_ptr& transaction_failure() const {
            return m_transaction_failure;
        }

        bool has_temporary_rows() const {
            return m_orchestrator->get_block_manager(CacheType::LINEAR_ATTENTION_CACHE)
                .has_temporary_blocks(m_sequence_id);
        }

        bool all_physical_blocks_released() const {
            for (const CacheType type : {CacheType::KV_CACHE, CacheType::LINEAR_ATTENTION_CACHE}) {
                const auto& block_manager = m_orchestrator->get_block_manager(type);
                if (block_manager.num_free_blocks() != block_manager.get_total_block_count()) {
                    return false;
                }
            }
            return true;
        }

    protected:
        void _commit_linear_attention_checkpoint_transactions(ContinuousBatchingScheduler::Output& scheduler_output,
                                                              const SamplerOutput& sampler_output) override {
            if (!m_corrupt_next_transaction) {
                ContinuousBatchingImpl::_commit_linear_attention_checkpoint_transactions(scheduler_output,
                                                                                          sampler_output);
                return;
            }
            const auto acceptance_it = sampler_output.acceptance_by_sequence.find(m_sequence_id);
            m_sampler_acceptance_observed = acceptance_it != sampler_output.acceptance_by_sequence.end();
            OPENVINO_ASSERT(m_sampler_acceptance_observed, "Expected real sampler acceptance before test corruption");
            OPENVINO_ASSERT(scheduler_output.has_linear_attention_paging_data(m_sequence_id),
                            "Expected a real linear-attention transition plan before test corruption");
            OPENVINO_ASSERT(scheduler_output.m_kv_paged_attention_data.erase(m_sequence_id) == 1,
                            "Expected a real KV transition plan before test corruption");
            try {
                ContinuousBatchingImpl::_commit_linear_attention_checkpoint_transactions(scheduler_output,
                                                                                          sampler_output);
            } catch (...) {
                m_transaction_failure = std::current_exception();
                throw;
            }
        }

    private:
        std::shared_ptr<CacheOrchestrator> m_orchestrator;
        uint64_t m_sequence_id = 0;
        bool m_corrupt_next_transaction = false;
        bool m_sampler_acceptance_observed = false;
        std::exception_ptr m_transaction_failure;
    };
};

TEST_F(CBNotificationOrderingTest, CandidateFailurePublishesNoGeneratedOutputAndPreservesException) {
    for (const size_t max_new_tokens : {0, 1, 2, 3}) {
        SCOPED_TRACE(max_new_tokens);
        PipelineTestInstance pipeline;
        GenerationConfig config = utils::get_greedy_config();
        config.max_new_tokens = max_new_tokens;
        config.echo = max_new_tokens == 0;
        if (max_new_tokens == 3) {
            config.stop_token_ids = {7};
        }
        GenerationHandle handle = pipeline.add_test_request(0, {1}, config);
        pipeline.fail_at_candidate_commit();

        std::exception_ptr step_failure;
        try {
            pipeline.step();
        } catch (...) {
            step_failure = std::current_exception();
        }

        expect_same_exception(step_failure, pipeline.get_candidate_failure());
        EXPECT_FALSE(pipeline.output_was_visible_at_candidate_commit());
        EXPECT_EQ(handle->get_status(), GenerationStatus::FAILED);
        std::exception_ptr reader_failure;
        try {
            handle->read();
        } catch (...) {
            reader_failure = std::current_exception();
        }
        expect_same_exception(reader_failure, pipeline.get_candidate_failure());
    }
}

TEST_F(CBLinearAttentionFailureGateTest,
       MissingKvTransitionPlanAfterRealSamplingFailsPipelineAndReleasesHybridCache) {
    PipelineTestInstance pipeline;
    GenerationHandle active_handle = pipeline.add_active_request(0);
    pipeline.step();
    ASSERT_TRUE(active_handle->can_read());
    std::ignore = active_handle->read();
    pipeline.prepare_validation_candidates();
    GenerationHandle awaiting_handle = pipeline.add_awaiting_request(1);

    std::exception_ptr step_failure;
    try {
        pipeline.step();
    } catch (...) {
        step_failure = std::current_exception();
    }

    ASSERT_NE(step_failure, nullptr);
    expect_same_exception(step_failure, pipeline.transaction_failure());
    EXPECT_TRUE(pipeline.sampler_acceptance_observed());
    EXPECT_TRUE(active_handle->can_read());
    EXPECT_TRUE(awaiting_handle->can_read());
    EXPECT_EQ(active_handle->get_status(), GenerationStatus::FAILED);
    EXPECT_EQ(awaiting_handle->get_status(), GenerationStatus::FAILED);

    const auto expect_original_failure = [&pipeline](GenerationHandle& handle, bool read_all) {
        std::exception_ptr reader_failure;
        try {
            if (read_all) {
                std::ignore = handle->read_all();
            } else {
                std::ignore = handle->read();
            }
        } catch (...) {
            reader_failure = std::current_exception();
        }
        expect_same_exception(reader_failure, pipeline.transaction_failure());
    };
    expect_original_failure(active_handle, false);
    expect_original_failure(awaiting_handle, true);
    EXPECT_FALSE(pipeline.has_temporary_rows());
    EXPECT_TRUE(pipeline.all_physical_blocks_released());

    std::exception_ptr reuse_failure;
    try {
        pipeline.step();
    } catch (...) {
        reuse_failure = std::current_exception();
    }
    expect_same_exception(reuse_failure, pipeline.transaction_failure());
}

TEST_F(CBNotificationOrderingTest, SuccessfulStepPublishesGeneratedAndTerminalOutputAfterCandidateCommit) {
    PipelineTestInstance streaming_pipeline;
    GenerationConfig streaming_config = utils::get_greedy_config();
    streaming_config.max_new_tokens = 3;
    GenerationHandle streaming_handle = streaming_pipeline.add_test_request(0, {1}, streaming_config);

    streaming_pipeline.step();

    EXPECT_FALSE(streaming_pipeline.output_was_visible_at_candidate_commit());
    ASSERT_TRUE(streaming_handle->can_read());
    EXPECT_EQ(streaming_handle->read().at(0).generated_ids, TokenIds({7}));
    EXPECT_EQ(streaming_handle->get_status(), GenerationStatus::RUNNING);

    streaming_pipeline.discard_appended_candidates();
    streaming_pipeline.step();

    ASSERT_TRUE(streaming_handle->can_read());
    EXPECT_EQ(streaming_handle->read().at(0).generated_ids, TokenIds({7}));
    EXPECT_EQ(streaming_handle->get_status(), GenerationStatus::RUNNING);

    PipelineTestInstance terminal_pipeline;
    GenerationConfig terminal_config = utils::get_greedy_config();
    terminal_config.max_new_tokens = 1;
    GenerationHandle terminal_handle = terminal_pipeline.add_test_request(1, {1}, terminal_config);

    terminal_pipeline.step();

    EXPECT_FALSE(terminal_pipeline.output_was_visible_at_candidate_commit());
    ASSERT_TRUE(terminal_handle->can_read());
    EXPECT_EQ(terminal_handle->read().at(0).generated_ids, TokenIds({7}));
    EXPECT_EQ(terminal_handle->get_status(), GenerationStatus::FINISHED);
}

TEST_F(CBNotificationOrderingTest, ChunkedEchoUsesRangeCapturedBeforeProcessedCounterUpdate) {
    PipelineTestInstance pipeline(1);
    GenerationConfig config = utils::get_greedy_config();
    config.echo = true;
    config.max_new_tokens = 0;
    GenerationHandle handle = pipeline.add_test_request(0, {4, 5}, config);

    pipeline.step();

    EXPECT_FALSE(pipeline.output_was_visible_at_candidate_commit());
    EXPECT_EQ(pipeline.processed_tokens_at_candidate_commit(), 1);
    ASSERT_TRUE(handle->can_read());
    EXPECT_EQ(handle->read().at(0).generated_ids, TokenIds({4}));
    EXPECT_EQ(handle->get_status(), GenerationStatus::RUNNING);

    pipeline.step();

    EXPECT_FALSE(pipeline.output_was_visible_at_candidate_commit());
    EXPECT_EQ(pipeline.processed_tokens_at_candidate_commit(), 2);
    ASSERT_TRUE(handle->can_read());
    EXPECT_EQ(handle->read().at(0).generated_ids, TokenIds({5}));
    EXPECT_EQ(handle->get_status(), GenerationStatus::FINISHED);
}

TEST_F(CBFailureBoundaryTest, StepFailureFailsActiveAndAwaitingRequestsAndRejectsReuse) {
    PipelineTestInstance pipeline;
    GenerationHandle active_handle = pipeline.add_test_request(0, true);
    GenerationHandle awaiting_handle = pipeline.add_test_request(1, false);
    std::promise<void> active_reader_started_promise;
    std::future<void> active_reader_started = active_reader_started_promise.get_future();
    auto active_reader = std::async(std::launch::async, [&active_handle, &active_reader_started_promise] {
        active_reader_started_promise.set_value();
        return active_handle->read();
    });
    std::promise<void> awaiting_reader_started_promise;
    std::future<void> awaiting_reader_started = awaiting_reader_started_promise.get_future();
    auto awaiting_reader = std::async(std::launch::async, [&awaiting_handle, &awaiting_reader_started_promise] {
        awaiting_reader_started_promise.set_value();
        return awaiting_handle->read_all();
    });
    const std::vector<GenerationHandle> handles{active_handle, awaiting_handle};
    HandleReleaseGuard handle_release_guard(handles);

    using namespace std::chrono_literals;
    ASSERT_EQ(active_reader_started.wait_for(1s), std::future_status::ready);
    ASSERT_EQ(awaiting_reader_started.wait_for(1s), std::future_status::ready);
    ASSERT_EQ(active_reader.wait_for(100ms), std::future_status::timeout);
    ASSERT_EQ(awaiting_reader.wait_for(100ms), std::future_status::timeout);
    ASSERT_TRUE(pipeline.has_block_table(0));
    ASSERT_TRUE(pipeline.has_block_table(1));
    ASSERT_TRUE(pipeline.has_sampler_context(0));
    ASSERT_TRUE(pipeline.has_sampler_context(1));
    pipeline.fail_next_step();

    std::exception_ptr step_failure;
    try {
        pipeline.step();
    } catch (...) {
        step_failure = std::current_exception();
    }

    expect_same_exception(step_failure, pipeline.get_step_failure());
    EXPECT_EQ(active_handle->get_status(), GenerationStatus::FAILED);
    EXPECT_EQ(awaiting_handle->get_status(), GenerationStatus::FAILED);
    ASSERT_EQ(active_reader.wait_for(1s), std::future_status::ready);
    ASSERT_EQ(awaiting_reader.wait_for(1s), std::future_status::ready);
    const auto expect_reader_failure = [&pipeline](auto& reader) {
        std::exception_ptr reader_failure;
        try {
            reader.get();
        } catch (...) {
            reader_failure = std::current_exception();
        }
        expect_same_exception(reader_failure, pipeline.get_step_failure());
    };
    expect_reader_failure(active_reader);
    expect_reader_failure(awaiting_reader);
    EXPECT_TRUE(pipeline.is_clean());
    EXPECT_FALSE(pipeline.has_block_table(0));
    EXPECT_FALSE(pipeline.has_block_table(1));
    EXPECT_FALSE(pipeline.has_sampler_context(0));
    EXPECT_FALSE(pipeline.has_sampler_context(1));

    std::exception_ptr reuse_failure;
    try {
        pipeline.step();
    } catch (...) {
        reuse_failure = std::current_exception();
    }
    expect_same_exception(reuse_failure, pipeline.get_step_failure());
}

TEST_F(CBFailureBoundaryTest, FailedPipelineRejectsAdmissionWithOriginalFailure) {
    PipelineTestInstance pipeline;
    pipeline.add_test_request(0, true);
    pipeline.fail_next_step();
    EXPECT_THROW(pipeline.step(), std::runtime_error);

    const auto expect_original_failure = [&pipeline](const auto& operation) {
        std::exception_ptr failure;
        try {
            operation();
        } catch (...) {
            failure = std::current_exception();
        }
        expect_same_exception(failure, pipeline.get_step_failure());
    };

    expect_original_failure([&pipeline] { pipeline.add_request(1, ov::Tensor{}, GenerationConfig{}); });
    expect_original_failure([&pipeline] { pipeline.add_request(1, std::string{"prompt"}, GenerationConfig{}); });
    expect_original_failure([&pipeline] { pipeline.step(); });
    expect_original_failure([&pipeline] { pipeline.has_non_finished_requests(); });
    expect_original_failure([&pipeline] { pipeline.get_awaiting_requests(); });
    expect_original_failure([&pipeline] {
        pipeline.generate(std::vector<ov::Tensor>{ov::Tensor{}},
                          std::vector<GenerationConfig>{GenerationConfig{}},
                          StreamerVariant{});
    });
}
