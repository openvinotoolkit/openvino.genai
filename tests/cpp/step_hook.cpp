// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "continuous_batching/cache/cache_orchestrator.hpp"
#include "continuous_batching/pipeline_impl.hpp"
#include "continuous_batching/scheduler.hpp"
#include "continuous_batching/step_hook.hpp"
#include "helper.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "scheduler_test_utils.hpp"
#include "sequence_group.hpp"
#include "utils.hpp"

using namespace ov::genai;

namespace {

// Greedy picks 7 and 6 is the unique second best, so no test depends on tie-breaking.
const std::vector<float> LOGITS_ROW{0, 0, 0, 0, 0, 0, 0.5f, 1};

// 0 is never the argmax, so only max_new_tokens or the hook ends a request.
GenerationConfig make_config(size_t max_new_tokens) {
    auto config = utils::get_greedy_config();
    config.max_new_tokens = max_new_tokens;
    config.eos_token_id = 0;
    config.stop_token_ids = {0};
    return config;
}

ov::Tensor token_tensor(const std::vector<int64_t>& tokens, const ov::Shape& shape) {
    ov::Tensor tensor(ov::element::i64, shape);
    std::copy(tokens.begin(), tokens.end(), tensor.data<int64_t>());
    return tensor;
}

float last_hidden_value(const Sequence::Ptr& sequence) {
    const auto hidden = sequence->get_hidden_state();
    return hidden.data<const float>()[hidden.get_size() - 1];
}

std::exception_ptr capture_failure(const std::function<void()>& operation) {
    try {
        operation();
    } catch (...) {
        return std::current_exception();
    }
    return nullptr;
}

// A loose substring of the plan's assert message, so a test fails if some other check fired first.
void expect_ov_failure_mentioning(const std::exception_ptr& failure, const std::string& message_part) {
    ASSERT_NE(failure, nullptr);
    try {
        std::rethrow_exception(failure);
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string{error.what()}.find(message_part), std::string::npos) << error.what();
    } catch (...) {
        FAIL() << "Expected an ov::Exception";
    }
}

class CountedState : public StepRequestState {
public:
    CountedState(int tag, std::shared_ptr<size_t> destroyed) : tag{tag}, destroyed{std::move(destroyed)} {}
    ~CountedState() override {
        ++*destroyed;
    }

    const int tag;
    const std::shared_ptr<size_t> destroyed;
};

class FakeStepHook : public IStepHook {
public:
    struct Payload : StepRequestPayload {
        explicit Payload(int tag) : tag{tag} {}
        const int tag;
    };

    struct RemoveCall {
        int tag;
        GenerationStatus status;
    };

    using Requests = std::vector<SequenceGroup::Ptr>;
    using Outcomes = std::function<std::vector<RequestOutcome>(const Requests&)>;

    std::unique_ptr<StepRequestState> before_add_request(
        ov::Tensor&,
        std::optional<ov::Tensor>&,
        SamplingOverrides& sampling_overrides,
        const std::shared_ptr<StepRequestPayload>& step_payload) override {
        created_payloads.push_back(step_payload);
        const auto payload = std::dynamic_pointer_cast<Payload>(step_payload);
        if (decline || !payload) {
            return nullptr;
        }
        sampling_overrides = overrides;
        return std::make_unique<CountedState>(payload->tag, destroyed);
    }

    std::vector<RequestOutcome> after_sampling(const Requests& requests) override {
        after_sampling_calls.push_back(request_ids(requests));
        return outcomes ? outcomes(requests) : std::vector<RequestOutcome>(requests.size());
    }

    void after_remove_request(StepRequestState& state, GenerationStatus status) override {
        const auto tag = static_cast<CountedState&>(state).tag;
        remove_calls.push_back({tag, status});
        if (status != failing_status) {
            return;
        }
        if (const auto failure = remove_failures_by_tag.find(tag); failure != remove_failures_by_tag.end()) {
            std::rethrow_exception(failure->second);
        }
        if (remove_failure) {
            std::rethrow_exception(remove_failure);
        }
    }

    size_t remove_count(int tag) const {
        return std::count_if(remove_calls.begin(), remove_calls.end(), [tag](const RemoveCall& call) {
            return call.tag == tag;
        });
    }

    bool decline = false;
    SamplingOverrides overrides;
    Outcomes outcomes;
    GenerationStatus failing_status = GenerationStatus::RUNNING;
    std::exception_ptr remove_failure;
    std::map<int, std::exception_ptr> remove_failures_by_tag;

    const std::shared_ptr<size_t> destroyed = std::make_shared<size_t>(0);
    std::vector<std::shared_ptr<StepRequestPayload>> created_payloads;
    std::vector<std::vector<uint64_t>> after_sampling_calls;
    std::vector<RemoveCall> remove_calls;

private:
    static std::vector<uint64_t> request_ids(const Requests& requests) {
        std::vector<uint64_t> ids;
        for (const auto& request : requests) {
            ids.push_back(request->get_request_id());
        }
        return ids;
    }
};

std::shared_ptr<StepRequestPayload> payload(int tag) {
    return std::make_shared<FakeStepHook::Payload>(tag);
}

IStepHook::RequestOutcome ending_outcome(GenerationStatus status, GenerationFinishReason reason) {
    IStepHook::RequestOutcome outcome;
    outcome.ending = IStepHook::Ending{status, reason};
    return outcome;
}

FakeStepHook::Outcomes every_request(const IStepHook::RequestOutcome& outcome) {
    return [outcome](const FakeStepHook::Requests& requests) {
        return std::vector<IStepHook::RequestOutcome>(requests.size(), outcome);
    };
}

FakeStepHook::Outcomes fail_request(uint64_t request_id, const std::exception_ptr& error) {
    return [request_id, error](const FakeStepHook::Requests& requests) {
        std::vector<IStepHook::RequestOutcome> outcomes(requests.size());
        for (size_t i = 0; i < requests.size(); ++i) {
            if (requests[i]->get_request_id() == request_id) {
                outcomes[i].error = error;
            }
        }
        return outcomes;
    };
}

std::exception_ptr request_failure() {
    return std::make_exception_ptr(std::runtime_error("request failure"));
}

class CountingStreamer : public StreamerBase {
public:
    using StreamerBase::write;

    StreamingStatus write(int64_t) override {
        ++writes;
        return StreamingStatus::RUNNING;
    }

    void end() override {
        ++ends;
    }

    std::atomic<size_t> writes{0};
    std::atomic<size_t> ends{0};
};

}  // namespace

class CBStepHookTest : public testing::Test, public ContinuousBatchingPipeline {
protected:
    class PipelineTestInstance : public ContinuousBatchingPipeline::ContinuousBatchingImpl {
    public:
        explicit PipelineTestInstance(const std::shared_ptr<IStepHook>& step_hook,
                                      size_t max_num_batched_tokens = 16,
                                      bool enable_prefix_caching = false) {
            SchedulerConfig scheduler_config;
            scheduler_config.num_kv_blocks = 16;
            scheduler_config.max_num_batched_tokens = max_num_batched_tokens;
            scheduler_config.max_num_seqs = 4;
            scheduler_config.enable_prefix_caching = enable_prefix_caching;
            m_scheduler = std::make_shared<ContinuousBatchingScheduler>(init_cache_orchestrator(scheduler_config, 4, 1),
                                                                        scheduler_config);
            m_sampler = std::make_shared<Sampler>();
            const auto request = ov::Core().compile_model(make_model(), "CPU").create_infer_request();
            m_model_runner = std::make_shared<ModelRunner>(request, 4, 1);
            set_step_hook(step_hook);
        }

        ~PipelineTestInstance() override {
            drop_requests();
        }

        using ContinuousBatchingImpl::_apply_model_ending;
        using ContinuousBatchingImpl::drop_requests;

        GenerationHandle add(uint64_t request_id,
                             const std::vector<int64_t>& prompt,
                             const GenerationConfig& config,
                             std::shared_ptr<StepRequestPayload> step_payload = nullptr) {
            // A dropped handle stops its request, so the fixture keeps every handle alive.
            return m_handles.emplace_back(_add_request(request_id,
                                                       token_tensor(prompt, {prompt.size()}),
                                                       config,
                                                       std::nullopt,
                                                       std::nullopt,
                                                       std::move(step_payload)));
        }

        const std::vector<SequenceGroup::Ptr>& requests() const {
            return m_requests;
        }

        const std::vector<SequenceGroup::Ptr>& awaiting() const {
            return m_awaiting_requests;
        }

        void free_non_running() {
            _free_non_running_requests();
        }

        // Mirrors the full branch of the scheduler's preemption by recompute, which a test cannot trigger on demand.
        void preempt(const SequenceGroup::Ptr& group) {
            for (const auto& sequence : group->get_not_finished_sequences()) {
                m_scheduler->free_sequence(sequence->get_id());
            }
            // The scheduler clears the waiting flag within the same schedule() call, so it is not set here.
            group->preempt_tokens(group->get_num_processed_tokens());
        }

        bool has_block_table(uint64_t sequence_id) const {
            return m_scheduler->has_block_table(sequence_id);
        }

        Sampler& sampler() {
            return *m_sampler;
        }

        // Handles stay unnotified, so the stream shows only what a model ending applies.
        SequenceGroup::Ptr sampled_group(size_t max_new_tokens) {
            const auto group = std::make_shared<SequenceGroup>(0, TokenIds{1}, make_config(max_new_tokens));
            group->schedule_tokens(1);
            ov::Tensor logits(ov::element::f32, {1, 1, LOGITS_ROW.size()});
            std::copy(LOGITS_ROW.begin(), LOGITS_ROW.end(), logits.data<float>());
            std::ignore = m_sampler->sample({group}, logits, false, /*notify_handles=*/false);
            return group;
        }

        void fail_next_pull() {
            m_pull_failure = std::make_exception_ptr(std::runtime_error("injected pull failure"));
        }

        const std::exception_ptr& pull_failure() const {
            return m_pull_failure;
        }

    protected:
        void _pull_awaiting_requests() override {
            if (m_pull_failure) {
                std::rethrow_exception(m_pull_failure);
            }
            ContinuousBatchingImpl::_pull_awaiting_requests();
        }

    private:
        // The hidden row of a token is its id, so a test can tell which rows reached a sequence.
        static std::shared_ptr<ov::Model> make_model() {
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

            const auto ids = std::make_shared<ov::op::v0::Convert>(parameters[0], ov::element::f32);
            const auto column = std::make_shared<ov::op::v1::Reshape>(
                ids, ov::op::v0::Constant::create(ov::element::i64, {3}, {-1, 1, 1}), false);
            column->output(0).get_tensor().set_names({"hidden_states"});
            const auto zeros = ov::op::v0::Constant::create(ov::element::f32, {1, 1, LOGITS_ROW.size()}, {0.0f});
            const auto bias = ov::op::v0::Constant::create(ov::element::f32, {1, 1, LOGITS_ROW.size()}, LOGITS_ROW);
            const auto logits =
                std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(column, zeros), bias);
            logits->output(0).get_tensor().set_names({"logits"});
            return std::make_shared<ov::Model>(ov::OutputVector{logits, column}, parameters);
        }

        std::exception_ptr m_pull_failure;
        std::vector<GenerationHandle> m_handles;
    };

    // Destroyed after the pipeline a test builds, which drops its requests through this hook.
    const std::shared_ptr<FakeStepHook> hook = std::make_shared<FakeStepHook>();
};

TEST(SequenceGroupStepState, take_detaches_once) {
    const auto group = std::make_shared<SequenceGroup>(0, TokenIds{1}, utils::get_greedy_config());
    auto state = std::make_unique<CountedState>(1, std::make_shared<size_t>(0));
    const auto* raw_state = state.get();
    group->set_step_state(std::move(state));
    ASSERT_EQ(group->get_step_state(), raw_state);

    const auto taken = group->take_step_state();
    EXPECT_EQ(taken.get(), raw_state);
    EXPECT_FALSE(group->take_step_state());
    EXPECT_EQ(group->get_step_state(), nullptr);
}

TEST(SequenceGroupStepState, state_dies_with_group) {
    const auto destroyed = std::make_shared<size_t>(0);
    auto group = std::make_shared<SequenceGroup>(0, TokenIds{1}, utils::get_greedy_config());
    group->set_step_state(std::make_unique<CountedState>(1, destroyed));
    EXPECT_EQ(*destroyed, 0);

    group.reset();

    EXPECT_EQ(*destroyed, 1);
}

namespace {

struct UnsupportedMode {
    std::string name;
    std::function<void(GenerationConfig&)> apply;
};

}  // namespace

class CBStepHookAddRequestTest : public CBStepHookTest, public testing::WithParamInterface<UnsupportedMode> {};

TEST_F(CBStepHookAddRequestTest, payload_reaches_hook_and_state_attaches) {
    PipelineTestInstance pipeline(hook);
    const auto step_payload = payload(3);

    pipeline.add(0, {1}, make_config(4), step_payload);

    ASSERT_EQ(pipeline.awaiting().size(), 1);
    const auto* state = dynamic_cast<CountedState*>(pipeline.awaiting().front()->get_step_state());
    ASSERT_NE(state, nullptr);
    EXPECT_EQ(state->tag, 3);
    ASSERT_EQ(hook->created_payloads.size(), 1);
    EXPECT_EQ(hook->created_payloads.front(), step_payload);
}

TEST_F(CBStepHookAddRequestTest, request_without_payload_gets_no_state) {
    PipelineTestInstance pipeline(hook);
    auto beam_config = make_config(4);
    beam_config.num_beams = 2;

    // Mode rejection applies to hooked requests only, so a plain beam request stays allowed.
    EXPECT_NO_THROW(pipeline.add(0, {1}, beam_config));

    ASSERT_EQ(pipeline.awaiting().size(), 1);
    EXPECT_EQ(pipeline.awaiting().front()->get_step_state(), nullptr);
}

TEST_F(CBStepHookAddRequestTest, overrides_from_hook_reach_sampler) {
    PipelineTestInstance pipeline(hook);
    hook->overrides.suppressed_token_ids = std::set<int64_t>{7};
    pipeline.add(0, {1}, make_config(4), payload(1));

    pipeline.step();

    ASSERT_EQ(pipeline.requests().size(), 1);
    EXPECT_EQ(pipeline.requests().front()->get_sequences().front()->get_generated_ids(), TokenIds({6}));
}

TEST_P(CBStepHookAddRequestTest, rejects_unsupported_mode) {
    PipelineTestInstance pipeline(hook);
    auto config = make_config(4);
    GetParam().apply(config);

    EXPECT_THROW(pipeline.add(0, {1}, config, payload(1)), ov::Exception);

    EXPECT_TRUE(pipeline.awaiting().empty());
    EXPECT_EQ(*hook->destroyed, 1);
    EXPECT_TRUE(hook->remove_calls.empty());
}

INSTANTIATE_TEST_SUITE_P(StepHook,
                         CBStepHookAddRequestTest,
                         testing::Values(UnsupportedMode{"beam_search",
                                                         [](GenerationConfig& config) {
                                                             config.num_beams = 2;
                                                         }},
                                         UnsupportedMode{"parallel_sampling",
                                                         [](GenerationConfig& config) {
                                                             config.do_sample = true;
                                                             config.num_return_sequences = 2;
                                                         }},
                                         UnsupportedMode{"prompt_lookup",
                                                         [](GenerationConfig& config) {
                                                             config.max_ngram_size = 3;
                                                             config.num_assistant_tokens = 2;
                                                         }}),
                         [](const testing::TestParamInfo<UnsupportedMode>& info) {
                             return info.param.name;
                         });

TEST_F(CBStepHookAddRequestTest, rejects_payload_no_hook_took) {
    PipelineTestInstance pipeline(hook);
    hook->decline = true;

    EXPECT_THROW(pipeline.add(0, {1}, make_config(4), payload(1)), ov::Exception);

    EXPECT_TRUE(pipeline.awaiting().empty());
}

TEST_F(CBStepHookAddRequestTest, plain_request_keeps_no_hidden_state) {
    PipelineTestInstance pipeline(hook);
    pipeline.add(0, {1, 2, 3}, make_config(4));
    const auto sequence = pipeline.awaiting().front()->get_sequences().front();

    pipeline.step();

    EXPECT_FALSE(static_cast<bool>(sequence->get_hidden_state()));
}

TEST_F(CBStepHookAddRequestTest, hooked_request_keeps_hidden_state) {
    PipelineTestInstance pipeline(hook);
    pipeline.add(0, {1, 2, 5}, make_config(4), payload(1));
    const auto sequence = pipeline.awaiting().front()->get_sequences().front();

    pipeline.step();

    ASSERT_TRUE(static_cast<bool>(sequence->get_hidden_state()));
    EXPECT_EQ(last_hidden_value(sequence), 5.0f);
}

class CBStepHookRemoveRequestTest : public CBStepHookTest, public testing::WithParamInterface<GenerationStatus> {};

TEST_F(CBStepHookRemoveRequestTest, finished_request_ends_once) {
    PipelineTestInstance pipeline(hook);
    pipeline.add(0, {1}, make_config(1), payload(5));

    pipeline.step();

    ASSERT_EQ(hook->remove_calls.size(), 1);
    EXPECT_EQ(hook->remove_calls[0].tag, 5);
    EXPECT_EQ(hook->remove_calls[0].status, GenerationStatus::FINISHED);

    pipeline.step();

    EXPECT_EQ(hook->remove_calls.size(), 1);
    EXPECT_EQ(*hook->destroyed, 1);
}

TEST_P(CBStepHookRemoveRequestTest, handle_end_maps_to_status) {
    PipelineTestInstance pipeline(hook);
    const auto handle = pipeline.add(0, {1}, make_config(4), payload(1));
    if (GetParam() == GenerationStatus::STOP) {
        handle->stop();
    } else {
        handle->cancel();
    }

    pipeline.step();

    // A request whose handle ended before it samples reaches no hook call.
    EXPECT_TRUE(hook->after_sampling_calls.empty());
    ASSERT_EQ(hook->remove_calls.size(), 1);
    EXPECT_EQ(hook->remove_calls[0].status, GetParam());
}

INSTANTIATE_TEST_SUITE_P(StepHook,
                         CBStepHookRemoveRequestTest,
                         testing::Values(GenerationStatus::STOP, GenerationStatus::CANCEL),
                         [](const testing::TestParamInfo<GenerationStatus>& info) {
                             return info.param == GenerationStatus::STOP ? std::string{"stop"} : std::string{"cancel"};
                         });

TEST_F(CBStepHookRemoveRequestTest, drop_ends_running_and_awaiting_once) {
    PipelineTestInstance pipeline(hook);
    pipeline.add(0, {1}, make_config(4), payload(1));
    pipeline.step();
    pipeline.add(1, {2}, make_config(4), payload(2));
    ASSERT_EQ(pipeline.requests().size(), 1);
    ASSERT_EQ(pipeline.awaiting().size(), 1);

    pipeline.drop_requests();
    pipeline.drop_requests();

    ASSERT_EQ(hook->remove_calls.size(), 2);
    for (const auto& call : hook->remove_calls) {
        EXPECT_EQ(call.status, GenerationStatus::FAILED);
    }
    EXPECT_EQ(hook->remove_count(1), 1);
    EXPECT_EQ(hook->remove_count(2), 1);
}

TEST_F(CBStepHookRemoveRequestTest, failing_drop_ends_every_request_and_fails_pipeline) {
    PipelineTestInstance pipeline(hook);
    pipeline.add(0, {1}, make_config(4), payload(1));
    pipeline.add(1, {2}, make_config(4), payload(2));
    pipeline.step();
    ASSERT_EQ(pipeline.requests().size(), 2);
    ASSERT_NO_THROW(pipeline.sampler().get_logit_processor(0));
    ASSERT_NO_THROW(pipeline.sampler().get_logit_processor(1));
    const auto awaiting_handle = pipeline.add(2, {3}, make_config(4), payload(3));
    ASSERT_EQ(pipeline.awaiting().size(), 1);
    std::vector<uint64_t> sequence_ids;
    for (const auto* group : {&pipeline.requests(), &pipeline.awaiting()}) {
        for (const auto& request : *group) {
            for (const auto& sequence : request->get_sequences()) {
                sequence_ids.push_back(sequence->get_id());
            }
        }
    }
    // Set only now, so a failed precondition above cannot make the fixture destructor throw.
    hook->failing_status = GenerationStatus::FAILED;
    for (const int tag : {1, 2, 3}) {
        hook->remove_failures_by_tag[tag] =
            std::make_exception_ptr(std::runtime_error("remove failure " + std::to_string(tag)));
    }

    EXPECT_NO_THROW(pipeline.drop_requests());

    EXPECT_EQ(hook->remove_count(1), 1);
    EXPECT_EQ(hook->remove_count(2), 1);
    EXPECT_EQ(hook->remove_count(3), 1);
    EXPECT_TRUE(pipeline.requests().empty());
    EXPECT_TRUE(pipeline.awaiting().empty());
    for (const auto sequence_id : sequence_ids) {
        EXPECT_FALSE(pipeline.has_block_table(sequence_id)) << sequence_id;
    }
    // A sampler context left behind is reused by the next request with the same id.
    for (const uint64_t request_id : {0, 1}) {
        EXPECT_THROW(pipeline.sampler().get_logit_processor(request_id), ov::Exception) << request_id;
    }
    EXPECT_EQ(awaiting_handle->get_status(), GenerationStatus::FAILED);
    expect_same_exception(capture_failure([&pipeline] {
                              pipeline.step();
                          }),
                          hook->remove_failures_by_tag.at(1));
    // Keeps the fixture destructor from aborting the run if drop_requests throws again.
    hook->remove_failures_by_tag.clear();
}

TEST_F(CBStepHookRemoveRequestTest, dropped_awaiting_request_is_never_pulled) {
    PipelineTestInstance pipeline(hook);
    pipeline.add(0, {1}, make_config(4), payload(1));

    pipeline.drop_requests();
    pipeline.step();

    EXPECT_TRUE(pipeline.awaiting().empty());
    EXPECT_TRUE(pipeline.requests().empty());
    EXPECT_TRUE(hook->after_sampling_calls.empty());
    EXPECT_EQ(hook->remove_count(1), 1);
}

TEST_F(CBStepHookRemoveRequestTest, dropped_awaiting_request_fails_its_handle) {
    // Another caller may have queued it, so its handle must not wait for a step that never comes.
    PipelineTestInstance pipeline(hook);
    const auto handle = pipeline.add(0, {1}, make_config(4));

    pipeline.drop_requests();

    EXPECT_EQ(handle->get_status(), GenerationStatus::FAILED);
    EXPECT_THROW(handle->read_all(), std::runtime_error);
}

TEST_F(CBStepHookRemoveRequestTest, drop_frees_blocks_restored_for_awaiting_request) {
    PipelineTestInstance pipeline(hook, 16, /*enable_prefix_caching=*/true);
    const std::vector<int64_t> prompt{1, 2, 3, 4, 5, 6, 7, 8, 9};
    pipeline.add(0, prompt, make_config(1));
    pipeline.step();
    ASSERT_TRUE(pipeline.requests().empty());

    pipeline.add(1, prompt, make_config(1));
    ASSERT_EQ(pipeline.awaiting().size(), 1);
    const auto sequence_id = pipeline.awaiting().front()->get_sequences().front()->get_id();
    ASSERT_TRUE(pipeline.has_block_table(sequence_id));

    pipeline.drop_requests();

    EXPECT_FALSE(pipeline.has_block_table(sequence_id));
}

TEST_F(CBStepHookRemoveRequestTest, pipeline_failure_removes_request_as_failed) {
    PipelineTestInstance pipeline(hook);
    pipeline.add(0, {1}, make_config(4), payload(1));
    pipeline.fail_next_pull();

    EXPECT_THROW(pipeline.step(), std::runtime_error);

    ASSERT_EQ(hook->remove_calls.size(), 1);
    EXPECT_EQ(hook->remove_calls[0].status, GenerationStatus::FAILED);
}

TEST_F(CBStepHookRemoveRequestTest, discarded_request_ends_once) {
    PipelineTestInstance pipeline(hook);
    pipeline.add(0, {1}, make_config(4), payload(1));

    pipeline.discard_awaiting_request(0);

    EXPECT_TRUE(pipeline.awaiting().empty());
    ASSERT_EQ(hook->remove_calls.size(), 1);
    EXPECT_EQ(hook->remove_calls[0].status, GenerationStatus::FAILED);
}

TEST_F(CBStepHookRemoveRequestTest, out_of_memory_ends_with_ignored) {
    PipelineTestInstance pipeline(hook);
    const auto handle = pipeline.add(0, {1}, make_config(4), payload(4));
    pipeline.step();
    ASSERT_EQ(pipeline.requests().size(), 1);
    ASSERT_EQ(pipeline.requests().front()->get_sequences().front()->get_generated_ids(), TokenIds({7}));

    // An OOM group has no running sequence, so it counts as finished and out of memory at once.
    pipeline.requests().front()->set_out_of_memory();
    pipeline.free_non_running();

    ASSERT_EQ(hook->remove_calls.size(), 1);
    EXPECT_EQ(hook->remove_calls[0].tag, 4);
    EXPECT_EQ(hook->remove_calls[0].status, GenerationStatus::IGNORED);
    EXPECT_TRUE(pipeline.requests().empty());
    EXPECT_EQ(handle->get_status(), GenerationStatus::IGNORED);
}

namespace {

struct ExpectedStream {
    GenerationStatus status;
    GenerationFinishReason reason;
};

struct ExpectedSequence {
    bool finished;
    GenerationFinishReason reason;
    bool dropped;
};

struct EndingCase {
    std::string name;
    size_t max_new_tokens;
    IStepHook::Ending ending;
    ExpectedStream stream;
    ExpectedSequence sequence;
};

}  // namespace

class CBStepHookEndingTest : public CBStepHookTest, public testing::WithParamInterface<EndingCase> {};

TEST_P(CBStepHookEndingTest, applies_single_precedence_rule) {
    const auto& expected = GetParam();
    PipelineTestInstance pipeline(hook);
    const auto group = pipeline.sampled_group(expected.max_new_tokens);
    const auto sequence = group->get_sequences().front();
    // A fresh output holds only what the ending adds, not the sampler's own drops.
    SamplerOutput sampler_output;

    pipeline._apply_model_ending(group, expected.ending, sampler_output);

    const auto stream = group->get_generation_stream();
    EXPECT_EQ(stream->get_status(), expected.stream.status);
    EXPECT_EQ(stream->get_finish_reason(), expected.stream.reason);
    EXPECT_EQ(sequence->has_finished(), expected.sequence.finished);
    EXPECT_EQ(sequence->get_finish_reason(), expected.sequence.reason);
    const auto expected_dropped =
        expected.sequence.dropped ? std::vector<uint64_t>{sequence->get_id()} : std::vector<uint64_t>{};
    EXPECT_EQ(sampler_output.m_dropped_sequences, expected_dropped);
}

INSTANTIATE_TEST_SUITE_P(
    StepHook,
    CBStepHookEndingTest,
    testing::Values(EndingCase{"model_finishes_running_request",
                               10,
                               {GenerationStatus::FINISHED, GenerationFinishReason::STOP},
                               ExpectedStream{GenerationStatus::RUNNING, GenerationFinishReason::NONE},
                               ExpectedSequence{true, GenerationFinishReason::STOP, true}},
                    EndingCase{"model_reason_replaces_length",
                               1,
                               {GenerationStatus::FINISHED, GenerationFinishReason::STOP},
                               ExpectedStream{GenerationStatus::RUNNING, GenerationFinishReason::NONE},
                               ExpectedSequence{true, GenerationFinishReason::STOP, false}},
                    EndingCase{"stop_keeps_tool_call",
                               10,
                               {GenerationStatus::STOP, GenerationFinishReason::TOOL_CALL},
                               ExpectedStream{GenerationStatus::STOP, GenerationFinishReason::TOOL_CALL},
                               ExpectedSequence{false, GenerationFinishReason::NONE, false}},
                    EndingCase{"stop_on_last_token_marks_sequences",
                               1,
                               {GenerationStatus::STOP, GenerationFinishReason::NONE},
                               ExpectedStream{GenerationStatus::STOP, GenerationFinishReason::STOP},
                               ExpectedSequence{true, GenerationFinishReason::STOP, false}},
                    EndingCase{"stop_keeps_length",
                               10,
                               {GenerationStatus::STOP, GenerationFinishReason::LENGTH},
                               ExpectedStream{GenerationStatus::STOP, GenerationFinishReason::LENGTH},
                               ExpectedSequence{false, GenerationFinishReason::NONE, false}},
                    EndingCase{"cancel_sets_sequence_reason",
                               10,
                               {GenerationStatus::CANCEL, GenerationFinishReason::STOP},
                               ExpectedStream{GenerationStatus::CANCEL, GenerationFinishReason::NONE},
                               ExpectedSequence{false, GenerationFinishReason::STOP, false}}),
    [](const testing::TestParamInfo<EndingCase>& info) {
        return info.param.name;
    });

TEST_F(CBStepHookEndingTest, rejects_non_terminal_status) {
    PipelineTestInstance pipeline(hook);
    const auto group = pipeline.sampled_group(10);
    SamplerOutput sampler_output;

    const auto failure = capture_failure([&] {
        pipeline._apply_model_ending(group, {GenerationStatus::RUNNING, GenerationFinishReason::NONE}, sampler_output);
    });

    expect_ov_failure_mentioning(failure, "only as FINISHED, STOP or CANCEL");
}

TEST_F(CBStepHookEndingTest, model_finish_is_notified_in_same_step) {
    PipelineTestInstance pipeline(hook);
    hook->outcomes = every_request(ending_outcome(GenerationStatus::FINISHED, GenerationFinishReason::STOP));
    const auto handle = pipeline.add(0, {1}, make_config(10), payload(1));

    pipeline.step();

    EXPECT_EQ(handle->get_status(), GenerationStatus::FINISHED);
    const auto outputs = handle->read_all();
    ASSERT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].generated_ids, TokenIds({7}));
    EXPECT_EQ(outputs[0].finish_reason, GenerationFinishReason::STOP);
}

using CBStepHookStepTest = CBStepHookTest;

TEST_F(CBStepHookStepTest, chunked_prefill_calls_hook_only_when_sampling) {
    PipelineTestInstance pipeline(hook, 4);
    pipeline.add(0, {1, 2, 3, 4, 5, 6}, make_config(2), payload(1));

    pipeline.step();

    EXPECT_TRUE(hook->after_sampling_calls.empty());

    pipeline.step();

    EXPECT_EQ(hook->after_sampling_calls, (std::vector<std::vector<uint64_t>>{{0}}));
}

TEST_F(CBStepHookStepTest, plain_request_reaches_no_hook_call) {
    PipelineTestInstance pipeline(hook);
    pipeline.add(0, {1}, make_config(2), payload(1));
    pipeline.add(1, {2}, make_config(2));

    pipeline.step();
    pipeline.step();

    EXPECT_EQ(hook->after_sampling_calls, (std::vector<std::vector<uint64_t>>{{0}, {0}}));
}

TEST_F(CBStepHookStepTest, forced_token_from_hook_overrides_is_sampled) {
    PipelineTestInstance pipeline(hook);
    hook->overrides.forced_token = [] {
        return std::optional<int64_t>{3};
    };
    pipeline.add(0, {1}, make_config(4), payload(1));
    pipeline.add(1, {2}, make_config(4));

    pipeline.step();
    pipeline.step();

    ASSERT_EQ(pipeline.requests().size(), 2);
    EXPECT_EQ(pipeline.requests()[0]->get_sequences().front()->get_generated_ids(), TokenIds({3, 3}));
    // Only the request the hook took gets its overrides.
    EXPECT_NE(pipeline.requests()[1]->get_sequences().front()->get_generated_ids(), TokenIds({3, 3}));
}

TEST_F(CBStepHookStepTest, after_sampling_runs_once_for_all_requests) {
    PipelineTestInstance pipeline(hook);
    std::map<uint64_t, float> last_hidden_values;
    hook->outcomes = [&last_hidden_values](const FakeStepHook::Requests& requests) {
        for (const auto& request : requests) {
            last_hidden_values[request->get_request_id()] = last_hidden_value(request->get_sequences().front());
        }
        return std::vector<IStepHook::RequestOutcome>(requests.size());
    };
    pipeline.add(0, {1, 2, 5}, make_config(2), payload(1));
    pipeline.add(1, {4}, make_config(2), payload(2));

    pipeline.step();

    ASSERT_EQ(hook->after_sampling_calls.size(), 1);
    EXPECT_EQ(hook->after_sampling_calls[0].size(), 2);
    EXPECT_EQ(last_hidden_values, (std::map<uint64_t, float>{{0, 5.0f}, {1, 4.0f}}));
}

TEST_F(CBStepHookStepTest, replayed_tokens_reach_no_hook_call) {
    PipelineTestInstance pipeline(hook, 2);
    std::vector<size_t> generated_lens;
    hook->outcomes = [&generated_lens](const FakeStepHook::Requests& requests) {
        for (const auto& request : requests) {
            generated_lens.push_back(request->get_sequences().front()->get_generated_len());
        }
        return std::vector<IStepHook::RequestOutcome>(requests.size());
    };
    const auto handle = pipeline.add(0, {1, 2, 3}, make_config(4), payload(1));
    const auto sequence = pipeline.awaiting().front()->get_sequences().front();
    const auto requests_per_step = [&pipeline, this](size_t num_steps) {
        std::vector<size_t> counts;
        for (size_t step = 0; step < num_steps; ++step) {
            const auto calls_before = hook->after_sampling_calls.size();
            pipeline.step();
            counts.push_back(hook->after_sampling_calls.size() == calls_before
                                 ? 0
                                 : hook->after_sampling_calls.back().size());
        }
        return counts;
    };

    const std::vector<size_t> expected_requests_before_preemption{0, 1, 1};
    EXPECT_EQ(requests_per_step(expected_requests_before_preemption.size()), expected_requests_before_preemption);

    ASSERT_EQ(pipeline.requests().size(), 1);
    pipeline.preempt(pipeline.requests().front());
    // Replaying 1,2 then 3,7 stays within the context already sampled, so only the window past it samples.
    const std::vector<size_t> expected_requests_after_preemption{0, 0, 1, 1};
    EXPECT_EQ(requests_per_step(expected_requests_after_preemption.size()), expected_requests_after_preemption);

    EXPECT_EQ(generated_lens, (std::vector<size_t>{1, 2, 3, 4}));
    EXPECT_EQ(sequence->get_generated_ids(), TokenIds({7, 7, 7, 7}));
    EXPECT_EQ(sequence->get_finish_reason(), GenerationFinishReason::LENGTH);
    EXPECT_EQ(handle->get_status(), GenerationStatus::FINISHED);
    ASSERT_EQ(hook->remove_calls.size(), 1);
    EXPECT_EQ(hook->remove_calls[0].status, GenerationStatus::FINISHED);
}

namespace {

struct BadOutcomeCase {
    std::string name;
    std::function<void(FakeStepHook&)> script;
    std::string message_part;
};

}  // namespace

class CBStepHookContractTest : public CBStepHookTest, public testing::WithParamInterface<BadOutcomeCase> {};

TEST_P(CBStepHookContractTest, rejects_bad_outcomes) {
    PipelineTestInstance pipeline(hook);
    GetParam().script(*hook);
    const auto handle = pipeline.add(0, {1}, make_config(4), payload(1));

    expect_ov_failure_mentioning(capture_failure([&pipeline] {
                                     pipeline.step();
                                 }),
                                 GetParam().message_part);

    EXPECT_EQ(handle->get_status(), GenerationStatus::FAILED);
}

INSTANTIATE_TEST_SUITE_P(
    StepHook,
    CBStepHookContractTest,
    testing::Values(BadOutcomeCase{"no_outcome_for_request",
                                   [](FakeStepHook& fake) {
                                       fake.outcomes = [](const FakeStepHook::Requests&) {
                                           return std::vector<IStepHook::RequestOutcome>{};
                                       };
                                   },
                                   "one outcome per request"},
                    BadOutcomeCase{"ending_with_next_input",
                                   [](FakeStepHook& fake) {
                                       auto outcome = ending_outcome(GenerationStatus::FINISHED,
                                                                     GenerationFinishReason::STOP);
                                       outcome.next_embedding = ov::Tensor(ov::element::f32, {1, 1, 1});
                                       fake.outcomes = every_request(outcome);
                                   },
                                   "must not return a next input"},
                    BadOutcomeCase{"position_ids_without_next_input",
                                   [](FakeStepHook& fake) {
                                       IStepHook::RequestOutcome outcome;
                                       outcome.position_ids = token_tensor({1}, {1, 1});
                                       fake.outcomes = every_request(outcome);
                                   },
                                   "Position ids need a next input"}),
    [](const testing::TestParamInfo<BadOutcomeCase>& info) {
        return info.param.name;
    });

class CBStepHookFailureTest : public CBStepHookTest {
protected:
    // Fails the only request at its second sampling step, after one token was already streamed.
    void fail_second_sampling_step(const std::exception_ptr& error) {
        hook->outcomes = [this, error](const FakeStepHook::Requests& requests) {
            std::vector<IStepHook::RequestOutcome> outcomes(requests.size());
            if (hook->after_sampling_calls.size() == 2) {
                outcomes.front().error = error;
            }
            return outcomes;
        };
    }

    static std::vector<EncodedGenerationResult> run_generate(PipelineTestInstance& pipeline,
                                                              const StreamerVariant& streamer) {
        return pipeline.generate_with_omni_streamer({token_tensor({1}, {1, 1})},
                                                    {make_config(3)},
                                                    streamer,
                                                    nullptr,
                                                    std::nullopt,
                                                    std::nullopt,
                                                    std::nullopt,
                                                    {payload(1)});
    }
};

TEST_F(CBStepHookFailureTest, outcome_error_fails_only_its_request) {
    PipelineTestInstance pipeline(hook);
    const auto error = request_failure();
    hook->outcomes = fail_request(0, error);
    const auto failed_handle = pipeline.add(0, {1}, make_config(3), payload(1));
    const auto failed_sequence_id = pipeline.awaiting().front()->get_sequences().front()->get_id();
    const auto other_handle = pipeline.add(1, {2}, make_config(3), payload(2));

    EXPECT_NO_THROW(pipeline.step());

    EXPECT_EQ(failed_handle->get_status(), GenerationStatus::FAILED);
    expect_same_exception(capture_failure([&failed_handle] {
                              std::ignore = failed_handle->read_all();
                          }),
                          error);
    ASSERT_EQ(hook->remove_count(1), 1);
    EXPECT_EQ(hook->remove_calls[0].status, GenerationStatus::FAILED);
    EXPECT_FALSE(pipeline.has_block_table(failed_sequence_id));
    ASSERT_EQ(pipeline.requests().size(), 1);
    EXPECT_EQ(pipeline.requests().front()->get_request_id(), 1);

    pipeline.step();
    pipeline.step();

    EXPECT_EQ(other_handle->get_status(), GenerationStatus::FINISHED);
    EXPECT_EQ(hook->remove_count(1), 1);
}

TEST_F(CBStepHookFailureTest, pipeline_stays_usable_after_outcome_error) {
    PipelineTestInstance pipeline(hook);
    hook->outcomes = fail_request(0, request_failure());
    pipeline.add(0, {1}, make_config(3), payload(1));
    pipeline.step();
    hook->outcomes = nullptr;

    const size_t max_new_tokens = 3;
    const auto handle = pipeline.add(1, {2}, make_config(max_new_tokens), payload(2));
    for (size_t step = 0; step < max_new_tokens; ++step) {
        pipeline.step();
    }

    EXPECT_EQ(handle->get_status(), GenerationStatus::FINISHED);
    EXPECT_FALSE(pipeline.has_non_finished_requests());
}

TEST_F(CBStepHookFailureTest, generate_rethrows_outcome_error) {
    PipelineTestInstance pipeline(hook);
    const auto error = request_failure();
    fail_second_sampling_step(error);

    expect_same_exception(capture_failure([&pipeline] {
                              std::ignore = run_generate(pipeline, std::monostate{});
                          }),
                          error);
    ASSERT_EQ(hook->remove_calls.size(), 1);
    EXPECT_EQ(hook->remove_calls[0].status, GenerationStatus::FAILED);

    hook->outcomes = nullptr;
    const auto results = run_generate(pipeline, std::monostate{});
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].m_status, GenerationStatus::FINISHED);
}

TEST_F(CBStepHookFailureTest, generate_with_text_streamer_rethrows_outcome_error) {
    PipelineTestInstance pipeline(hook);
    const auto error = request_failure();
    fail_second_sampling_step(error);
    const auto streamer = std::make_shared<CountingStreamer>();

    expect_same_exception(capture_failure([&pipeline, &streamer] {
                              std::ignore = run_generate(pipeline, streamer);
                          }),
                          error);
    EXPECT_EQ(streamer->writes.load(), 1);
    EXPECT_EQ(streamer->ends.load(), 1);
    // drop_requests() in the generate loop's catch must not end the released request again.
    ASSERT_EQ(hook->remove_calls.size(), 1);
    EXPECT_EQ(hook->remove_calls[0].status, GenerationStatus::FAILED);

    hook->outcomes = nullptr;
    const auto next_streamer = std::make_shared<CountingStreamer>();
    const auto results = run_generate(pipeline, next_streamer);
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].m_status, GenerationStatus::FINISHED);
    EXPECT_EQ(next_streamer->ends.load(), 1);
}

TEST_F(CBStepHookFailureTest, remove_error_fails_stopped_request) {
    PipelineTestInstance pipeline(hook);
    const auto error = std::make_exception_ptr(std::runtime_error("after_remove_request failure"));
    hook->failing_status = GenerationStatus::STOP;
    hook->remove_failure = error;
    const auto handle = pipeline.add(0, {1}, make_config(4), payload(1));
    handle->stop();

    EXPECT_NO_THROW(pipeline.step());

    EXPECT_EQ(handle->get_status(), GenerationStatus::FAILED);
    expect_same_exception(capture_failure([&handle] {
                              std::ignore = handle->read_all();
                          }),
                          error);
}

TEST_F(CBStepHookFailureTest, remove_error_after_final_notification_fails_step) {
    PipelineTestInstance pipeline(hook);
    const auto error = std::make_exception_ptr(std::runtime_error("after_remove_request failure"));
    hook->failing_status = GenerationStatus::FINISHED;
    hook->remove_failure = error;
    pipeline.add(0, {1}, make_config(1), payload(1));

    // The handle already reported FINISHED, so only the step can carry the error.
    expect_same_exception(capture_failure([&pipeline] {
                              pipeline.step();
                          }),
                          error);
}

TEST_F(CBStepHookFailureTest, rejects_error_with_other_fields) {
    auto with_ending = ending_outcome(GenerationStatus::FINISHED, GenerationFinishReason::STOP);
    with_ending.error = request_failure();
    IStepHook::RequestOutcome with_next_input;
    with_next_input.next_embedding = ov::Tensor(ov::element::f32, {1, 1, 1});
    with_next_input.error = request_failure();

    for (const auto& outcome : {with_ending, with_next_input}) {
        SCOPED_TRACE(outcome.ending ? "error with ending" : "error with next input");
        PipelineTestInstance pipeline(hook);
        hook->outcomes = every_request(outcome);
        pipeline.add(0, {1}, make_config(4), payload(1));

        expect_ov_failure_mentioning(capture_failure([&pipeline] {
                                         pipeline.step();
                                     }),
                                     "must return nothing else");
    }
}

TEST_F(CBStepHookFailureTest, generate_rejects_payload_count_mismatch) {
    PipelineTestInstance pipeline(hook);

    expect_ov_failure_mentioning(capture_failure([&pipeline] {
                                     pipeline.generate_with_omni_streamer({token_tensor({1}, {1, 1})},
                                                                          {make_config(3)},
                                                                          std::monostate{},
                                                                          nullptr,
                                                                          std::nullopt,
                                                                          std::nullopt,
                                                                          std::nullopt,
                                                                          {payload(1), payload(2)});
                                 }),
                                 "one step payload per request");
    EXPECT_TRUE(hook->remove_calls.empty());
}
