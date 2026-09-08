// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include "sampling/sampler.hpp"
#include "openvino/genai/generation_config.hpp"
#include "utils.hpp"


using namespace ov::genai;

namespace {

using namespace std::chrono_literals;

class WorkerGate {
public:
    WorkerGate() : m_release_future(m_release_promise.get_future().share()) {}

    ~WorkerGate() {
        release();
    }

    std::shared_future<void> get_release_future() const {
        return m_release_future;
    }

    void release() {
        if (!m_released.exchange(true)) {
            m_release_promise.set_value();
        }
    }

private:
    std::promise<void> m_release_promise;
    std::shared_future<void> m_release_future;
    std::atomic<bool> m_released{false};
};

class WorkerGateReleaseGuard {
public:
    explicit WorkerGateReleaseGuard(WorkerGate& worker_gate) : m_worker_gate(worker_gate) {}

    ~WorkerGateReleaseGuard() {
        m_worker_gate.release();
    }

private:
    WorkerGate& m_worker_gate;
};

class BlockingLogitTransformer : public LogitTransformers::ILogitTransformer {
public:
    BlockingLogitTransformer(std::promise<void>& entered_promise, std::shared_future<void> release_future) :
        m_entered_promise(entered_promise),
        m_release_future(std::move(release_future)) {}

    void apply(Logits&) override {
        m_entered_promise.set_value();
        m_release_future.wait();
    }

private:
    std::promise<void>& m_entered_promise;
    std::shared_future<void> m_release_future;
};

class ThrowingLogitTransformer : public LogitTransformers::ILogitTransformer {
public:
    ThrowingLogitTransformer(std::shared_future<void> blocking_worker_entered,
                             std::promise<void>& throwing_worker_entered) :
        m_blocking_worker_entered(std::move(blocking_worker_entered)),
        m_throwing_worker_entered(throwing_worker_entered) {}

    void apply(Logits&) override {
        m_blocking_worker_entered.wait();
        m_throwing_worker_entered.set_value();
        OPENVINO_THROW("injected sampler worker failure");
    }

private:
    std::shared_future<void> m_blocking_worker_entered;
    std::promise<void>& m_throwing_worker_entered;
};

class TestLogitProcessor : public LogitProcessor {
public:
    using LogitProcessor::LogitProcessor;

    void set_transformer(std::shared_ptr<LogitTransformers::ILogitTransformer> transformer) {
        m_logit_transformers = {std::move(transformer)};
    }
};

void inject_transformer(Sampler& sampler,
                        uint64_t request_id,
                        const SequenceGroup::Ptr& sequence_group,
                        std::shared_ptr<LogitTransformers::ILogitTransformer> transformer) {
    const GenerationConfig& sampling_config = sequence_group->get_sampling_parameters();
    sampler.create_logit_processor(request_id, sampling_config, sequence_group->get_prompt_ids());
    TestLogitProcessor test_processor(sampling_config, sequence_group->get_prompt_ids());
    test_processor.set_transformer(std::move(transformer));
    sampler.get_logit_processor(request_id) = std::move(test_processor);
}

SequenceGroup::Ptr make_scheduled_group(uint64_t request_id, GenerationConfig sampling_config = ov::genai::utils::get_greedy_config()) {
    const std::vector<int64_t> prompt{0};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, prompt.size()}, prompt.data());
    auto sequence_group = std::make_shared<SequenceGroup>(request_id, input_tensor, sampling_config);
    sequence_group->schedule_tokens(sequence_group->get_num_available_tokens_for_batching());
    return sequence_group;
}

ov::Tensor make_logits(size_t num_tokens) {
    constexpr size_t vocab_size = 2;
    ov::Tensor logits(ov::element::f32, ov::Shape{num_tokens, 1, vocab_size});
    std::fill_n(logits.data<float>(), logits.get_size(), 0.0f);
    for (size_t token_id = 0; token_id < num_tokens; ++token_id) {
        logits.data<float>()[token_id * vocab_size + 1] = 1.0f;
    }
    return logits;
}

}  // namespace

TEST(SamplerWorkerJoiningTest, waits_for_blocked_worker_and_preserves_worker_exception) {
    SequenceGroup::Ptr failing_group = make_scheduled_group(0);
    SequenceGroup::Ptr blocked_group = make_scheduled_group(1);
    std::vector<SequenceGroup::Ptr> sequence_groups{failing_group, blocked_group};
    ov::Tensor logits = make_logits(2);
    WorkerGate worker_gate;
    std::promise<void> blocking_worker_entered_promise;
    std::shared_future<void> blocking_worker_entered = blocking_worker_entered_promise.get_future().share();
    std::promise<void> throwing_worker_entered_promise;
    std::future<void> throwing_worker_entered = throwing_worker_entered_promise.get_future();
    {
        Sampler sampler(2);
        inject_transformer(
            sampler,
            blocked_group->get_request_id(),
            blocked_group,
            std::make_shared<BlockingLogitTransformer>(blocking_worker_entered_promise,
                                                        worker_gate.get_release_future()));
        inject_transformer(
            sampler,
            failing_group->get_request_id(),
            failing_group,
            std::make_shared<ThrowingLogitTransformer>(blocking_worker_entered, throwing_worker_entered_promise));

        std::future<SamplerOutput> sampling_result = std::async(std::launch::async, [&] {
            return sampler.sample(sequence_groups, logits, false);
        });
        WorkerGateReleaseGuard worker_gate_release_guard(worker_gate);

        EXPECT_EQ(throwing_worker_entered.wait_for(5s), std::future_status::ready);
        EXPECT_EQ(sampling_result.wait_for(100ms), std::future_status::timeout);
        worker_gate.release();
        EXPECT_THROW(
            {
                try {
                    sampling_result.get();
                } catch (const ov::Exception& exception) {
                    EXPECT_NE(std::string(exception.what()).find("injected sampler worker failure"), std::string::npos);
                    throw;
                }
            },
            ov::Exception);

        sampler.clear_request_info(0);
        sampler.clear_request_info(1);
    }

    EXPECT_EQ(blocked_group->get_sequences().front()->get_generated_ids(), TokenIds({1}));
}

TEST(SamplerWorkerJoiningTest, waits_for_submitted_worker_on_submission_loop_failure) {
    SequenceGroup::Ptr submitted_group = make_scheduled_group(0);
    SequenceGroup::Ptr duplicate_group = make_scheduled_group(0);
    std::vector<SequenceGroup::Ptr> sequence_groups{submitted_group, duplicate_group};
    ov::Tensor logits = make_logits(2);
    WorkerGate worker_gate;
    std::promise<void> blocking_worker_entered_promise;
    std::future<void> blocking_worker_entered = blocking_worker_entered_promise.get_future();
    {
        Sampler sampler(1);
        inject_transformer(
            sampler,
            submitted_group->get_request_id(),
            submitted_group,
            std::make_shared<BlockingLogitTransformer>(blocking_worker_entered_promise,
                                                        worker_gate.get_release_future()));

        std::future<SamplerOutput> sampling_result = std::async(std::launch::async, [&] {
            return sampler.sample(sequence_groups, logits, false);
        });
        WorkerGateReleaseGuard worker_gate_release_guard(worker_gate);

        EXPECT_EQ(blocking_worker_entered.wait_for(5s), std::future_status::ready);
        EXPECT_EQ(sampling_result.wait_for(100ms), std::future_status::timeout);
        worker_gate.release();
        EXPECT_THROW(
            {
                try {
                    sampling_result.get();
                } catch (const std::exception& exception) {
                    EXPECT_NE(std::string(exception.what()).find("already submitted"), std::string::npos);
                    throw;
                }
            },
            std::exception);

        sampler.clear_request_info(0);
    }

    EXPECT_EQ(submitted_group->get_sequences().front()->get_generated_ids(), TokenIds({1}));
}

TEST(SamplerWorkerJoiningTest, preserves_successful_multi_request_results) {
    SequenceGroup::Ptr first_group = make_scheduled_group(0);
    SequenceGroup::Ptr second_group = make_scheduled_group(1);
    std::vector<SequenceGroup::Ptr> sequence_groups{first_group, second_group};
    ov::Tensor logits = make_logits(2);
    Sampler sampler(2);

    const SamplerOutput output = sampler.sample(sequence_groups, logits, false);

    EXPECT_EQ(output.num_generated_tokens, 2);
    EXPECT_EQ(output.num_generated_tokens_per_request.at(0), 1);
    EXPECT_EQ(output.num_generated_tokens_per_request.at(1), 1);
    EXPECT_EQ(first_group->get_sequences().front()->get_generated_ids(), TokenIds({1}));
    EXPECT_EQ(second_group->get_sequences().front()->get_generated_ids(), TokenIds({1}));
    EXPECT_TRUE(first_group->get_generation_stream()->can_read());
    EXPECT_TRUE(second_group->get_generation_stream()->can_read());
}

TEST(SamplerNotificationTest, defers_generated_output_until_explicit_notification) {
    SequenceGroup::Ptr sequence_group = make_scheduled_group(0);
    std::vector<SequenceGroup::Ptr> sequence_groups{sequence_group};
    Sampler sampler;

    const SamplerOutput output = sampler.sample(sequence_groups, make_logits(1), false, false);

    ASSERT_EQ(output.num_generated_tokens, 1);
    EXPECT_EQ(sequence_group->get_sequences().front()->get_generated_ids(), TokenIds({1}));
    EXPECT_FALSE(sequence_group->get_generation_stream()->can_read());

    sequence_group->notify_handle();
    ASSERT_TRUE(sequence_group->get_generation_stream()->can_read());
    const GenerationOutputs generation_outputs = sequence_group->get_generation_stream()->read();
    EXPECT_EQ(generation_outputs.at(0).generated_ids, TokenIds({1}));
}

TEST(SamplerNotificationTest, defers_terminal_output_until_explicit_notification) {
    GenerationConfig sampling_config = ov::genai::utils::get_greedy_config();
    sampling_config.max_new_tokens = 1;
    SequenceGroup::Ptr sequence_group = make_scheduled_group(0, sampling_config);
    std::vector<SequenceGroup::Ptr> sequence_groups{sequence_group};
    Sampler sampler;

    const SamplerOutput output = sampler.sample(sequence_groups, make_logits(1), false, false);

    ASSERT_EQ(output.m_dropped_sequences.size(), 1);
    EXPECT_TRUE(sequence_group->has_finished());
    EXPECT_EQ(sequence_group->get_generation_stream()->get_status(), GenerationStatus::RUNNING);
    EXPECT_FALSE(sequence_group->get_generation_stream()->can_read());

    sequence_group->notify_handle();
    EXPECT_EQ(sequence_group->get_generation_stream()->get_status(), GenerationStatus::FINISHED);
    ASSERT_TRUE(sequence_group->get_generation_stream()->can_read());
    EXPECT_EQ(sequence_group->get_generation_stream()->read().at(0).generated_ids, TokenIds({1}));
}

TEST(SamplerNotificationTest, precommitFailurePublishesNoGeneratedOutputAndPreservesException) {
    SequenceGroup::Ptr sequence_group = make_scheduled_group(0);
    std::vector<SequenceGroup::Ptr> sequence_groups{sequence_group};
    Sampler sampler;
    sampler.sample(sequence_groups, make_logits(1), false, false);
    ASSERT_FALSE(sequence_group->get_generation_stream()->can_read());

    const std::exception_ptr failure = std::make_exception_ptr(std::runtime_error("injected precommit failure"));
    sequence_group->fail_generation(failure);

    std::exception_ptr reader_failure;
    try {
        sequence_group->get_generation_stream()->read();
    } catch (...) {
        reader_failure = std::current_exception();
    }
    EXPECT_EQ(reader_failure, failure);
}

TEST(SamplerNotificationTest, deferredEchoUsesPreCounterUpdateRange) {
    GenerationConfig sampling_config = ov::genai::utils::get_greedy_config();
    sampling_config.echo = true;
    sampling_config.max_new_tokens = 0;
    const std::vector<int64_t> prompt{4, 5};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, prompt.size()}, prompt.data());
    SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, input_tensor, sampling_config);
    sequence_group->schedule_tokens(1);
    sequence_group->append_prompt_log_prob(1.0f);
    sequence_group->append_prompt_log_prob(2.0f);
    const size_t first_token_position = sequence_group->get_num_processed_tokens();
    const size_t last_token_position = sequence_group->get_context_len();
    sequence_group->finish_iteration();

    sequence_group->notify_handle_echo_only(first_token_position, last_token_position);

    const GenerationOutputs outputs = sequence_group->get_generation_stream()->read();
    EXPECT_EQ(outputs.at(0).generated_ids, TokenIds({4}));
    EXPECT_EQ(outputs.at(0).generated_log_probs, LogProbs({1.0f}));
}

TEST(SamplerStopTokenIdsTest, single_stop_token_match) {
    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    std::set<int64_t> stop_token_ids = {9};
    ASSERT_TRUE(is_stop_token_id_hit(generated_tokens.back(), stop_token_ids));
}

TEST(SamplerStopTokenIdsTest, multiple_stop_token_match) {
    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    std::set<int64_t> stop_token_ids = {7, 8, 9};
    ASSERT_TRUE(is_stop_token_id_hit(generated_tokens.back(), stop_token_ids));
}

TEST(SamplerStopTokenIdsTest, single_stop_sequence_no_match) {
    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    std::set<int64_t> stop_token_ids = { 10 };
    ASSERT_FALSE(is_stop_token_id_hit(generated_tokens.back(), stop_token_ids));
}

TEST(SamplerStopTokenIdsTest, multiple_stop_sequence_no_match) {
    std::vector<int64_t> generated_tokens = {3, 4, 5, 6, 7, 8, 9};
    std::set<int64_t> stop_token_ids = { 10, 10, 11 };
    ASSERT_FALSE(is_stop_token_id_hit(generated_tokens.back(), stop_token_ids));
}

TEST(SamplerValidationMode, gen_phase_to_cut_whole_seq) {
    auto sampling_config = ov::genai::utils::get_greedy_config();
    // create sequence group with prompt [0, 1, 2, 3, 4]
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    std::vector<SequenceGroup::Ptr> sequence_groups{
        SequenceGroup::Ptr(new SequenceGroup(0, input_tensor, sampling_config)),
    };

    // to emulate processed prompt and add next token [ 0 ]
    sequence_groups.front()->get_sequences().front()->append_token(0, 1.f);    
    constexpr size_t processed_before = 5;
    sequence_groups.front()->update_processed_tokens_num(processed_before);

    // append candidates [ 2, 3, 4 ]
    size_t num_validated_tokens = 3;
    for (size_t i = 1; i <= num_validated_tokens; ++i) {
        sequence_groups.front()->get_sequences().front()->append_token(i + 1, 1.f);
    }

    // generated sequence [0, 1, 2, 3, 4] -> [0, 2, 3, 4]
    sequence_groups.front()->set_num_validated_tokens(num_validated_tokens);
    const auto num_scheduled_tokens = sequence_groups.front()->get_num_available_tokens_for_batching();
    ASSERT_EQ(num_scheduled_tokens, num_validated_tokens + 1);
    sequence_groups.front()->schedule_tokens(num_scheduled_tokens);

    // create ref tensor : to generate candidates + next token
    std::vector<float> logits = {
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
        0, 0, 0, 0, 1.f,
    };

    // shape 4 tokens + 1 batch + 5 vocab
    ov::Tensor gen_input_ids(ov::element::f32, ov::Shape{4, 1, 5}, logits.data());

    Sampler sampler;
    sampler.sample(sequence_groups, gen_input_ids, true);

    TokenIds actual = sequence_groups.front()->get_sequences().front()->get_generated_ids(),
             expected{0, 1};
    ASSERT_EQ(sequence_groups.front()->get_sequences().front()->get_generated_ids(), expected);
    EXPECT_EQ(sequence_groups.front()->get_num_processed_tokens(), processed_before + 1)
        << "Full rejection must retain the target replacement token as accepted depth one";
}

TEST(SamplerValidationMode, gen_phase_to_cut_part_seq) {
    auto sampling_config = ov::genai::utils::get_greedy_config();
    // create sequence group with prompt [0, 1, 2, 3, 4]
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    std::vector<SequenceGroup::Ptr> sequence_groups{
        SequenceGroup::Ptr(new SequenceGroup(0, input_tensor, sampling_config)),
    };

    // to emulate processed prompt and add next token [ 0 ]
    sequence_groups.front()->get_sequences().front()->append_token(0, 1.f);    
    sequence_groups.front()->update_processed_tokens_num(5);

    // append candidates [ 1, 2, 2 ]
    size_t num_validated_tokens = 3;
    for (size_t i = 1; i <= num_validated_tokens; ++i) {
        int64_t token_id = i == num_validated_tokens ? i - 1 : i;
        sequence_groups.front()->get_sequences().front()->append_token(token_id, 1.f);
    }

    // generated sequence [0, 1, 2, 3, 4] -> [0, 1, 2, 2]
    sequence_groups.front()->set_num_validated_tokens(num_validated_tokens);
    const auto num_scheduled_tokens = sequence_groups.front()->get_num_available_tokens_for_batching();
    ASSERT_EQ(num_scheduled_tokens, num_validated_tokens + 1);
    sequence_groups.front()->schedule_tokens(num_scheduled_tokens);

    // create ref tensor : to generate candidates + next token
    std::vector<float> logits = {
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
        0, 0, 0, 0, 1.f,
    };

    // shape 4 tokens + 1 batch + 5 vocab
    ov::Tensor gen_input_ids(ov::element::f32, ov::Shape{4, 1, 5}, logits.data());

    Sampler sampler;
    sampler.sample(sequence_groups, gen_input_ids, true);

    TokenIds actual = sequence_groups.front()->get_sequences().front()->get_generated_ids(),
             expected{0, 1, 2, 3};
    ASSERT_EQ(sequence_groups.front()->get_sequences().front()->get_generated_ids(), expected);
}

TEST(SamplerValidationMode, gen_phase) {
    auto sampling_config = ov::genai::utils::get_greedy_config();
    // create sequence group with prompt [0, 1, 2, 3, 4]
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    std::vector<SequenceGroup::Ptr> sequence_groups{
        SequenceGroup::Ptr(new SequenceGroup(0, input_tensor, sampling_config)),
    };

    // to emulate processed prompt and add next token [ 0 ]
    sequence_groups.front()->get_sequences().front()->append_token(0, 1.f);    
    sequence_groups.front()->update_processed_tokens_num(5);

    // append candidates [ 1, 2, 3 ]
    size_t num_validated_tokens = 3;
    for (size_t i = 1; i <= num_validated_tokens; ++i) {
        sequence_groups.front()->get_sequences().front()->append_token(i, 1.f);
    }

    // generated sequence [0, 1, 2, 3, 4] -> [0, 1, 2, 3]
    sequence_groups.front()->set_num_validated_tokens(num_validated_tokens);
    const auto num_scheduled_tokens = sequence_groups.front()->get_num_available_tokens_for_batching();
    ASSERT_EQ(num_scheduled_tokens, num_validated_tokens + 1);
    sequence_groups.front()->schedule_tokens(num_scheduled_tokens);

    // create ref tensor : to generate candidates + next token
    std::vector<float> logits = {
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
        0, 0, 0, 0, 1.f,
    };

    // shape 4 tokens + 1 batch + 5 vocab
    ov::Tensor gen_input_ids(ov::element::f32, ov::Shape{4, 1, 5}, logits.data());

    Sampler sampler;
    sampler.sample(sequence_groups, gen_input_ids, true);

    TokenIds actual = sequence_groups.front()->get_sequences().front()->get_generated_ids(),
             expected{0, 1, 2, 3, 4};
    ASSERT_EQ(sequence_groups.front()->get_sequences().front()->get_generated_ids(), expected);
}

TEST(SamplerValidationMode, prompt_phase_to_cut_part_seq) {
    auto sampling_config = ov::genai::utils::get_greedy_config();
    // create sequence group with prompt [0, 1, 2, 3, 4]
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    std::vector<SequenceGroup::Ptr> sequence_groups{
        SequenceGroup::Ptr(new SequenceGroup(0, input_tensor, sampling_config)),
    };

    // append candidates [ 0, 1, 1 ]
    size_t num_validated_tokens = 3;
    for (size_t i = 0; i < num_validated_tokens; ++i) {
        int64_t token_id = i + 1 == num_validated_tokens ? i - 1 : i;
        sequence_groups.front()->get_sequences().front()->append_token(token_id, 1.f);
    }

    // generated sequence [0, 1, 2, 3, 4] -> [0, 1, 1]
    sequence_groups.front()->set_num_validated_tokens(num_validated_tokens);
    const auto num_scheduled_tokens = sequence_groups.front()->get_num_available_tokens_for_batching();
    // prompt len + validation
    ASSERT_EQ(num_scheduled_tokens, num_validated_tokens + input_vector.size());
    sequence_groups.front()->schedule_tokens(num_scheduled_tokens);

    // create ref tensor : to generate candidates + next token
    std::vector<float> logits = {
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
        0, 0, 0, 0, 1.f,
        1.f, 0, 0, 0, 0,
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
    };

    // shape 4 tokens + 1 batch + 5 vocab
    ov::Tensor gen_input_ids(ov::element::f32, ov::Shape{8, 1, 5}, logits.data());

    Sampler sampler;
    sampler.sample(sequence_groups, gen_input_ids, true);

    TokenIds actual = sequence_groups.front()->get_sequences().front()->get_generated_ids(),
             expected{0, 1, 2};
    ASSERT_EQ(sequence_groups.front()->get_sequences().front()->get_generated_ids(), expected);
}

TEST(SamplerValidationMode, prompt_phase_to_cut_whole_seq) {
    auto sampling_config = ov::genai::utils::get_greedy_config();
    // create sequence group with prompt [0, 1, 2, 3, 4]
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    std::vector<SequenceGroup::Ptr> sequence_groups{
        SequenceGroup::Ptr(new SequenceGroup(0, input_tensor, sampling_config)),
    };

    // append candidates [ 1, 2, 3 ]
    size_t num_validated_tokens = 3;
    for (size_t i = 0; i < num_validated_tokens; ++i) {
        sequence_groups.front()->get_sequences().front()->append_token(i + 1, 1.f);
    }

    // generated sequence [0, 1, 2, 3, 4] -> [1, 2, 3]
    sequence_groups.front()->set_num_validated_tokens(num_validated_tokens);
    const auto num_scheduled_tokens = sequence_groups.front()->get_num_available_tokens_for_batching();
    // prompt len + validation
    ASSERT_EQ(num_scheduled_tokens, num_validated_tokens + input_vector.size());
    sequence_groups.front()->schedule_tokens(num_scheduled_tokens);

    // create ref tensor : to generate candidates + next token
    std::vector<float> logits = {
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
        0, 0, 0, 0, 1.f,
        1.f, 0, 0, 0, 0,
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
    };

    // shape 4 tokens + 1 batch + 5 vocab
    ov::Tensor gen_input_ids(ov::element::f32, ov::Shape{8, 1, 5}, logits.data());

    Sampler sampler;
    sampler.sample(sequence_groups, gen_input_ids, true);

    TokenIds actual = sequence_groups.front()->get_sequences().front()->get_generated_ids(),
             expected{0};
    ASSERT_EQ(sequence_groups.front()->get_sequences().front()->get_generated_ids(), expected);
}

TEST(SamplerValidationMode, prompt_phase) {
    auto sampling_config = ov::genai::utils::get_greedy_config();
    // create sequence group with prompt [0, 1, 2, 3, 4]
    std::vector<int64_t> input_vector{0, 1, 2, 3, 4};
    ov::Tensor input_tensor(ov::element::i64, ov::Shape{1, 5}, input_vector.data());
    std::vector<SequenceGroup::Ptr> sequence_groups{
        SequenceGroup::Ptr(new SequenceGroup(0, input_tensor, sampling_config)),
    };

    // append candidates [ 0, 1, 2 ]
    size_t num_validated_tokens = 3;
    for (size_t i = 0; i < num_validated_tokens; ++i) {
        sequence_groups.front()->get_sequences().front()->append_token(i, 1.f);
    }

    // generated sequence [0, 1, 2, 3, 4] -> [0, 1, 2]
    sequence_groups.front()->set_num_validated_tokens(num_validated_tokens);
    const auto num_scheduled_tokens = sequence_groups.front()->get_num_available_tokens_for_batching();
    // prompt len + validation
    ASSERT_EQ(num_scheduled_tokens, num_validated_tokens + input_vector.size());
    sequence_groups.front()->schedule_tokens(num_scheduled_tokens);

    // create ref tensor : to generate candidates + next token
    std::vector<float> logits = {
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
        0, 0, 0, 0, 1.f,
        1.f, 0, 0, 0, 0,
        0, 1.f, 0, 0, 0,
        0, 0, 1.f, 0, 0,
        0, 0, 0, 1.f, 0,
    };

    // shape 4 tokens + 1 batch + 5 vocab
    ov::Tensor gen_input_ids(ov::element::f32, ov::Shape{8, 1, 5}, logits.data());

    Sampler sampler;
    sampler.sample(sequence_groups, gen_input_ids, true);

    TokenIds actual = sequence_groups.front()->get_sequences().front()->get_generated_ids(),
             expected{0, 1, 2, 3};
    ASSERT_EQ(sequence_groups.front()->get_sequences().front()->get_generated_ids(), expected);
}
