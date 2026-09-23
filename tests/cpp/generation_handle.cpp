// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <future>
#include <stdexcept>

#include "generation_stream.hpp"
#include "gtest/gtest.h"
#include "openvino/genai/generation_handle.hpp"
#include "sequence_group.hpp"
#include "utils.hpp"

namespace {

using namespace std::chrono_literals;

ov::genai::GenerationOutputs make_outputs(int64_t token_id) {
    ov::genai::GenerationOutput output;
    output.generated_ids = {token_id};
    output.generated_log_probs = {0.0f};
    return {{0, std::move(output)}};
}

ov::genai::GenerationHandle make_handle(const ov::genai::GenerationStream::Ptr& stream) {
    return std::make_shared<ov::genai::GenerationHandleImpl>(stream, ov::genai::GenerationConfig{});
}

class StreamStopGuard {
public:
    explicit StreamStopGuard(const ov::genai::GenerationStream::Ptr& stream) : m_stream(stream) {}

    ~StreamStopGuard() {
        m_stream->stop();
    }

private:
    const ov::genai::GenerationStream::Ptr& m_stream;
};

void expect_out_of_memory_notification(const ov::genai::GenerationConfig& config) {
    const std::vector<int64_t> prompt{1};
    const auto sequence_group = std::make_shared<ov::genai::SequenceGroup>(0, prompt, config);
    const auto stream = sequence_group->get_generation_stream();
    const auto handle = std::make_shared<ov::genai::GenerationHandleImpl>(stream, config);
    std::promise<void> reader_started_promise;
    std::future<void> reader_started = reader_started_promise.get_future();
    auto reader = std::async(std::launch::async, [&handle, &reader_started_promise] {
        reader_started_promise.set_value();
        return handle->read();
    });
    StreamStopGuard stream_stop_guard(stream);

    ASSERT_EQ(reader_started.wait_for(1s), std::future_status::ready);
    ASSERT_EQ(reader.wait_for(100ms), std::future_status::timeout);
    sequence_group->set_out_of_memory();
    sequence_group->notify_handle();

    EXPECT_EQ(reader.wait_for(1s), std::future_status::ready);
    EXPECT_NO_THROW(reader.get());
    EXPECT_EQ(handle->get_status(), ov::genai::GenerationStatus::IGNORED);
}

TEST(GenerationStreamContractTest, BlockedReadWakesAndRethrowsOriginalFailurePersistently) {
    const auto stream = ov::genai::GenerationStream::create();
    const auto handle = make_handle(stream);
    std::promise<void> reader_started_promise;
    std::future<void> reader_started = reader_started_promise.get_future();
    auto reader = std::async(std::launch::async, [&handle, &reader_started_promise] {
        reader_started_promise.set_value();
        return handle->read();
    });
    StreamStopGuard stream_stop_guard(stream);

    ASSERT_EQ(reader_started.wait_for(1s), std::future_status::ready);
    ASSERT_EQ(reader.wait_for(100ms), std::future_status::timeout);
    stream->fail(std::make_exception_ptr(std::runtime_error("original failure")));

    EXPECT_THROW(
        try {
            reader.get();
        } catch (const std::runtime_error& error) {
            EXPECT_STREQ(error.what(), "original failure");
            throw;
        },
        std::runtime_error);
    EXPECT_EQ(handle->get_status(), ov::genai::GenerationStatus::FAILED);
    EXPECT_TRUE(handle->can_read());
    EXPECT_THROW(handle->read(), std::runtime_error);
}

TEST(GenerationStreamContractTest, ReadReturnsQueuedOutputBeforePersistentFailure) {
    const auto stream = ov::genai::GenerationStream::create();
    const auto handle = make_handle(stream);
    stream->push(make_outputs(17));
    stream->fail(std::make_exception_ptr(std::logic_error("queued failure")));

    EXPECT_EQ(handle->read().at(0).generated_ids, std::vector<int64_t>({17}));
    for (size_t attempt = 0; attempt < 2; ++attempt) {
        EXPECT_THROW(
            try {
                handle->read();
            } catch (const std::logic_error& error) {
                EXPECT_STREQ(error.what(), "queued failure");
                throw;
            },
            std::logic_error);
    }
}

TEST(GenerationStreamContractTest, ReadAllDrainsThenRethrowsFailure) {
    const auto stream = ov::genai::GenerationStream::create();
    const auto handle = make_handle(stream);
    stream->push(make_outputs(23));
    auto reader = std::async(std::launch::async, [&handle] {
        return handle->read_all();
    });
    StreamStopGuard stream_stop_guard(stream);

    stream->fail(std::make_exception_ptr(std::runtime_error("read_all failure")));

    EXPECT_THROW(
        try {
            reader.get();
        } catch (const std::runtime_error& error) {
            EXPECT_STREQ(error.what(), "read_all failure");
            throw;
        },
        std::runtime_error);
}

TEST(GenerationStreamContractTest, TerminalOutputAndFinishedStatusArePublishedTogether) {
    const auto stream = ov::genai::GenerationStream::create();
    const auto handle = make_handle(stream);
    auto reader = std::async(std::launch::async, [&handle] {
        return handle->read();
    });
    StreamStopGuard stream_stop_guard(stream);

    stream->push_and_close(make_outputs(31), ov::genai::GenerationStatus::FINISHED);

    EXPECT_EQ(reader.get().at(0).generated_ids, std::vector<int64_t>({31}));
    EXPECT_EQ(handle->get_status(), ov::genai::GenerationStatus::FINISHED);
    EXPECT_FALSE(handle->can_read());
    EXPECT_THROW(handle->read(), ov::Exception);
}

TEST(GenerationStreamContractTest, ReadAllReturnsEntireSuccessfulTerminalTail) {
    const auto stream = ov::genai::GenerationStream::create();
    const auto handle = make_handle(stream);
    stream->push(make_outputs(31));
    auto reader = std::async(std::launch::async, [&handle] {
        return handle->read_all();
    });
    StreamStopGuard stream_stop_guard(stream);

    stream->push_and_close(make_outputs(37), ov::genai::GenerationStatus::FINISHED);

    const auto outputs = reader.get();
    ASSERT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs.front().generated_ids, std::vector<int64_t>({31, 37}));
    EXPECT_EQ(outputs.front().generated_log_probs, std::vector<float>({0.0f, 0.0f}));
}

TEST(GenerationStreamContractTest, BlockedReadWakesEmptyOnStopAndCancel) {
    for (const bool cancel : {false, true}) {
        const auto stream = ov::genai::GenerationStream::create();
        const auto handle = make_handle(stream);
        std::promise<void> reader_started_promise;
        std::future<void> reader_started = reader_started_promise.get_future();
        auto reader = std::async(std::launch::async, [&handle, &reader_started_promise] {
            reader_started_promise.set_value();
            return handle->read();
        });
        StreamStopGuard stream_stop_guard(stream);

        ASSERT_EQ(reader_started.wait_for(1s), std::future_status::ready);
        ASSERT_EQ(reader.wait_for(100ms), std::future_status::timeout);
        cancel ? handle->cancel() : handle->stop();

        EXPECT_TRUE(reader.get().empty());
        EXPECT_THROW(handle->read(), ov::Exception);
    }
}

TEST(GenerationStreamContractTest, StopAndCancelDiscardQueuedOutput) {
    for (const bool cancel : {false, true}) {
        const auto stream = ov::genai::GenerationStream::create();
        stream->push(make_outputs(41));
        cancel ? stream->cancel() : stream->stop();

        EXPECT_FALSE(stream->can_read());
        EXPECT_TRUE(stream->read().empty());
    }

    const auto stream = ov::genai::GenerationStream::create();
    stream->push(make_outputs(43));
    stream->set_generation_status(ov::genai::GenerationStatus::STOP);

    EXPECT_FALSE(stream->can_read());
    EXPECT_TRUE(stream->read().empty());
}

TEST(GenerationStreamContractTest, StopAndCancelRemainUnreadableUnlessFailureOverrides) {
    for (const bool cancel : {false, true}) {
        const auto stream = ov::genai::GenerationStream::create();
        const auto handle = make_handle(stream);
        cancel ? handle->cancel() : handle->stop();

        EXPECT_FALSE(handle->can_read());
        EXPECT_THROW(handle->read(), ov::Exception);

        stream->fail(std::make_exception_ptr(std::runtime_error("failure wins")));
        EXPECT_EQ(handle->get_status(), ov::genai::GenerationStatus::FAILED);
        EXPECT_TRUE(handle->can_read());
        EXPECT_THROW(handle->read(), std::runtime_error);
    }
}

TEST(GenerationStreamContractTest, TerminalStatusCannotBeOverwritten) {
    for (const auto status : {ov::genai::GenerationStatus::FINISHED, ov::genai::GenerationStatus::IGNORED}) {
        const auto stream = ov::genai::GenerationStream::create();
        {
            const auto handle = make_handle(stream);
            stream->push_and_close({}, status);
            handle->stop();
            handle->cancel();
            stream->fail(std::make_exception_ptr(std::runtime_error("late failure")));
        }
        EXPECT_EQ(stream->get_status(), status);
    }

    const auto failed_stream = ov::genai::GenerationStream::create();
    {
        const auto handle = make_handle(failed_stream);
        failed_stream->fail(std::make_exception_ptr(std::runtime_error("failure")));
        handle->stop();
        handle->cancel();
    }
    EXPECT_EQ(failed_stream->get_status(), ov::genai::GenerationStatus::FAILED);
}

TEST(GenerationStreamContractTest, FirstFailureWins) {
    const auto stream = ov::genai::GenerationStream::create();
    const auto handle = make_handle(stream);
    stream->fail(std::make_exception_ptr(std::runtime_error("first")));
    stream->fail(std::make_exception_ptr(std::logic_error("second")));

    EXPECT_THROW(
        try {
            handle->read();
        } catch (const std::runtime_error& error) {
            EXPECT_STREQ(error.what(), "first");
            throw;
        },
        std::runtime_error);
}

TEST(GenerationStreamContractTest, OutOfMemoryNotifiesBeamSearchHandle) {
    auto config = ov::genai::utils::get_greedy_config();
    config.num_beams = 2;
    expect_out_of_memory_notification(config);
}

TEST(GenerationStreamContractTest, OutOfMemoryNotifiesGreedyHandleWithNoGeneratedTokens) {
    auto config = ov::genai::utils::get_greedy_config();
    config.max_new_tokens = 0;
    expect_out_of_memory_notification(config);
}

TEST(GenerationStreamContractTest, OutOfMemoryNotifiesMultiReturnHandle) {
    auto config = ov::genai::utils::get_greedy_config();
    config.do_sample = true;
    config.num_return_sequences = 2;
    expect_out_of_memory_notification(config);
}

}  // namespace
