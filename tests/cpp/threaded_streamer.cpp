// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "threaded_streamer.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ov::genai::ThreadedStreamerWrapper;
using ::testing::An;

class MockStreamerBase : public ov::genai::StreamerBase {
private:
    std::chrono::milliseconds m_sleep_for{200};

public:
    bool should_sleep = false;
    bool should_drop = false;

    MockStreamerBase() {
        ON_CALL(*this, put(An<int64_t>())).WillByDefault([this](int64_t token) {
            if (should_sleep) {
                std::this_thread::sleep_for(m_sleep_for);
            }
            return should_drop;
        });

        ON_CALL(*this, put(An<const std::vector<int64_t>&>()))
            .WillByDefault([this](const std::vector<int64_t>& tokens) {
                if (should_sleep) {
                    std::this_thread::sleep_for(m_sleep_for);
                }
                return should_drop;
            });

        ON_CALL(*this, end()).WillByDefault([this]() {
            if (should_sleep) {
                std::this_thread::sleep_for(m_sleep_for);
            }
        });
    }

    MOCK_METHOD(bool, put, (int64_t), (override));
    MOCK_METHOD(bool, put, (const std::vector<int64_t>&), (override));
    MOCK_METHOD(void, end, (), (override));
};

TEST(TestThreadedStreamer, general_test) {
    ov::genai::Tokenizer tokenizer{};
    const auto streamer = std::make_shared<MockStreamerBase>();

    ThreadedStreamerWrapper threaded_streamer(streamer, tokenizer);

    threaded_streamer.start();

    EXPECT_FALSE(threaded_streamer.is_dropped());
    EXPECT_CALL(*streamer, put(0));
    EXPECT_CALL(*streamer, put(1));

    threaded_streamer.put(0);
    threaded_streamer.put(1);

    EXPECT_FALSE(threaded_streamer.is_dropped());
    std::vector<int64_t> value{0, 1, 2};
    EXPECT_CALL(*streamer, put(value));
    threaded_streamer.put(value);

    EXPECT_FALSE(threaded_streamer.is_dropped());
    EXPECT_CALL(*streamer, end());
    threaded_streamer.end();
}

TEST(TestThreadedStreamer, heavy_callback_test) {
    ov::genai::Tokenizer tokenizer{};
    const auto streamer = std::make_shared<MockStreamerBase>();
    streamer->should_sleep = true;

    ThreadedStreamerWrapper threaded_streamer(streamer, tokenizer);

    threaded_streamer.start();

    EXPECT_FALSE(threaded_streamer.is_dropped());
    EXPECT_CALL(*streamer, put(0)).Times(3);

    std::vector<int64_t> value{0, 1, 2};
    EXPECT_CALL(*streamer, put(value));

    EXPECT_CALL(*streamer, end());

    threaded_streamer.put(0);
    threaded_streamer.put(0);
    threaded_streamer.put(0);
    EXPECT_FALSE(threaded_streamer.is_dropped());
    threaded_streamer.put(value);
    EXPECT_FALSE(threaded_streamer.is_dropped());

    threaded_streamer.end();
}

TEST(TestThreadedStreamer, heavy_main_thread_test) {
    ov::genai::Tokenizer tokenizer{};
    const auto streamer = std::make_shared<MockStreamerBase>();

    ThreadedStreamerWrapper threaded_streamer(streamer, tokenizer);

    threaded_streamer.start();

    EXPECT_CALL(*streamer, put(0));
    EXPECT_CALL(*streamer, put(1));
    threaded_streamer.put(0);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    threaded_streamer.put(1);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::vector<int64_t> value{0, 1, 2};
    EXPECT_CALL(*streamer, put(value));
    threaded_streamer.put(value);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    EXPECT_CALL(*streamer, end());
    threaded_streamer.end();
}

TEST(TestThreadedStreamer, put_end_test) {
    ov::genai::Tokenizer tokenizer{};
    const auto streamer = std::make_shared<MockStreamerBase>();

    ThreadedStreamerWrapper threaded_streamer(streamer, tokenizer);

    threaded_streamer.start();

    EXPECT_FALSE(threaded_streamer.is_dropped());
    EXPECT_CALL(*streamer, put(0));
    EXPECT_CALL(*streamer, end());

    threaded_streamer.put(0);
    threaded_streamer.end();

    EXPECT_FALSE(threaded_streamer.is_dropped());
}

TEST(TestThreadedStreamer, drop_test) {
    ov::genai::Tokenizer tokenizer{};
    const auto streamer = std::make_shared<MockStreamerBase>();

    ThreadedStreamerWrapper threaded_streamer(streamer, tokenizer);

    threaded_streamer.start();

    EXPECT_FALSE(threaded_streamer.is_dropped());
    EXPECT_CALL(*streamer, put(0));
    threaded_streamer.put(0);

    // wait to process prev token
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    EXPECT_FALSE(threaded_streamer.is_dropped());
    streamer->should_drop = true;
    EXPECT_CALL(*streamer, put(1));
    threaded_streamer.put(1);

    // wait to process prev token
    std::this_thread::sleep_for(std::chrono::milliseconds(4));
    EXPECT_TRUE(threaded_streamer.is_dropped());
    EXPECT_CALL(*streamer, end());
    threaded_streamer.end();
}
