// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "threaded_streamer.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ov::genai::ThreadedStreamerWrapper;
using ::testing::An;
using ::testing::ElementsAreArray;

class MockStreamerBase : public ov::genai::StreamerBase {
private:
    std::chrono::milliseconds m_sleep_for{200};

public:
    bool should_sleep = false;
    bool should_drop = false;
    std::vector<int64_t> tokens;

    MockStreamerBase() {
        ON_CALL(*this, put(An<int64_t>())).WillByDefault([this](int64_t token) {
            if (should_sleep) {
                std::this_thread::sleep_for(m_sleep_for);
            }
            tokens.push_back(token);
            return should_drop;
        });

        ON_CALL(*this, end()).WillByDefault([this]() {
            if (should_sleep) {
                std::this_thread::sleep_for(m_sleep_for);
            }
        });
    }

    MOCK_METHOD(bool, put, (int64_t), (override));
    MOCK_METHOD(void, end, (), (override));
    virtual ~MockStreamerBase() override{};
};

class MockStreamerBaseFixture : public ::testing::TestWithParam<int> {
protected:
    ov::genai::Tokenizer tokenizer;
    std::shared_ptr<MockStreamerBase> streamer = std::make_shared<MockStreamerBase>();
    ThreadedStreamerWrapper threaded_streamer{streamer, tokenizer};
};

TEST_P(MockStreamerBaseFixture, general_test) {
    bool should_sleep = GetParam();
    streamer->should_sleep = should_sleep;

    std::vector<int64_t> generated_tokens{0, 1, 2, 3, 4};
    EXPECT_CALL(*streamer, put(An<int64_t>())).Times(generated_tokens.size());
    EXPECT_CALL(*streamer, end());

    threaded_streamer.start();

    EXPECT_FALSE(threaded_streamer.is_dropped());
    threaded_streamer.put(generated_tokens[0]);
    threaded_streamer.put(generated_tokens[1]);

    EXPECT_FALSE(threaded_streamer.is_dropped());
    std::vector<int64_t> value{generated_tokens[2], generated_tokens[3], generated_tokens[4]};
    threaded_streamer.put(value);

    EXPECT_FALSE(threaded_streamer.is_dropped());
    threaded_streamer.end();

    EXPECT_THAT(streamer->tokens, ElementsAreArray(generated_tokens));
}

INSTANTIATE_TEST_SUITE_P(SleepModes, MockStreamerBaseFixture, ::testing::Values(true, false));

TEST_F(MockStreamerBaseFixture, heavy_main_thread_test) {
    std::vector<int64_t> generated_tokens{0, 1, 2, 3, 4};
    EXPECT_CALL(*streamer, put(An<int64_t>())).Times(generated_tokens.size());
    EXPECT_CALL(*streamer, end());

    threaded_streamer.start();

    threaded_streamer.put(generated_tokens[0]);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    threaded_streamer.put(generated_tokens[1]);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::vector<int64_t> value{generated_tokens[2], generated_tokens[3], generated_tokens[4]};
    threaded_streamer.put(value);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    threaded_streamer.end();

    EXPECT_THAT(streamer->tokens, ElementsAreArray(generated_tokens));
}

TEST_F(MockStreamerBaseFixture, put_end_test) {
    EXPECT_CALL(*streamer, put(An<int64_t>()));
    EXPECT_CALL(*streamer, end());

    threaded_streamer.start();

    EXPECT_FALSE(threaded_streamer.is_dropped());

    threaded_streamer.put(0);
    threaded_streamer.end();

    EXPECT_FALSE(threaded_streamer.is_dropped());

    EXPECT_THAT(streamer->tokens, ElementsAreArray({0}));
}

TEST_F(MockStreamerBaseFixture, drop_test) {
    std::vector<int64_t> generated_tokens{0, 1};
    EXPECT_CALL(*streamer, put(An<int64_t>())).Times(generated_tokens.size());
    EXPECT_CALL(*streamer, end());

    threaded_streamer.start();

    EXPECT_FALSE(threaded_streamer.is_dropped());
    threaded_streamer.put(generated_tokens[0]);

    // wait to process prev token
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    EXPECT_FALSE(threaded_streamer.is_dropped());
    streamer->should_drop = true;
    threaded_streamer.put(generated_tokens[1]);

    // wait to process prev token
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    EXPECT_TRUE(threaded_streamer.is_dropped());

    threaded_streamer.end();

    EXPECT_THAT(streamer->tokens, ElementsAreArray(generated_tokens));
}
