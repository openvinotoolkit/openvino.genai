// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "automatic_speech_recognition/models/voxtral-realtime/cache_utils.hpp"
#include "automatic_speech_recognition/pipeline_base.hpp"

using ov::genai::ASRDecodedResults;
using ov::genai::ASRStreamImplBase;
using ov::genai::StreamingStatus;

namespace {

constexpr std::chrono::milliseconds BLOCKED_CHECK_TIMEOUT{100};

std::vector<float> make_frame(const size_t frame_id, const size_t producer_id = 0) {
    return {static_cast<float>(frame_id), static_cast<float>(producer_id)};
}

/// Records frames on the generate() thread and emits "<frame_id>," per frame.
class RecordingStream final : public ASRStreamImplBase {
public:
    explicit RecordingStream(const size_t max_queued_frames, const size_t failing_frame_id = SIZE_MAX)
        : ASRStreamImplBase{max_queued_frames},
          m_failing_frame_id{failing_frame_id} {}

    std::vector<std::vector<float>> processed_frames;
    size_t flush_calls = 0;

protected:
    std::string process_audio(const std::vector<float>& audio_frame) override {
        const size_t frame_id = static_cast<size_t>(audio_frame.at(0));
        OPENVINO_ASSERT(frame_id != m_failing_frame_id, "Injected inference failure.");
        processed_frames.push_back(audio_frame);
        return std::to_string(frame_id) + ",";
    }

    std::string flush() override {
        ++flush_calls;
        return "end";
    }

private:
    const size_t m_failing_frame_id;
};

}  // namespace

TEST(ASRStream, FramesArePushedWhileGenerationRuns) {
    constexpr size_t num_frames = 8;
    RecordingStream stream{2};
    std::vector<std::string> deltas;
    std::promise<void> first_delta;

    auto generation = std::async(std::launch::async, [&] {
        return stream.generate([&](std::string delta) {
            if (deltas.empty())
                first_delta.set_value();
            deltas.push_back(std::move(delta));
            return StreamingStatus::RUNNING;
        });
    });

    stream.push_audio(make_frame(0));
    // generate() has consumed frame 0 and is still running because end_audio() isn't called yet.
    first_delta.get_future().wait();
    EXPECT_EQ(generation.wait_for(std::chrono::seconds{0}), std::future_status::timeout);
    for (size_t frame_id = 1; frame_id < num_frames; ++frame_id)
        stream.push_audio(make_frame(frame_id));
    stream.end_audio();

    const ASRDecodedResults results = generation.get();
    ASSERT_EQ(results.texts.size(), 1);
    EXPECT_EQ(results.texts[0], "0,1,2,3,4,5,6,7,end");
    const std::vector<std::string> expected_deltas{"0,", "1,", "2,", "3,", "4,", "5,", "6,", "7,", "end"};
    EXPECT_EQ(deltas, expected_deltas);
    ASSERT_EQ(stream.processed_frames.size(), num_frames);
    for (size_t frame_id = 0; frame_id < num_frames; ++frame_id)
        EXPECT_EQ(stream.processed_frames[frame_id], make_frame(frame_id));
    EXPECT_EQ(stream.flush_calls, 1);
}

TEST(ASRStream, PushAudioBlocksWhileQueueIsFull) {
    RecordingStream stream{2};
    stream.push_audio(make_frame(0));
    stream.push_audio(make_frame(1));

    auto blocked_push = std::async(std::launch::async, [&] {
        stream.push_audio(make_frame(2));
    });
    EXPECT_EQ(blocked_push.wait_for(BLOCKED_CHECK_TIMEOUT), std::future_status::timeout);

    auto generation = std::async(std::launch::async, [&] {
        return stream.generate(nullptr);
    });
    blocked_push.get();
    stream.end_audio();

    EXPECT_EQ(generation.get().texts[0], "0,1,2,end");
}

TEST(ASRStream, ConcurrentProducersKeepPerProducerOrder) {
    constexpr size_t num_producers = 4;
    constexpr size_t frames_per_producer = 50;
    RecordingStream stream{3};

    auto generation = std::async(std::launch::async, [&] {
        return stream.generate(nullptr);
    });
    std::vector<std::thread> producers;
    for (size_t producer_id = 0; producer_id < num_producers; ++producer_id) {
        producers.emplace_back([&stream, producer_id] {
            for (size_t frame_id = 0; frame_id < frames_per_producer; ++frame_id)
                stream.push_audio(make_frame(frame_id, producer_id));
        });
    }
    for (std::thread& producer : producers)
        producer.join();
    stream.end_audio();
    generation.get();

    ASSERT_EQ(stream.processed_frames.size(), num_producers * frames_per_producer);
    std::map<size_t, size_t> next_frame_id;
    for (const std::vector<float>& frame : stream.processed_frames) {
        const size_t producer_id = static_cast<size_t>(frame.at(1));
        EXPECT_EQ(static_cast<size_t>(frame.at(0)), next_frame_id[producer_id]++);
    }
}

TEST(ASRStream, EndOfStreamRules) {
    RecordingStream stream{2};
    stream.push_audio(make_frame(0));
    stream.end_audio();
    EXPECT_THROW(stream.push_audio(make_frame(1)), ov::Exception);
    EXPECT_THROW(stream.end_audio(), ov::Exception);

    EXPECT_EQ(stream.generate(nullptr).texts[0], "0,end");
    EXPECT_THROW(stream.generate(nullptr), ov::Exception);
}

TEST(ASRStream, CallbackStopEndsGenerationAndRejectsAudio) {
    RecordingStream stream{2};
    stream.push_audio(make_frame(0));
    stream.push_audio(make_frame(1));

    const ASRDecodedResults results = stream.generate([](std::string) {
        return StreamingStatus::STOP;
    });

    EXPECT_EQ(results.texts[0], "0,");
    EXPECT_EQ(stream.flush_calls, 0);
    EXPECT_THROW(stream.push_audio(make_frame(2)), ov::Exception);
}

TEST(ASRStream, InferenceErrorPropagatesAndReleasesBlockedProducer) {
    constexpr size_t failing_frame_id = 2;
    RecordingStream stream{1, failing_frame_id};
    std::atomic<bool> producer_failed{false};

    std::thread producer([&] {
        try {
            for (size_t frame_id = 0;; ++frame_id)
                stream.push_audio(make_frame(frame_id));
        } catch (const ov::Exception&) {
            producer_failed = true;
        }
    });

    EXPECT_THROW(stream.generate(nullptr), ov::Exception);
    producer.join();
    EXPECT_TRUE(producer_failed);
    EXPECT_EQ(stream.processed_frames.size(), failing_frame_id);
    EXPECT_EQ(stream.flush_calls, 0);
}

TEST(VoxtralCache, CropsAndRephasesRotaryKeys) {
    ov::Tensor key(ov::element::f32, {1, 1, 5, 4});
    ov::Tensor value(ov::element::f32, {1, 1, 5, 4});
    for (size_t index = 0; index < key.get_size(); ++index) {
        key.data<float>()[index] = static_cast<float>(index + 1);
        value.data<float>()[index] = static_cast<float>(index + 11);
    }

    constexpr size_t max_context = 3;
    constexpr float rope_theta = 10000.0f;
    const auto [cropped_key, cropped_value] =
        ov::genai::voxtral_realtime::crop_and_rephase_cache(key, value, max_context, rope_theta);

    ASSERT_EQ(cropped_key.get_shape(), (ov::Shape{1, 1, max_context, 4}));
    ASSERT_EQ(cropped_value.get_shape(), cropped_key.get_shape());
    const float* cropped_key_data = cropped_key.data<const float>();
    const float* cropped_value_data = cropped_value.data<const float>();
    EXPECT_FLOAT_EQ(cropped_value_data[0], value.data<const float>()[8]);
    EXPECT_FLOAT_EQ(cropped_value_data[11], value.data<const float>()[19]);

    constexpr size_t evicted_tokens = 2;
    const double angle = -static_cast<double>(evicted_tokens);
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    const double first = key.data<const float>()[8];
    const double second = key.data<const float>()[10];
    EXPECT_NEAR(cropped_key_data[0], first * cosine - second * sine, 1e-6);
    EXPECT_NEAR(cropped_key_data[2], second * cosine + first * sine, 1e-6);
}
