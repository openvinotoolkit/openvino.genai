// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "automatic_speech_recognition/models/voxtral-realtime/paged_text_decoder.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/variable.hpp"

using ov::genai::voxtral_realtime::count_blocks_outside_window;
using ov::genai::voxtral_realtime::PagedTextDecoder;

namespace {

constexpr size_t NUM_CLASSES = 2;
// Odd, so that one of two classes always prevails in a full window.
constexpr size_t SLIDING_WINDOW = 5;
constexpr size_t NUM_LAYERS = 1;
constexpr size_t NUM_DELAY_TOKENS = 0;
constexpr size_t CLASS_RUN_LENGTH = 7;
// Longer than three KV-cache blocks of any device.
constexpr size_t NUM_POSITIONS = 100;

/// One attention layer with a zero query. It averages the embeddings in its window, so for one-hot embeddings the
/// decoded token is the class that prevails among the attended positions.
std::shared_ptr<ov::Model> make_window_majority_decoder() {
    // SDPAToPagedAttention finds parameters by friendly name.
    const auto make_parameter = [](const std::string& name, const ov::element::Type& type, const ov::PartialShape& shape) {
        auto parameter = std::make_shared<ov::op::v0::Parameter>(type, shape);
        parameter->set_friendly_name(name);
        parameter->output(0).set_names({name});
        return parameter;
    };
    auto inputs_embeds = make_parameter("inputs_embeds", ov::element::f32, ov::PartialShape{-1, -1, NUM_CLASSES});
    auto beam_idx = make_parameter("beam_idx", ov::element::i32, ov::PartialShape{-1});
    auto num_delay_tokens = make_parameter("num_delay_tokens", ov::element::f32, ov::PartialShape{1});
    const auto axis = [](const int64_t value) {
        return ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {value});
    };

    // [batch, heads, positions, NUM_CLASSES]
    auto current = std::make_shared<ov::op::v0::Unsqueeze>(inputs_embeds, axis(1));
    ov::SinkVector sinks;
    const auto make_cache = [&](const std::string& variable_id) {
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape{-1, 1, -1, NUM_CLASSES}, ov::element::f32, variable_id});
        auto initial =
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 0, NUM_CLASSES}, std::vector<float>{});
        auto read = std::make_shared<ov::op::v6::ReadValue>(initial, variable);
        auto past = std::make_shared<ov::op::v8::Gather>(read, beam_idx, axis(0));
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past, current}, 2);
        sinks.push_back(std::make_shared<ov::op::v6::Assign>(concat, variable));
        return concat;
    };
    auto key = make_cache("key");
    auto value = make_cache("value");
    auto query = std::make_shared<ov::op::v1::Multiply>(
        current,
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f}));
    auto mask = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, {0.0f});
    auto attention = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, mask, false);
    auto hidden =
        std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v0::Squeeze>(attention, axis(1)), num_delay_tokens);
    auto lm_head =
        ov::op::v0::Constant::create(ov::element::f32, ov::Shape{NUM_CLASSES, NUM_CLASSES}, {1.0f, 0.0f, 0.0f, 1.0f});
    auto logits = std::make_shared<ov::op::v0::Result>(std::make_shared<ov::op::v0::MatMul>(hidden, lm_head));
    logits->output(0).set_names({"logits"});
    return std::make_shared<ov::Model>(ov::ResultVector{logits},
                                       sinks,
                                       ov::ParameterVector{inputs_embeds, beam_idx, num_delay_tokens});
}

std::shared_ptr<PagedTextDecoder> make_decoder(const ov::genai::SchedulerConfig& scheduler_config = {}) {
    return std::make_shared<PagedTextDecoder>(make_window_majority_decoder(),
                                              "CPU",
                                              ov::AnyMap{},
                                              scheduler_config,
                                              NUM_LAYERS,
                                              SLIDING_WINDOW,
                                              NUM_DELAY_TOKENS);
}

int64_t decode_class(PagedTextDecoder::Stream& stream, const size_t class_id) {
    ov::Tensor embedding{ov::element::f32, ov::Shape{1, NUM_CLASSES}};
    std::fill_n(embedding.data<float>(), NUM_CLASSES, 0.0f);
    embedding.data<float>()[class_id] = 1.0f;
    return stream.decode(embedding).token_id;
}

/// Runs of CLASS_RUN_LENGTH equal classes, starting with first_class.
std::vector<size_t> make_classes(const size_t first_class) {
    std::vector<size_t> classes(NUM_POSITIONS);
    for (size_t position = 0; position < NUM_POSITIONS; ++position)
        classes[position] = (first_class + position / CLASS_RUN_LENGTH) % NUM_CLASSES;
    return classes;
}

/// Class that prevails among the SLIDING_WINDOW classes ending at position.
int64_t prevailing_class(const std::vector<size_t>& classes, const size_t position) {
    const auto window_end = classes.begin() + position + 1;
    const size_t ones = std::count(window_end - SLIDING_WINDOW, window_end, size_t{1});
    return ones * NUM_CLASSES > SLIDING_WINDOW ? 1 : 0;
}

void expect_prevailing_classes(const std::vector<size_t>& classes, const std::vector<int64_t>& tokens) {
    ASSERT_EQ(tokens.size(), classes.size());
    // Earlier windows aren't full and may be tied.
    for (size_t position = SLIDING_WINDOW - 1; position < classes.size(); ++position)
        EXPECT_EQ(tokens[position], prevailing_class(classes, position)) << "position " << position;
}

}  // namespace

TEST(VoxtralPagedTextDecoder, CountsBlocksOutsideWindow) {
    constexpr size_t block_size = 32;
    constexpr size_t window = 8192;
    // The next position attends to the last window - 1 cached tokens.
    EXPECT_EQ(count_blocks_outside_window(0, window, block_size), 0);
    EXPECT_EQ(count_blocks_outside_window(window - 1, window, block_size), 0);
    EXPECT_EQ(count_blocks_outside_window(window - 1 + block_size - 1, window, block_size), 0);
    EXPECT_EQ(count_blocks_outside_window(window - 1 + block_size, window, block_size), 1);
    EXPECT_EQ(count_blocks_outside_window(window - 1 + 3 * block_size + 5, window, block_size), 3);
    // A window of one token attends to no cached token.
    EXPECT_EQ(count_blocks_outside_window(2 * block_size, 1, block_size), 2);
    EXPECT_THROW(count_blocks_outside_window(block_size, 0, block_size), ov::Exception);
}

TEST(VoxtralPagedTextDecoder, AttendsToSlidingWindowAndReleasesBlocks) {
    const std::shared_ptr<PagedTextDecoder> decoder = make_decoder();
    const std::unique_ptr<PagedTextDecoder::Stream> stream = decoder->create_stream();
    const std::vector<size_t> classes = make_classes(0);
    std::vector<int64_t> tokens;
    size_t block_size_bytes = 0;
    size_t max_kv_cache_size = 0;
    for (const size_t class_id : classes) {
        tokens.push_back(decode_class(*stream, class_id));
        if (block_size_bytes == 0)
            block_size_bytes = stream->get_kv_cache_size();
        max_kv_cache_size = std::max(max_kv_cache_size, stream->get_kv_cache_size());
    }

    // Without the window the classes would be tied at the end of every run of class 1, which decodes class 0.
    expect_prevailing_classes(classes, tokens);
    EXPECT_EQ(stream->get_window_size(), SLIDING_WINDOW);
    ASSERT_GT(block_size_bytes, 0);
    // The window never spans more than two blocks, although the stream is longer than three.
    EXPECT_EQ(max_kv_cache_size, 2 * block_size_bytes);
}

TEST(VoxtralPagedTextDecoder, DecodesActiveStreamsInOneBatchWithoutMixingThem) {
    const std::shared_ptr<PagedTextDecoder> decoder = make_decoder();
    const std::vector<std::vector<size_t>> classes{make_classes(0), make_classes(1)};
    std::vector<std::unique_ptr<PagedTextDecoder::Stream>> streams;
    std::vector<PagedTextDecoder::Stream::ActiveScope> active_scopes;
    for (size_t index = 0; index < classes.size(); ++index) {
        streams.push_back(decoder->create_stream());
        // Both streams are active before either decodes, so every step waits for both.
        active_scopes.push_back(streams.back()->activate());
    }

    std::vector<std::vector<int64_t>> tokens(classes.size());
    std::vector<std::thread> threads;
    for (size_t index = 0; index < classes.size(); ++index) {
        threads.emplace_back([&, index] {
            for (const size_t class_id : classes[index])
                tokens[index].push_back(decode_class(*streams[index], class_id));
        });
    }
    for (std::thread& thread : threads)
        thread.join();
    active_scopes.clear();

    // The streams hold opposite classes, so tokens of a stream decoded from the cache of the other would be wrong.
    for (size_t index = 0; index < classes.size(); ++index)
        expect_prevailing_classes(classes[index], tokens[index]);
}

TEST(VoxtralPagedTextDecoder, StreamWithoutFreeBlocksFailsAloneUntilBlocksAreReleased) {
    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.num_kv_blocks = 2;
    const std::shared_ptr<PagedTextDecoder> decoder = make_decoder(scheduler_config);
    const std::unique_ptr<PagedTextDecoder::Stream> first = decoder->create_stream();
    std::unique_ptr<PagedTextDecoder::Stream> second = decoder->create_stream();
    const std::vector<size_t> classes = make_classes(0);
    std::vector<int64_t> tokens;

    size_t position = 0;
    const size_t block_size_bytes = [&] {
        tokens.push_back(decode_class(*first, classes[position++]));
        return first->get_kv_cache_size();
    }();
    while (first->get_kv_cache_size() == block_size_bytes)
        tokens.push_back(decode_class(*first, classes[position++]));
    // The first stream holds both blocks.
    EXPECT_THROW(decode_class(*second, 0), ov::Exception);

    // Its first block leaves the window a few positions later.
    while (first->get_kv_cache_size() != block_size_bytes)
        tokens.push_back(decode_class(*first, classes[position++]));
    EXPECT_EQ(decode_class(*second, 1), 1);

    // A destroyed stream returns its blocks.
    second.reset();
    while (position < classes.size())
        tokens.push_back(decode_class(*first, classes[position++]));
    expect_prevailing_classes(classes, tokens);
}
