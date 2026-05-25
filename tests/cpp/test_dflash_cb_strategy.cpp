// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "speculative_decoding/continuous_batching/dflash_strategy_utils.hpp"

namespace {

ov::Tensor make_token_major_hidden_delta(size_t seq_len, size_t hidden_size, float start = 0.0f) {
    ov::Tensor tensor(ov::element::f32, ov::Shape{seq_len, 1, hidden_size});
    std::iota(tensor.data<float>(), tensor.data<float>() + tensor.get_size(), start);
    return tensor;
}

std::vector<float> tensor_values(const ov::Tensor& tensor) {
    const auto* data = tensor.data<const float>();
    return std::vector<float>(data, data + tensor.get_size());
}

std::vector<int64_t> int64_tensor_values(const ov::Tensor& tensor) {
    const auto* data = tensor.data<const int64_t>();
    return std::vector<int64_t>(data, data + tensor.get_size());
}

}  // namespace

TEST(DFlashCBHiddenDeltaBuffer, AppendsAndMaterializesSingleChunk) {
    ov::genai::dflash_cb::HiddenDeltaBuffer buffer;
    auto hidden_delta = make_token_major_hidden_delta(2, 3);

    buffer.append(hidden_delta);
    auto materialized = buffer.materialize();

    ASSERT_FALSE(buffer.empty());
    ASSERT_EQ(buffer.token_count(), 2);
    ASSERT_EQ(materialized.get_shape(), ov::Shape({2, 1, 3}));
    ASSERT_EQ(tensor_values(materialized), tensor_values(hidden_delta));
}

TEST(DFlashCBHiddenDeltaBuffer, MergesChunksInOrder) {
    ov::genai::dflash_cb::HiddenDeltaBuffer buffer;
    auto first = make_token_major_hidden_delta(2, 2, 0.0f);
    auto second = make_token_major_hidden_delta(1, 2, 4.0f);

    buffer.append(first);
    buffer.append(second);
    auto materialized = buffer.materialize();

    ASSERT_EQ(buffer.token_count(), 3);
    ASSERT_EQ(materialized.get_shape(), ov::Shape({3, 1, 2}));
    ASSERT_EQ(tensor_values(materialized), (std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}));
}

TEST(DFlashCBHiddenState, TruncatesRejectedTail) {
    auto hidden_delta = make_token_major_hidden_delta(4, 2);

    auto unchanged = ov::genai::dflash_cb::truncate_normalized_hidden_state_from_end(hidden_delta, 0);
    ASSERT_EQ(unchanged.get_shape(), ov::Shape({4, 1, 2}));
    ASSERT_EQ(tensor_values(unchanged), tensor_values(hidden_delta));

    auto truncated = ov::genai::dflash_cb::truncate_normalized_hidden_state_from_end(hidden_delta, 1);
    ASSERT_EQ(truncated.get_shape(), ov::Shape({3, 1, 2}));
    ASSERT_EQ(tensor_values(truncated), (std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}));

    auto empty = ov::genai::dflash_cb::truncate_normalized_hidden_state_from_end(hidden_delta, 4);
    ASSERT_EQ(empty.get_shape(), ov::Shape({0, 1, 2}));
    ASSERT_EQ(empty.get_size(), 0);
}

TEST(DFlashCBDraftInputs, BuildsSeedMaskBlock) {
    auto input_ids = ov::genai::dflash_cb::build_draft_input_ids(42, 99, 4);

    ASSERT_EQ(input_ids.get_shape(), ov::Shape({1, 4}));
    ASSERT_EQ(int64_tensor_values(input_ids), (std::vector<int64_t>{42, 99, 99, 99}));
}

TEST(DFlashCBDraftInputs, BuildsPositionIdsFromCommittedLength) {
    auto position_ids = ov::genai::dflash_cb::build_draft_position_ids(5, 2, 4);

    ASSERT_EQ(position_ids.get_shape(), ov::Shape({1, 6}));
    ASSERT_EQ(int64_tensor_values(position_ids), (std::vector<int64_t>{5, 6, 7, 8, 9, 10}));
}

TEST(DFlashCBCandidatePlanning, RespectsFinalToken) {
    ASSERT_EQ(ov::genai::dflash_cb::candidate_count(4, 0, 10), 3);
    ASSERT_EQ(ov::genai::dflash_cb::candidate_count(4, 8, 10), 1);
    ASSERT_EQ(ov::genai::dflash_cb::candidate_count(4, 9, 10), 0);
    ASSERT_EQ(ov::genai::dflash_cb::candidate_count(4, 10, 10), 0);
}

TEST(DFlashCBValidationAccounting, ComputesAcceptedAndRejected) {
    auto full_accept = ov::genai::dflash_cb::validation_accounting(3, 1, 5);
    ASSERT_TRUE(full_accept.target_extended);
    ASSERT_EQ(full_accept.accepted, 3);
    ASSERT_EQ(full_accept.rejected, 0);

    auto partial_accept = ov::genai::dflash_cb::validation_accounting(3, 1, 3);
    ASSERT_TRUE(partial_accept.target_extended);
    ASSERT_EQ(partial_accept.accepted, 1);
    ASSERT_EQ(partial_accept.rejected, 2);

    auto full_reject = ov::genai::dflash_cb::validation_accounting(3, 1, 2);
    ASSERT_TRUE(full_reject.target_extended);
    ASSERT_EQ(full_reject.accepted, 0);
    ASSERT_EQ(full_reject.rejected, 3);

    auto no_target_extension = ov::genai::dflash_cb::validation_accounting(3, 1, 1);
    ASSERT_FALSE(no_target_extension.target_extended);
    ASSERT_EQ(no_target_extension.accepted, 0);
    ASSERT_EQ(no_target_extension.rejected, 0);
}
