// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>

#include "continuous_batching/cache/hidden_state_cache.hpp"
#include "sequence_group.hpp"

namespace {

constexpr size_t kBlockSize = 2;
constexpr size_t kHiddenSize = 4;
constexpr size_t kBlockBytes = kBlockSize * kHiddenSize * sizeof(float);

ov::Tensor make_row(float value) {
    ov::Tensor row(ov::element::f32, {1, kHiddenSize});
    std::fill_n(row.data<float>(), kHiddenSize, value);
    return row;
}

std::vector<ov::Tensor> make_block(float value) {
    std::vector<ov::Tensor> rows;
    for (size_t i = 0; i < kBlockSize; ++i) {
        rows.push_back(make_row(value + static_cast<float>(i)));
    }
    return rows;
}

ov::genai::Sequence::Ptr make_accumulating_sequence() {
    const auto group =
        std::make_shared<ov::genai::SequenceGroup>(0, ov::genai::TokenIds{1, 2, 3}, ov::genai::GenerationConfig{});
    ov::genai::Sequence::Ptr sequence = (*group)[0];
    sequence->set_accumulate_hidden_states(true);
    return sequence;
}

std::vector<float> first_values(const std::vector<ov::Tensor>& rows) {
    std::vector<float> values;
    for (const ov::Tensor& row : rows) {
        values.push_back(row.data<const float>()[0]);
    }
    return values;
}

}  // namespace

TEST(HiddenStateCache, GetPrefixReturnsRowsOfLeadingCachedBlocksInOrder) {
    ov::genai::HiddenStateCache cache;
    cache.put(1, make_block(10.f));
    cache.put(2, make_block(20.f));

    const auto rows = cache.get_prefix({1, 2, 3});

    EXPECT_EQ(first_values(rows), (std::vector<float>{10.f, 11.f, 20.f, 21.f}));
}

TEST(HiddenStateCache, GetPrefixStopsAtFirstMissingBlock) {
    ov::genai::HiddenStateCache cache;
    cache.put(1, make_block(10.f));
    cache.put(3, make_block(30.f));

    EXPECT_EQ(first_values(cache.get_prefix({1, 2, 3})), (std::vector<float>{10.f, 11.f}));
    EXPECT_TRUE(cache.get_prefix({2, 3}).empty());
}

TEST(HiddenStateCache, EvictsLeastRecentlyUsedBlockOverBudget) {
    ov::genai::HiddenStateCache cache(2 * kBlockBytes);
    cache.put(1, make_block(10.f));
    cache.put(2, make_block(20.f));
    cache.get_prefix({1});

    cache.put(3, make_block(30.f));

    EXPECT_TRUE(cache.get_prefix({2}).empty());
    EXPECT_EQ(cache.get_prefix({1}).size(), kBlockSize);
    EXPECT_EQ(cache.get_prefix({3}).size(), kBlockSize);
    EXPECT_EQ(cache.used_bytes(), 2 * kBlockBytes);
}

TEST(HiddenStateCache, PutOfCachedHashKeepsStoredRows) {
    ov::genai::HiddenStateCache cache;
    cache.put(1, make_block(10.f));

    cache.put(1, make_block(99.f));

    EXPECT_EQ(first_values(cache.get_prefix({1})), (std::vector<float>{10.f, 11.f}));
    EXPECT_EQ(cache.used_bytes(), kBlockBytes);
}

TEST(HiddenStateCache, BlockLargerThanBudgetIsNotStored) {
    ov::genai::HiddenStateCache cache(kBlockBytes - 1);

    cache.put(1, make_block(10.f));

    EXPECT_TRUE(cache.get_prefix({1}).empty());
    EXPECT_EQ(cache.used_bytes(), 0);
}

TEST(IntermediateHiddenStates, PositionComputedAgainAfterPreemptionReplacesItsState) {
    const auto sequence = make_accumulating_sequence();
    for (size_t position = 0; position < 3; ++position) {
        sequence->set_intermediate_hidden_state(position, make_row(static_cast<float>(position)));
    }

    sequence->set_intermediate_hidden_state(1, make_row(10.f));
    sequence->set_intermediate_hidden_state(2, make_row(20.f));

    EXPECT_EQ(first_values(sequence->get_all_intermediate_hidden_states()), (std::vector<float>{0.f, 10.f, 20.f}));
}

TEST(IntermediateHiddenStates, ComputedStatesFollowRestoredPrefix) {
    const auto sequence = make_accumulating_sequence();
    sequence->prefill_intermediate_hidden_states(make_block(0.f));

    sequence->set_intermediate_hidden_state(kBlockSize, make_row(5.f));

    EXPECT_EQ(first_values(sequence->get_all_intermediate_hidden_states()), (std::vector<float>{0.f, 1.f, 5.f}));
}

TEST(IntermediateHiddenStates, StateBeyondKnownPrefixIsDropped) {
    const auto sequence = make_accumulating_sequence();

    sequence->set_intermediate_hidden_state(4, make_row(4.f));

    EXPECT_TRUE(sequence->get_all_intermediate_hidden_states().empty());
}

TEST(IntermediateHiddenStates, PrefillAfterComputedStatesThrows) {
    const auto sequence = make_accumulating_sequence();
    sequence->set_intermediate_hidden_state(0, make_row(0.f));

    EXPECT_THROW(sequence->prefill_intermediate_hidden_states(make_block(1.f)), ov::Exception);
}
