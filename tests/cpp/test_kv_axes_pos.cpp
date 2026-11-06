// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/variable.hpp"
#include "utils.hpp"

namespace {

struct StateSpec {
    std::string id;
    ov::PartialShape init_shape;
};

std::shared_ptr<ov::Model> make_stateful_model(const std::vector<StateSpec>& states) {
    ov::ParameterVector parameters;
    ov::SinkVector sinks;
    for (const auto& state : states) {
        auto init = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, state.init_shape);
        parameters.push_back(init);
        auto variable = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{state.init_shape, ov::element::f32, state.id});
        auto read_value = std::make_shared<ov::op::v6::ReadValue>(init, variable);
        sinks.push_back(std::make_shared<ov::op::v6::Assign>(read_value, variable));
    }
    auto result = std::make_shared<ov::op::v0::Result>(parameters.front());
    return std::make_shared<ov::Model>(ov::ResultVector{result}, sinks, parameters);
}

const StateSpec ssm_state{"ssm_state_l0", {-1, 48, 128, 128}};
// [batch, seq_len, heads, head_dim] layout, as produced by the GGUF frontend
const StateSpec kv_state{"cache_k_l3", {-1, 0, 4, 256}};

}  // namespace

TEST(GetKVAxesPos, KVCacheOnlyUsesZeroLengthAxis) {
    const auto model = make_stateful_model({{"past_key_values.0.key", {-1, 4, 0, 64}}});
    const auto kv_pos = ov::genai::utils::get_kv_axes_pos(model);
    EXPECT_EQ(kv_pos.batch, 0u);
    EXPECT_EQ(kv_pos.seq_len, 2u);
}

TEST(GetKVAxesPos, SeqLenAxisIsOne) {
    const auto model = make_stateful_model({kv_state});
    const auto kv_pos = ov::genai::utils::get_kv_axes_pos(model);
    EXPECT_EQ(kv_pos.batch, 0u);
    EXPECT_EQ(kv_pos.seq_len, 1u);
}

TEST(GetKVAxesPos, LinearAttentionStateBeforeKVCacheIsSkipped) {
    const auto model = make_stateful_model({ssm_state, kv_state});
    const auto kv_pos = ov::genai::utils::get_kv_axes_pos(model);
    EXPECT_EQ(kv_pos.batch, 0u);
    EXPECT_EQ(kv_pos.seq_len, 1u);
}

TEST(GetKVAxesPos, LinearAttentionStateAfterKVCacheIsSkipped) {
    const auto model = make_stateful_model({kv_state, ssm_state});
    const auto kv_pos = ov::genai::utils::get_kv_axes_pos(model);
    EXPECT_EQ(kv_pos.batch, 0u);
    EXPECT_EQ(kv_pos.seq_len, 1u);
}
