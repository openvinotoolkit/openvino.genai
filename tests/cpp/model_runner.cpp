// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <numeric>

#include "openvino/runtime/core.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

#include "visual_language/inputs_embedder.hpp"
#include "continuous_batching/model_runner.hpp"
#include "continuous_batching/cache/block_manager.hpp"
#include "utils.hpp"

using namespace ov::genai;

namespace {

std::shared_ptr<ov::Model> create_dummy_la_paging_model() {
    ov::ParameterVector params;

    auto add_input = [&](const std::string& name, const ov::element::Type& type, const ov::PartialShape& pshape) {
        auto p = std::make_shared<ov::op::v0::Parameter>(type, pshape);
        p->output(0).get_tensor().set_names({name});
        params.push_back(p);
        return p;
    };

    add_input("input_ids", ov::element::i64, ov::PartialShape::dynamic(1));
    add_input("position_ids", ov::element::i64, ov::PartialShape::dynamic(1));
    add_input("past_lens", ov::element::i32, ov::PartialShape::dynamic(1));
    add_input("subsequence_begins", ov::element::i32, ov::PartialShape::dynamic(1));
    add_input("block_indices", ov::element::i32, ov::PartialShape::dynamic(1));
    add_input("block_indices_begins", ov::element::i32, ov::PartialShape::dynamic(1));
    add_input("max_context_len", ov::element::i32, ov::PartialShape{});

    // Linear attention paging inputs with "paged_conv_" prefix.
    add_input("paged_conv_block_indices", ov::element::i32, ov::PartialShape::dynamic(1));
    add_input("paged_conv_block_indices_begins", ov::element::i32, ov::PartialShape::dynamic(1));
    add_input("paged_conv_past_lens", ov::element::i32, ov::PartialShape::dynamic(1));
    add_input("paged_conv_cache_interval", ov::element::i32, ov::PartialShape::dynamic(1));

    auto logits = ov::op::v0::Constant::create(ov::element::f32, {1, 1, 8}, {0.0f});
    logits->output(0).get_tensor().set_names({"logits"});

    return std::make_shared<ov::Model>(ov::OutputVector{logits}, params);
}

Scheduler::Output make_output_for_single_sequence(uint64_t seq_id,
                                                  size_t num_scheduled_tokens,
                                                  const BlocksPerLayer& la_blocks) {
    Scheduler::Output out;
    out.m_scheduled_sequence_groups_ids = {0};
    out.m_total_num_scheduled_tokens = num_scheduled_tokens;

    // ModelRunner expects one KV block table per decoder layer. Use a single layer for tests.
    out.m_block_tables[seq_id] = {BlocksPerLayer{std::make_shared<CacheBlock>(0)}};
    out.m_linear_attention_block_table[seq_id] = la_blocks;
    return out;
}

std::vector<int32_t> tensor_to_i32_vector(const ov::Tensor& tensor) {
    const int32_t* data = tensor.data<const int32_t>();
    return std::vector<int32_t>(data, data + tensor.get_size());
}

}  // namespace

TEST(TestModelRunnerLinearAttentionPaging, prefill_uses_read_plus_interval_write_blocks_layout) {
    ov::Core core;
    ov::InferRequest request = core.compile_model(create_dummy_la_paging_model(), "CPU").create_infer_request();
    ModelRunner runner(request, /*block_size=*/4, /*num_decoder_layers=*/1);

    std::vector<uint64_t> prompt_tokens(260);
    std::iota(prompt_tokens.begin(), prompt_tokens.end(), 0);
    auto sequence_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {prompt_tokens.size()}, prompt_tokens.data()),
        utils::get_greedy_config());
    sequence_group->schedule_tokens(prompt_tokens.size());

    const auto seq_id = sequence_group->get_running_sequences()[0]->get_id();

    // Target layout for prefill with cache_interval=128 and scheduled=260:
    // [read_init, write_0, write_1, write_2].
    auto out = make_output_for_single_sequence(
        seq_id,
        prompt_tokens.size(),
        BlocksPerLayer{
            std::make_shared<CacheBlock>(10),
            std::make_shared<CacheBlock>(11),
            std::make_shared<CacheBlock>(12),
            std::make_shared<CacheBlock>(13)});

    std::ignore = runner.forward({sequence_group}, out);

    auto ireq = runner.get_infer_request();

    const auto la_indices = ireq.get_tensor("paged_conv_block_indices");
    const auto la_begins = ireq.get_tensor("paged_conv_block_indices_begins");
    const auto la_past_lens = ireq.get_tensor("paged_conv_past_lens");
    const auto la_interval = ireq.get_tensor("paged_conv_cache_interval");

    EXPECT_EQ(tensor_to_i32_vector(la_indices), (std::vector<int32_t>{10, 11, 12, 13}));
    EXPECT_EQ(tensor_to_i32_vector(la_begins), (std::vector<int32_t>{0, 4}));
    EXPECT_EQ(tensor_to_i32_vector(la_past_lens), (std::vector<int32_t>{0}));
    EXPECT_EQ(tensor_to_i32_vector(la_interval), (std::vector<int32_t>{128}));
}

TEST(TestModelRunnerLinearAttentionPaging, generation_uses_read_and_write_blocks_and_total_processed_past_lens) {
    ov::Core core;
    ov::InferRequest request = core.compile_model(create_dummy_la_paging_model(), "CPU").create_infer_request();
    ModelRunner runner(request, /*block_size=*/4, /*num_decoder_layers=*/1);

    std::vector<uint64_t> prompt_tokens = {0, 1, 2, 3};
    auto sequence_group = std::make_shared<SequenceGroup>(
        0,
        ov::Tensor(ov::element::i64, {prompt_tokens.size()}, prompt_tokens.data()),
        utils::get_greedy_config());

    // Simulate generation step after prompt processing.
    sequence_group->update_processed_tokens_num(prompt_tokens.size());
    auto running_sequence = sequence_group->get_running_sequences()[0];
    running_sequence->append_token(42, 0.9f);
    sequence_group->schedule_tokens(1);

    const auto seq_id = running_sequence->get_id();
    auto out = make_output_for_single_sequence(
        seq_id,
        /*num_scheduled_tokens=*/1,
        BlocksPerLayer{
            std::make_shared<CacheBlock>(20),
            std::make_shared<CacheBlock>(21)});

    std::ignore = runner.forward({sequence_group}, out);

    auto ireq = runner.get_infer_request();

    const auto la_indices = ireq.get_tensor("paged_conv_block_indices");
    const auto la_begins = ireq.get_tensor("paged_conv_block_indices_begins");
    const auto la_past_lens = ireq.get_tensor("paged_conv_past_lens");
    const auto la_interval = ireq.get_tensor("paged_conv_cache_interval");

    EXPECT_EQ(tensor_to_i32_vector(la_indices), (std::vector<int32_t>{20, 21}));
    EXPECT_EQ(tensor_to_i32_vector(la_begins), (std::vector<int32_t>{0, 2}));
    EXPECT_EQ(tensor_to_i32_vector(la_past_lens), (std::vector<int32_t>{4}));
    EXPECT_EQ(tensor_to_i32_vector(la_interval), (std::vector<int32_t>{128}));
}
