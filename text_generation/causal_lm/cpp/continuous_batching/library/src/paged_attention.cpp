// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention.hpp"

#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/runtime/core.hpp"

PagedAttention::PagedAttention(const ov::OutputVector& inputs)
    : ov::op::Op(inputs) {
    constructor_validate_and_infer_types();
}

PagedAttention::PagedAttention(const ov::Output<ov::Node>& query,
                                                  const ov::Output<ov::Node>& key,
                                                  const ov::Output<ov::Node>& value,
                                                  const ov::Output<ov::Node>& key_cache,
                                                  const ov::Output<ov::Node>& value_cache,
                                                  // start of arguments from InputMetadata
                                                  const ov::Output<ov::Node>& is_prompt,
                                                  const ov::Output<ov::Node>& slot_mapping,
                                            //    const ov::Output<ov::Node>& prompt_lens,
                                            //    const ov::Output<ov::Node>& max_seq_len,
                                            //    const ov::Output<ov::Node>& start_loc,
                                                  const ov::Output<ov::Node>& max_context_len,
                                                  const ov::Output<ov::Node>& context_lens,
                                                  const ov::Output<ov::Node>& block_tables,
                                            //    const ov::Output<ov::Node>& use_cuda_graph,
                                            //    const ov::Output<ov::Node>& attn_bias
                                                  // end of arguments from InputMetadata
                                                  const ov::Output<ov::Node>& scale,
                                                  const ov::Output<ov::Node>& alibi_slopes,
                                                  const ov::Output<ov::Node>& sliding_window)
    : PagedAttention(ov::OutputVector{query, key, value, key_cache, value_cache,
                      is_prompt, slot_mapping, max_context_len, context_lens, block_tables, scale, alibi_slopes, sliding_window
    }) {}

void PagedAttention::validate_and_infer_types() {
    // value_cache: shape = [num_blocks, num_kv_heads, head_size, block_size]
    auto value_cache_shape = get_input_partial_shape(4);
    // m_num_kv_heads = value_cache_shape[1];
    // m_head_size = value_cache_shape[2];
    // m_block_size = value_cache_shape[3];
    NODE_VALIDATION_CHECK(this,
        value_cache_shape.size() == 4,
        "Value cache shape must be 4 dims");

    // key_cache: shape [num_blocks, num_kv_heads, head_size/x, block_size, x]
    auto key_cache_shape = get_input_partial_shape(3);
    NODE_VALIDATION_CHECK(this,
        value_cache_shape.size() == 4,
        // value_cache_shape[0] == key_cache_shape[0] && // num_blocks
        // key_cache_shape[1] == m_num_kv_heads &&
        // key_cache_shape[2] * key_cache_shape[4] == m_head_size &&
        // m_block_size == key_cache_shape[3], // block_size,
        "Key cache shape must be 4 dims");

    // query: shape [batch_size, seq_len, num_heads * head_size]
    auto query_type = get_input_element_type(0);
    auto query_shape = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this,
        // query_type.is_real() &&
        query_shape.size() == 3,
        // query_shape[2] == m_num_heads * m_head_size,
        "Query type must be real, shape must be like [batch_size, seq_len, num_heads * head_size]. ",
        "Got element type ", query_type, ", shape ", query_shape);

    // key: shape [batch_size, seq_len, num_kv_heads * head_size]
    auto key_type = get_input_element_type(1);
    auto key_shape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this,
        query_type == key_type &&
        key_shape.size() == 3,
        "Key type must be the same as query, shape must be the same as query. "
        "Got element type ", key_type, ", shape ", key_shape);

    // value: shape [batch_size, seq_len, num_kv_heads * head_size]
    auto value_type = get_input_element_type(2);
    auto value_shape = get_input_partial_shape(2);
    // NODE_VALIDATION_CHECK(this,
    //     key_type == value_type &&
    //     key_shape == value_shape, "Value type must be the same as key, shape must be the same as key. "
    //     "\nGot element type value ", value_type, ", shape ", value_shape,
    //     "\nGot element type ", key_type, ", shape ", key_shape);

    // is_prompt: boolean scalar
    NODE_VALIDATION_CHECK(this,
        // get_input_element_type(5) == ov::element::boolean &&
        get_input_shape(5) == ov::Shape({}),
        "is_prompt validation failed. ",
        "Got element type ", get_input_element_type(5), ", shape ", get_input_shape(5));

    // slot_mapping: shape [batch_size, max_context_len]
    auto slot_mapping_shape = get_input_partial_shape(6);
    NODE_VALIDATION_CHECK(this,
        // get_input_element_type(6) == ov::element::i64 &&
        slot_mapping_shape.size() == 2,
        "slot_mapping validation failed. ",
        "Got element type ", get_input_element_type(6), ", shape ", slot_mapping_shape);

    // max_context_len: integer scalar
    NODE_VALIDATION_CHECK(this,
        // get_input_element_type(7) == ov::element::i32 &&
        get_input_shape(7) == ov::Shape({}),
        "max_context_len validation failed. ",
        "Got element type ", get_input_element_type(7), ", shape ", get_input_shape(7));

    // context_lens: shape [batch_size]
    auto context_lens_shape = get_input_partial_shape(8);
    NODE_VALIDATION_CHECK(this,
        // get_input_element_type(8) == ov::element::i32 &&
        context_lens_shape.size() == 1,
        "context_lens validation failed. ",
        "Got element type ", get_input_element_type(8), ", shape ", context_lens_shape);

    // block_tables: shape [batch_size, max_block_per_request]
    NODE_VALIDATION_CHECK(this,
        // get_input_element_type(9) == ov::element::i32 &&
        get_input_partial_shape(9).size() == 2,
        "block_tables validation failed. ",
        "Got element type ", get_input_element_type(9), ", shape ", get_input_partial_shape(9));

    // scale: float scalar
    NODE_VALIDATION_CHECK(this,
        // get_input_element_type(10) == ov::element::f32 &&
        get_input_shape(10) == ov::Shape({}),
        "block_tables validation failed. ",
        "Got element type ", get_input_element_type(10), ", shape ", get_input_shape(10));

    // alibi_slopes: 1D float tensor
    NODE_VALIDATION_CHECK(this,
        // get_input_element_type(11) == ov::element::f32 &&
        get_input_partial_shape(11).rank().get_length() == 1,
        "alibi_slopes should be a 1D float tensor. ",
        "Got element type ", get_input_element_type(11), ", shape ", get_input_partial_shape(11));

    // sliding_window: int scalar
    NODE_VALIDATION_CHECK(this,
        // get_input_element_type(12) == ov::element::i32 &&
        get_input_partial_shape(12).rank().get_length() == 0,
        "sliding_window argument should be an i32 scalar. ",
        "Got element type ", get_input_element_type(12), ", shape ", get_input_partial_shape(12));

    set_output_type(0, query_type, query_shape);
}

std::shared_ptr<ov::Node> PagedAttention::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<PagedAttention>(new_args);
}

bool PagedAttention::has_evaluate() const {
    return get_input_element_type(0) == ov::element::f32;
}

bool PagedAttention::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    return false;
}
