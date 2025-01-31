// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "make_tokenizer_stateful.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/assign.hpp"


using namespace ov;
using namespace ov::op;

bool ov::genai::MakeAddSpecialTokensSatateful::run_on_model(const std::shared_ptr<ov::Model>& model) {
    std::shared_ptr<ov::Node> combine_seg_node;
    for (auto node: model->get_ordered_ops()) {
        if (strcmp(node->get_type_info().name, "CombineSegments") == 0) {
            combine_seg_node = node;
        }
    }
    if (!combine_seg_node || combine_seg_node->input_value(1).get_element_type() != ov::element::i32) {
        return false;
    }
    
    std::shared_ptr<v0::Constant> input_1_const = std::dynamic_pointer_cast<v0::Constant>(combine_seg_node->get_input_node_shared_ptr(1));
    if (!input_1_const) {
        return false;
    }
    
    op::util::VariableInfo var_info{ov::Shape{}, ov::element::boolean, ADD_SPECIAL_TOKENS_VAR_ID};
    auto variable = std::make_shared<op::util::Variable>(var_info);

    // Default mode is add_special_tokens.
    auto default_mode_const = std::make_shared<v0::Constant>(ov::element::boolean, ov::Shape{}, std::vector{true});
    auto read_value = std::make_shared<v6::ReadValue>(default_mode_const, variable);
    auto zero_constant = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{}, std::vector{0});
    auto select_node = std::make_shared<v1::Select>(read_value, input_1_const, zero_constant);
    combine_seg_node->input(1).replace_source_output(select_node->output(0));

    auto assign = std::make_shared<v6::Assign>(read_value, variable);
    
    model->add_sinks({assign});
    model->add_variables({variable});
    return true;
}


bool ov::genai::MakeVocabDecoderSatateful::run_on_model(const std::shared_ptr<ov::Model>& model) {
    std::shared_ptr<ov::Node> vocab_decoder_node;
    for (auto node: model->get_ordered_ops()) {
        if (strcmp(node->get_type_info().name, "VocabDecoder") == 0)
            vocab_decoder_node = node;
    }

    if (!vocab_decoder_node || vocab_decoder_node->get_input_size() < 5)
        return false;
    if (!vocab_decoder_node->input_value(4).get_element_type().is_integral_number())
        return false;
    
    std::shared_ptr<v0::Constant> skip_tokens_const = std::dynamic_pointer_cast<v0::Constant>(vocab_decoder_node->get_input_node_shared_ptr(4));
    std::shared_ptr<v8::Slice> skip_tokens_slice = std::dynamic_pointer_cast<v8::Slice>(vocab_decoder_node->get_input_node_shared_ptr(4));
    if (!skip_tokens_const && !skip_tokens_slice)
        return false;

    auto start_const = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector{0});
    auto int_max_const = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector{std::numeric_limits<int>::max()});
    auto one_const = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector{1});
    
    // By default, INT_MAX will multiply with 1 and all skip_tokens will be selected.
    op::util::VariableInfo var_info{ov::Shape{1}, ov::element::i32, SKIP_SPECIAL_TOKENS_VAR_ID};
    auto variable = std::make_shared<op::util::Variable>(var_info);
    auto read_value = std::make_shared<v6::ReadValue>(one_const, variable);
    // if flag is set, then slice up to the int_max which means skip all tokens.
    auto stop = std::make_shared<v1::Multiply>(int_max_const, read_value);

    // If already has slice just replace the stop input.
    if (skip_tokens_slice) {
        skip_tokens_slice->input(2).replace_source_output(stop);
    } else {
        std::shared_ptr<v8::Slice> slice_node = std::make_shared<v8::Slice>(skip_tokens_const, start_const, stop, one_const);
        vocab_decoder_node->input(4).replace_source_output(slice_node->output(0));
    }
    
    auto assign = std::make_shared<v6::Assign>(read_value, variable);
    model->add_sinks({assign});
    model->add_variables({variable});
    return true;
}


bool ov::genai::MakePaddingSatateful::run_on_model(const std::shared_ptr<ov::Model>& model) {

    std::shared_ptr<ov::Node> ragged_to_dense_node;
    for (auto node: model->get_ordered_ops()) {
        if (strcmp(node->get_type_info().name, "RaggedToDense") == 0) {
            ragged_to_dense_node = node;
        }
    }
    
    if (!ragged_to_dense_node || ragged_to_dense_node->input_value(3).get_element_type() != ov::element::i32) {
        return false;
    }
   
    op::util::VariableInfo var_info{ov::Shape{1}, ov::element::i32, MAX_PAD_LENGTH_VAR_ID};
    auto variable = std::make_shared<op::util::Variable>(var_info);

    // By default pad to the max length.
    auto default_pad_length = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector{0});
    auto read_max_pad_value = std::make_shared<v6::ReadValue>(default_pad_length, variable);
    auto max_op = std::make_shared<v1::Maximum>(ragged_to_dense_node->input_value(3), read_max_pad_value);
    // TODO: handle when max_length is already set during tokenzation
    ragged_to_dense_node->input(3).replace_source_output(max_op->output(0));

    auto assign = std::make_shared<v6::Assign>(read_max_pad_value, variable);
    model->add_sinks({assign});
    model->add_variables({variable});
    return true;
}


bool ov::genai::MakeTruncationSatateful::run_on_model(const std::shared_ptr<ov::Model>& model) {

    std::shared_ptr<ov::Node> combine_segments_node;
    for (auto node: model->get_ordered_ops()) {
        if (strcmp(node->get_type_info().name, "CombineSegments") == 0) {
            combine_segments_node = node;
        }
    }
    
    if (!combine_segments_node || combine_segments_node->input_value(4).get_element_type() != ov::element::i32) {
        return false;
    }
    
    std::shared_ptr<Node> add_or_sub_node = combine_segments_node->input_value(4).get_node_shared_ptr();
    // If Add then it's a right truncation, if Subtract then it's a left truncation.
    if (!ov::as_type_ptr<v1::Add>(add_or_sub_node) && !ov::as_type_ptr<v1::Subtract>(add_or_sub_node)) {
        // Exit if it's neither, because in that case it's not a truncation.
        return false;
    }

    // auto pattern_2 = ov::pass::pattern::wrap_type<ov::op::v0::Constant>(ov::pass::pattern::rank_equals(1));
    // auto unsqueeze = ov::pass::pattern::wrap_type<ov::op::v1::Reshape, ov::op::v0::Unsqueeze>({cell, pattern_2});
    // ov::pass::pattern::Matcher matcher(unsqueeze);

    // Minimum between max_length and length of token sequence.
    auto min_node = ov::as_type_ptr<v1::Minimum>(add_or_sub_node->get_input_node_shared_ptr(1));
    if (!min_node) { return false; }
    
    // Node which subtracts from max_truncation_length number of added_tokens.
    auto sub_node = ov::as_type_ptr<v1::Subtract>(min_node->get_input_node_shared_ptr(1));
    if (!sub_node) { return false; }

    // max_truncation_length constant containing final length at the end of pipeline.
    auto const_node = ov::as_type_ptr<v0::Constant>(sub_node->get_input_node_shared_ptr(0));
    if (!const_node) { return false; }

    op::util::VariableInfo var_info{const_node->get_output_shape(0), const_node->get_output_element_type(0), MAX_TRUNCATION_LENGTH_VAR_ID};
    auto variable = std::make_shared<op::util::Variable>(var_info);
    
    // Save targets before adding new target with ReadValue to avoid recursion.
    auto target_inputs = const_node->output(0).get_target_inputs();
    auto read_trunc_value = std::make_shared<v6::ReadValue>(const_node, variable);
    
    for (auto target_input : target_inputs) {
        target_input.replace_source_output(read_trunc_value->output(0));
    }

    // We need to check if user requested to not add special tokens.
    std::shared_ptr<v6::ReadValue> read_value_spec_tokens;
    for (const auto& sink : model->get_sinks()) {
        // Check if sink accepts input from Assign, and if that't the case get the ReadValus node input.
        if (auto read_value = ov::as_type_ptr<v6::ReadValue>(sink->get_input_node_shared_ptr(0))) {
            if (read_value->get_variable()->get_info().variable_id == ADD_SPECIAL_TOKENS_VAR_ID) {
                read_value_spec_tokens = read_value;
                break;
            }
        }
    }
    
    // Constant which stores number of added_tokens.
    auto num_added_tokens_const = ov::as_type_ptr<v0::Constant>(sub_node->get_input_node_shared_ptr(1));
    // If user requested to not add special tokens in order to correctly calculate 
    // truncation we need to enforce num_added_tokens to 0 regardless the hardcoded value of Constant.
    if (read_value_spec_tokens && num_added_tokens_const) {
        auto zero_constant = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{}, std::vector{0});
        auto select_node = std::make_shared<v1::Select>(read_value_spec_tokens, num_added_tokens_const, zero_constant);
        sub_node->input(1).replace_source_output(select_node->output(0));
    }

    auto assign = std::make_shared<v6::Assign>(read_trunc_value, variable);
    model->add_sinks({assign});
    model->add_variables({variable});
    return true;
}
