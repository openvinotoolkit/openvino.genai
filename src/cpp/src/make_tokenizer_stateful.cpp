// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "make_tokenizer_stateful.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/assign.hpp"


using namespace ov;
using namespace ov::op;

bool ov::genai::MakeCombineSegmentsSatateful::run_on_model(const std::shared_ptr<ov::Model>& model) {

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
