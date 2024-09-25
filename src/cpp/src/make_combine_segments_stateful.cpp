// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/make_combine_segments_stateful.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/read_value.hpp"
#include <openvino/pass/graph_rewrite.hpp>

namespace {
    const std::string ADD_SPECIAL_TOKENS_VAR_ID = "ADD_SPECIAL_TOKENS_VAL";
}

bool MakeCombineSegmentsSatateful::run_on_model(const std::shared_ptr<ov::Model>& model) {  
    std::shared_ptr<ov::Node> combine_seg_node;
    for (auto node: model->get_ordered_ops()) {
        if (strcmp(node->get_type_info().name, "CombineSegments") == 0) {
            combine_seg_node = node;
        }
    }
    if (!combine_seg_node) {
        return false;
    }
    
    auto input_1 = std::dynamic_pointer_cast<ov::op::v0::Constant>(combine_seg_node->get_input_node_shared_ptr(1));
    if (!input_1 || ov::PartialShape{input_1->get_shape()} != ov::PartialShape{} || input_1->get_element_type() != ov::element::i32) {
        return false;
    }

    ov::op::util::VariableInfo var_info{ov::Shape{}, ov::element::boolean, ADD_SPECIAL_TOKENS_VAR_ID};
    auto variable = std::make_shared<ov::op::util::Variable>(var_info);
    
    auto new_constant = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{}, std::vector{true});
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(new_constant, variable);
    auto assign = std::make_shared<ov::op::v6::Assign>(read_value, variable);
    
    auto zero_constant = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector{0});
    auto mul_node = std::make_shared<ov::op::v1::Select>(read_value, input_1, zero_constant);
    combine_seg_node->input(1).replace_source_output(mul_node->output(0));
    
    model->add_sinks({assign});
    model->add_variables({variable});
    return true;
}
