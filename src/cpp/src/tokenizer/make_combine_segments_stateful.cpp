// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "make_combine_segments_stateful.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/select.hpp"
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
