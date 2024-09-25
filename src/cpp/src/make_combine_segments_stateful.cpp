// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/make_combine_segments_stateful.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/read_value.hpp"
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <openvino/pass/pattern/matcher.hpp>
#include <openvino/pass/graph_rewrite.hpp>
#include <openvino/op/op.hpp>

namespace {
    const std::string ADD_SPECIAL_TOKENS_VAR_ID = "ADD_SPECIAL_TOKENS_VAL";
}


/* 
* We get graph either in the format of, or 

*   +--------------+  +--------+  +------------------+
*   |  DefaultMode |  |  ends  |  | const value = 0  |
*   +--------------+  +--------+  +------------------+
*              \        |         /   
*               \       |        / 
*                v      v       v  
*                 +--------------+ 
*                 |    Select    | 
*                 +--------------+ 
*                        | 
*                        v 
*           +-------------------------+
*           |      CombineSegments    |
*           +-------------------------+
*
* or as a if model was not 
*
*                 +------------+ 
*                 |    ends    | 
*                 +------------+ 
*                        | 
*                        v 
*           +-------------------------+
*           |      CombineSegments    |
*           +-------------------------+
*/
bool MakeCombineSegmentsSatateful::run_on_model(const std::shared_ptr<ov::Model>& model) {

    auto default_mode_const = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto ends_const = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto zero_const = ov::pass::pattern::wrap_type<ov::op::v0::Constant>([](const ov::Output<ov::Node>& node) {
        auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(node.get_node_shared_ptr());
        if (!const_node)
            return false;
        if (const_node->get_element_type() != ov::element::i32)
            return false;
        if (const_node->get_vector<int32_t>() == std::vector<int32_t>{0})
            return false;
        return  true;
    });

    auto select_node = ov::pass::pattern::wrap_type<ov::op::v1::Select>({default_mode_const, ends_const, zero_const});
    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(select_node, "SelectCombineSegmentsMatcher");
    

    std::shared_ptr<ov::Node> combine_seg_node;
    // matcher->match(combine_seg_node->input_value(1)) == true;
    
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

    // todo: read this constant from compilation flag.
    auto new_constant = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector{1});

    ov::op::util::VariableInfo var_info{ov::Shape{}, ov::element::i32, ADD_SPECIAL_TOKENS_VAR_ID};
    auto variable = std::make_shared<ov::op::util::Variable>(var_info);
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(new_constant, variable);
    auto assign = std::make_shared<ov::op::v6::Assign>(read_value, variable);
    
    auto mul_node = std::make_shared<ov::op::v1::Multiply>(read_value, input_1);
    combine_seg_node->input(1).replace_source_output(mul_node->output(0));
    model->add_sinks({assign});
    model->add_variables({variable});
    return true;
}
