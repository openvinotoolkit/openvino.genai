// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "make_combine_segments_stateful.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/matcher.hpp"

namespace {
    const std::string ADD_SPECIAL_TOKENS_VAR_ID = "add_special_tokens";
}

using namespace ov;
using namespace ov::op;

/* 
* For the newly converted models input(1) to CombineSegments stores default mode,
* in that case we just need to insert ReadValue in between DefaultMode -> Select.
* 
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
* If IR is old, then default mode to add special tokens is true, and we insert 
* the whole new subgraph with Select.
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
    std::shared_ptr<v1::Select> input_1_select = std::dynamic_pointer_cast<v1::Select>(combine_seg_node->get_input_node_shared_ptr(1));
    if (!input_1_const && !input_1_select) {
        return false;
    }
    
    op::util::VariableInfo var_info{ov::Shape{}, ov::element::boolean, ADD_SPECIAL_TOKENS_VAR_ID};
    auto variable = std::make_shared<op::util::Variable>(var_info);

    std::shared_ptr<v6::ReadValue> read_value;
    if (input_1_select) {
        // Select already exists, need just to insert to input(0) ReadValue
        // instead of default mode Const.
        read_value = std::make_shared<v6::ReadValue>(input_1_select->input_value(0), variable);
        input_1_select->input(0).replace_source_output(read_value->output(0));
    } else {
        // If there is end then default mode is add_special_tokens.
        bool add_special_tokens = true;
        auto default_mode_const = std::make_shared<v0::Constant>(ov::element::boolean, ov::Shape{}, std::vector{add_special_tokens});
        read_value = std::make_shared<v6::ReadValue>(default_mode_const, variable);
        auto zero_constant = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{}, std::vector{0});
        auto select_node = std::make_shared<v1::Select>(read_value, input_1_const, zero_constant);
        combine_seg_node->input(1).replace_source_output(select_node->output(0));
    }

    // here need to store.
    auto assign = std::make_shared<v6::Assign>(read_value, variable);
    
    model->add_sinks({assign});
    model->add_variables({variable});
    return true;
}
