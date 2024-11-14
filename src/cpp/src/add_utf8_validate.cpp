// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "add_utf8_validate.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/assign.hpp"


using namespace ov;
using namespace ov::op;

bool ov::genai::AddUTF8Validate::run_on_model(const std::shared_ptr<ov::Model>& model) {

    std::shared_ptr<ov::Node> combine_seg_node;
    for (auto node: model->get_ordered_ops()) {
        if (strcmp(node->get_type_info().name, "CombineSegments") == 0) {
            combine_seg_node = node;
        }
    }
    
    return true;
}
