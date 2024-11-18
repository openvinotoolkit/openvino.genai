// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "add_utf8_validate.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/core/node.hpp"
#include <memory>
#include "utf8_validate.hpp"
#include "openvino/core/rt_info.hpp"

using namespace ov;
using namespace ov::op;

bool ov::genai::AddUTF8Validate::run_on_model(const std::shared_ptr<ov::Model>& model) {

    std::shared_ptr<ov::Node> str_tensor_pack;
    for (auto node: model->get_ordered_ops()) {
        if (strcmp(node->get_type_info().name, "StringTensorPack") == 0) {
            str_tensor_pack = node;
        }
    }
    
    if (!str_tensor_pack) {
        return false;
    }
    bool replace_mode = true;
    auto utf8_validate = std::make_shared<UTF8Validate>(str_tensor_pack->input_values(), replace_mode);
    OPENVINO_ASSERT(utf8_validate != nullptr, "Couldn't create operation: ", "UTF8Validate");


    for (size_t idx = 0; idx < str_tensor_pack->inputs().size(); idx++) {
        str_tensor_pack->input(idx).replace_source_output(utf8_validate->output(idx));
    }
    ov::copy_runtime_info(str_tensor_pack, utf8_validate);

    return true;
}
