// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "custom_add.hpp"

#include <cstdlib>
#include <iostream>

using namespace TemplateExtension;

MyAdd::MyAdd(const ov::OutputVector& args) : Op(args) {
    constructor_validate_and_infer_types();
}

void MyAdd::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() == 2, "MyAdd expects exactly 2 inputs: data and bias");

    const ov::element::Type data_type = get_input_element_type(0);
    const ov::element::Type bias_type = get_input_element_type(1);
    OPENVINO_ASSERT(data_type == ov::element::f32, "MyAdd supports only f32 data input, got: ", data_type.to_string());
    OPENVINO_ASSERT(bias_type == data_type,
                    "MyAdd expects bias type to match data type. data type: ",
                    data_type.to_string(),
                    ", bias type: ",
                    bias_type.to_string());

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<ov::Node> MyAdd::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OPENVINO_ASSERT(new_args.size() == 2, "Incorrect number of new arguments");
    return std::make_shared<MyAdd>(new_args);
}

bool MyAdd::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}

bool MyAdd::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
#if defined(_WIN32)
    const int set_env_result = _putenv_s("EXTENSION_LIB_CALLED", "1");
#else
    const int set_env_result = setenv("EXTENSION_LIB_CALLED", "1", 1);
#endif
    OPENVINO_ASSERT(inputs.size() >= 2, "MyAdd evaluate expects at least 2 input tensors");
    OPENVINO_ASSERT(outputs.size() >= 1, "MyAdd evaluate expects at least 1 output tensor");

    OPENVINO_ASSERT(inputs[0].get_element_type() == ov::element::f32,
                    "Unexpected data type: ",
                    inputs[0].get_element_type().to_string());
    OPENVINO_ASSERT(inputs[1].get_element_type() == ov::element::f32,
                    "Unexpected bias type: ",
                    inputs[1].get_element_type().to_string());
    const float* pBias = static_cast<const float*>(inputs[1].data());

    const auto& in = inputs[0];
    auto& out = outputs[0];

    out.set_shape(in.get_shape());
    auto total = in.get_size();
    const float* ptr_in = static_cast<const float*>(in.data());
    float* ptr_out = static_cast<float*>(out.data());
    for (size_t i = 0; i < total; i++) {
        ptr_out[i] = ptr_in[i] + pBias[0];
    }

    return true;
}

bool MyAdd::has_evaluate() const {
    return true;
}
