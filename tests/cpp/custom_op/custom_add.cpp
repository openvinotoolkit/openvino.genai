// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom_add.hpp"

using namespace TemplateExtension;

MyAdd::MyAdd(const ov::OutputVector &args) : Op(args)
{
    constructor_validate_and_infer_types();
    // this->_bias = bias;
}

void MyAdd::validate_and_infer_types()
{
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<ov::Node> MyAdd::clone_with_new_inputs(const ov::OutputVector &new_args) const
{
    OPENVINO_ASSERT(new_args.size() == 2, "Incorrect number of new arguments");
    return std::make_shared<MyAdd>(new_args);
}

bool MyAdd::visit_attributes(ov::AttributeVisitor &visitor)
{
    // visitor.on_attribute("bias", this->_bias);
    return true;
}

bool MyAdd::evaluate(ov::TensorVector &outputs, const ov::TensorVector &inputs) const
{
    std::cout << "== MyAdd::evaluate is called." << std::endl;
    float *inpData = reinterpret_cast<float *>(const_cast<void*>(inputs[0].data()));
    if (inputs[1].get_element_type() != ov::element::f32)
        OPENVINO_THROW("Unexpected bias type: " + inputs[1].get_element_type().to_string());
    float *pBias = reinterpret_cast<float *>(const_cast<void*>(inputs[1].data()));

    const auto &in = inputs[0];
    auto &out = outputs[0];

    out.set_shape(in.get_shape());
    auto total = in.get_size();
    if (in.get_element_type() == ov::element::f32)
    {
        auto *ptr_in = reinterpret_cast<float *>(const_cast<void*>(in.data()));
        auto *ptr_out = reinterpret_cast<float *>(out.data());
        for (size_t i = 0; i < total; i++)
        {
            ptr_out[i] = ptr_in[i] + pBias[0];
        }
    }
    else
    {
        std::cout << "Error: Not implemented for data type: " << in.get_element_type() << std::endl;
    }

    return true;
}

bool MyAdd::has_evaluate() const
{
    return true;
}
