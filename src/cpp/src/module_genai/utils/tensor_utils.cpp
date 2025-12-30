// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor_utils.hpp"

namespace ov::genai::module::tensor_utils {

std::vector<ov::Tensor> split(const ov::Tensor &tensor) {
    std::vector<ov::Tensor> outputs;
    ov::Shape tensor_shape = tensor.get_shape();
    outputs.reserve(tensor_shape[0]);
    ov::Shape single_output_shape(tensor_shape.begin() + 1, tensor_shape.end());
    const uint8_t *tensor_bytes = static_cast<const uint8_t *>(tensor.data());
    for (size_t i = 0; i < tensor_shape[0]; ++i) {
        ov::Tensor single_output(tensor.get_element_type(), single_output_shape);
        uint8_t *tensor_data = static_cast<uint8_t *>(single_output.data());
        std::memcpy(tensor_data, tensor_bytes + i * single_output.get_byte_size(), single_output.get_byte_size());
        outputs.push_back(single_output);
    }
    return outputs;
}

ov::Tensor unsqueeze(const ov::Tensor &tensor, size_t dim) {
    ov::Shape original_shape = tensor.get_shape();
    OPENVINO_ASSERT(original_shape.size() > dim, "Cannot unsqueeze dimension " + std::to_string(dim) + " of shape " + original_shape.to_string());
    ov::Shape new_shape = original_shape;
    new_shape.insert(new_shape.begin() + dim, 1);
    ov::Tensor unsqueezed_tensor(tensor.get_element_type(), new_shape);
    const uint8_t *tensor_data = static_cast<const uint8_t *>(tensor.data());
    uint8_t *unsqueezed_data = static_cast<uint8_t *>(unsqueezed_tensor.data());
    std::memcpy(unsqueezed_data, tensor_data, tensor.get_byte_size());
    return unsqueezed_tensor;
}

ov::Tensor squeeze(const ov::Tensor &tensor, size_t dim) {
    ov::Shape original_shape = tensor.get_shape();
    OPENVINO_ASSERT(original_shape.size() > dim && original_shape[dim] == 1,
            "Cannot squeeze dimension ", dim, " of shape ", original_shape.to_string());
    ov::Shape new_shape = original_shape;
    new_shape.erase(new_shape.begin() + dim);
    ov::Tensor squeezed_tensor(tensor.get_element_type(), new_shape);
    const uint8_t *tensor_data = static_cast<const uint8_t *>(tensor.data());
    uint8_t *squeezed_data = static_cast<uint8_t *>(squeezed_tensor.data());
    std::memcpy(squeezed_data, tensor_data, tensor.get_byte_size());
    return squeezed_tensor;
}

ov::Tensor stack(const std::vector<ov::Tensor>& tensors) {
    ov::Shape stacked_shape = tensors[0].get_shape();
    stacked_shape.insert(stacked_shape.begin(), tensors.size());
    ov::Tensor stacked_tensor(tensors[0].get_element_type(), stacked_shape);
    uint8_t *stacked_data = static_cast<uint8_t *>(stacked_tensor.data());
    for (size_t i = 0; i < tensors.size(); ++i) {
        const uint8_t *tensor_data = static_cast<const uint8_t *>(tensors[i].data());
        std::memcpy(stacked_data + i * tensors[i].get_byte_size(), tensor_data, tensors[i].get_byte_size());
    }    
    return stacked_tensor;
}

}
