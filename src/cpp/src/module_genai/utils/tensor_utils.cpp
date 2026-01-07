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

ov::Tensor slice_tensor(const ov::Tensor& tensor, ov::Coordinate begin, ov::Coordinate end) {
    OPENVINO_ASSERT(begin.size() == end.size(), "Begin and end coordinates must have the same number of dimensions.");
    ov::Shape tensor_shape = tensor.get_shape();
    OPENVINO_ASSERT(begin.size() == tensor_shape.size(), "Begin coordinate size must match tensor rank.");

    ov::Tensor sliced_view = ov::Tensor(tensor, begin, end);
    ov::Tensor deep_copy(sliced_view.get_element_type(), sliced_view.get_shape());

    sliced_view.copy_to(deep_copy);
    return deep_copy;
}

ov::Tensor concat_tensors(const std::vector<ov::Tensor>& tensors, size_t axis) {
    OPENVINO_ASSERT(!tensors.empty(), "Input tensor list is empty.");

    ov::Shape result_shape = tensors[0].get_shape();
    OPENVINO_ASSERT(axis < result_shape.size(), "Axis is out of bounds.");

    size_t concat_dim_size = 0;
    for (const auto& t : tensors) {
        concat_dim_size += t.get_shape()[axis];
    }
    result_shape[axis] = concat_dim_size;
    ov::Tensor result_tensor(tensors[0].get_element_type(), result_shape);

    size_t element_byte_size = tensors[0].get_element_type().size();
    size_t inner_size = 1;
    for (size_t i = axis + 1; i < result_shape.size(); ++i) {
        inner_size *= result_shape[i];
    }

    size_t outer_iterations = 1;
    for (size_t i = 0; i < axis; ++i) {
        outer_iterations *= result_shape[i];
    }

    uint8_t* dst_ptr = static_cast<uint8_t*>(result_tensor.data());
    for (size_t outer = 0; outer < outer_iterations; ++outer) {
        for (const auto& t : tensors) {
            const uint8_t* src_ptr = static_cast<const uint8_t*>(t.data());
            size_t axis_dim = t.get_shape()[axis];

            size_t bytes_to_copy = axis_dim * inner_size * element_byte_size;
            size_t src_offset = outer * bytes_to_copy;

            std::memcpy(dst_ptr, src_ptr + src_offset, bytes_to_copy);
            dst_ptr += bytes_to_copy;
        }
    }

    return result_tensor;
}

const std::string shape_to_string(const ov::Shape& shape) {
    std::ostringstream oss;
    oss << shape;
    return oss.str();
}
}  // namespace ov::genai::module::tensor_utils
