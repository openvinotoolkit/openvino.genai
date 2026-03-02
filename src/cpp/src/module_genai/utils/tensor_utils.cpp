// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor_utils.hpp"
#include "openvino/op/parameter.hpp"
#include "module_genai/utils/profiler.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"

namespace ov::genai::module::tensor_utils {

InferRequest init_slice_request(const std::string &device) {
    auto input = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::PartialShape{-1, -1, -1, -1});
    auto begin = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::Shape{4});
    auto end = std::make_shared<ov::op::v0::Parameter>(
        ov::element::i64, ov::Shape{4});
    auto stride = std::make_shared<ov::op::v0::Constant>(
        ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, 1, 1, 1});
    
    std::vector<int64_t> begin_mask = {0, 0, 0, 0};
    std::vector<int64_t> end_mask = {0, 0, 0, 0};
    
    auto sliced_tensor = std::make_shared<ov::op::v1::StridedSlice>(
        input, begin, end, stride, begin_mask, end_mask);
    
    auto model = std::make_shared<ov::Model>(
        ov::OutputVector {sliced_tensor}, ov::ParameterVector {input, begin, end});

    auto compiled_model = ov::genai::utils::singleton_core().compile_model(model, device);
    return compiled_model.create_infer_request();
}

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

ov::Tensor stack(const std::vector<ov::Tensor>& tensors, size_t axis) {
    ov::Shape input_shape = tensors[0].get_shape();
    ov::Shape stacked_shape = input_shape;
    stacked_shape.insert(stacked_shape.begin() + static_cast<long>(axis), tensors.size());
    ov::Tensor stacked_tensor(tensors[0].get_element_type(), stacked_shape);
    size_t elem_size = tensors[0].get_element_type().size();

    size_t outer = 1;
    for (size_t i = 0; i < axis; ++i) {
        outer *= input_shape[i];
    }
    size_t inner = 1;
    for (size_t i = axis; i < input_shape.size(); ++i) {
        inner *= input_shape[i];
    }

    uint8_t *stacked_data = static_cast<uint8_t *>(stacked_tensor.data());
    for (size_t o = 0; o < outer; ++o) {
        for (size_t j = 0; j < tensors.size(); ++j) {
            const uint8_t *tensor_data = static_cast<const uint8_t *>(tensors[j].data());
            size_t in_offset  = o * inner * elem_size;
            size_t out_offset = (o * tensors.size() + j) * inner * elem_size;
            std::memcpy(stacked_data + out_offset, tensor_data + in_offset, inner * elem_size);
        }
    }
    return stacked_tensor;
}

ov::Tensor slice_tensor_with_model(const ov::Tensor& tensor, ov::Coordinate begin, ov::Coordinate end, InferRequest infer_request) {
    OPENVINO_ASSERT(begin.size() == end.size(), "Begin and end coordinates must have the same number of dimensions.");
    ov::Shape tensor_shape = tensor.get_shape();
    OPENVINO_ASSERT(begin.size() == tensor_shape.size(), "Begin coordinate size must match tensor rank.");

    ov::Tensor begin_tensor(ov::element::i64, ov::Shape{begin.size()}, begin.data());
    ov::Tensor end_tensor(ov::element::i64, ov::Shape{end.size()}, end.data());

    infer_request.set_input_tensor(0, tensor);
    infer_request.set_input_tensor(1, begin_tensor);
    infer_request.set_input_tensor(2, end_tensor);

    ov::Shape output_shape;
    for (size_t i = 0; i < begin.size(); ++i) {
        output_shape.push_back(end[i] - begin[i]);
    }
    ov::Tensor output_tensor(tensor.get_element_type(), output_shape);
    infer_request.set_output_tensor(0, output_tensor);

    infer_request.infer();

    return output_tensor;
}

ov::Tensor slice_tensor(const ov::Tensor& tensor, ov::Coordinate begin, ov::Coordinate end) {
    OPENVINO_ASSERT(begin.size() == end.size(), "Begin and end coordinates must have the same number of dimensions.");
    ov::Shape tensor_shape = tensor.get_shape();
    OPENVINO_ASSERT(begin.size() == tensor_shape.size(), "Begin coordinate size must match tensor rank.");

#if 0
    ov::Tensor sliced_view = ov::Tensor(tensor, begin, end);
    ov::Tensor deep_copy(sliced_view.get_element_type(), sliced_view.get_shape());

    sliced_view.copy_to(deep_copy);
    return deep_copy;
#else
    ov::Shape roi_shape;
    for (size_t i = 0; i < begin.size(); ++i) {
        roi_shape.push_back(end[i] - begin[i]);
    }

    ov::Tensor dst_tensor(tensor.get_element_type(), roi_shape);

    const uint8_t* src_ptr = static_cast<const uint8_t*>(tensor.data());
    uint8_t* dst_ptr = static_cast<uint8_t*>(dst_tensor.data());

    size_t elem_size = tensor.get_element_type().size();
    size_t rank = tensor_shape.size();

    // Calc original Tensor's each dim srides.
    std::vector<size_t> src_strides(rank);
    src_strides[rank - 1] = elem_size;
    for (int i = rank - 2; i >= 0; --i) {
        src_strides[i] = src_strides[i + 1] * tensor_shape[i + 1];
    }

    // last dim's copy size.
    size_t last_dim_len = roi_shape.back() * elem_size;

    size_t num_rows = 1;
    for (size_t i = 0; i < rank - 1; ++i) {
        num_rows *= roi_shape[i];
    }

    for (size_t i = 0; i < num_rows; ++i) {
        size_t src_offset = 0;
        size_t temp_idx = i;

        for (int d = rank - 2; d >= 0; --d) {
            size_t coord_d = (temp_idx % roi_shape[d]) + begin[d];
            src_offset += coord_d * src_strides[d];
            temp_idx /= roi_shape[d];
        }
        src_offset += begin.back() * elem_size;

        std::memcpy(dst_ptr + i * last_dim_len, src_ptr + src_offset, last_dim_len);
    }

    return dst_tensor;
#endif
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

float calculate_l2_norm(const ov::Tensor &tensor, size_t start_idx, size_t end_idx) {
    const float* data = tensor.data<const float>();
    float sum = 0.0f;
    for (size_t i = start_idx; i < end_idx; ++i) {
        sum += data[i] * data[i];
    }
    return std::sqrt(sum);
}

}  // namespace ov::genai::module::tensor_utils
