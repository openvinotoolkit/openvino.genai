// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <iostream>
#include <fstream>

#include <openvino/runtime/tensor.hpp>

template <typename T>
void print_array(T * array, size_t size) {
    std::cout << " => [ ";
    for (size_t i = 0; i < std::min(size, size_t(10)); ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << " ] " << std::endl;
}

inline void print_tensor(std::string name, ov::Tensor tensor) {
    std::cout << name;
    std::cout << " " << tensor.get_shape().to_string();
    if (tensor.get_element_type() == ov::element::i32) {
        print_array(tensor.data<int>(), tensor.get_size());
    } else if (tensor.get_element_type() == ov::element::i64) {
        print_array(tensor.data<int64_t>(), tensor.get_size());
    } else if (tensor.get_element_type() == ov::element::f32) {
        print_array(tensor.data<float>(), tensor.get_size());
    } else if (tensor.get_element_type() == ov::element::boolean) {
        print_array(tensor.data<bool>(), tensor.get_size());
    } else if (tensor.get_element_type() == ov::element::f16) {
        print_array(tensor.data<ov::float16>(), tensor.get_size());
    }
}

template <typename tensor_T, typename file_T>
void _read_tensor_step(tensor_T* data, size_t i, std::ifstream& file, size_t& printed_elements, bool assign) {
    const size_t print_size = 10;

    file_T value;
    file >> value;

    // this mode is used to fallback to reference data to check further execution
    if (assign)
        data[i] = value;

    if (std::abs(value - data[i]) > 1e-7 && printed_elements < print_size) {
        std::cout << i << ") ref = " << value << " act = " << static_cast<file_T>(data[i]) << std::endl;
        ++printed_elements;
    }
}

inline void read_tensor(const std::string& file_name, ov::Tensor tensor, bool assign = false) {
    std::ifstream file(file_name.c_str());
    OPENVINO_ASSERT(file.is_open(), "Failed to open file ", file_name);

    std::cout << "Opening " << file_name << std::endl;
    std::cout << "tensor shape " << tensor.get_shape() << std::endl;

    for (size_t i = 0, printed_elements = 0; i < tensor.get_size(); ++i) {
        if (tensor.get_element_type() == ov::element::f32)
            _read_tensor_step<float, float>(tensor.data<float>(), i, file, printed_elements, assign);
        else if (tensor.get_element_type() == ov::element::f64)
            _read_tensor_step<double, double>(tensor.data<double>(), i, file, printed_elements, assign);
        else if (tensor.get_element_type() == ov::element::u8)
            _read_tensor_step<uint8_t, float>(tensor.data<uint8_t>(), i, file, printed_elements, assign);
        else {
            OPENVINO_THROW("Unsupported tensor type ", tensor.get_element_type(), " by read_tensor");
        }
    }

    std::cout << "Closing " << file_name << std::endl;
}
