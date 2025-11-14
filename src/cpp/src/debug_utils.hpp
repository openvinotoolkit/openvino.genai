// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fstream>
#include <iostream>
#include <openvino/runtime/tensor.hpp>
#include <string>

template <typename T>
void print_array(T* array, size_t size) {
    std::cout << " => [ ";
    for (size_t i = 0; i < std::min(size, size_t(10)); ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << " ] " << std::endl;
}

template <typename T>
void print_tensor(ov::Tensor tensor) {
    const auto shape = tensor.get_shape();
    const size_t rank = shape.size();
    const auto* data = tensor.data<T>();

    if (rank > 3) {
        print_array(data, tensor.get_size());
        return;
    }

    const size_t batch_size = shape[0];
    const size_t seq_length = shape[1];

    std::cout << " => [ \n";
    for (size_t batch = 0; batch < batch_size; ++batch) {
        std::cout << "  [ ";
        const size_t batch_offset = batch * seq_length;

        if (rank == 2) {
            for (size_t j = 0; j < std::min(seq_length, size_t(10)); ++j) {
                std::cout << data[batch_offset + j] << " ";
            }
            std::cout << "]\n";
            continue;
        }

        const size_t hidden_size = shape[2];

        for (size_t seq = 0; seq < seq_length; ++seq) {
            if (seq != 0)
                std::cout << "    ";
            std::cout << "[ ";
            const size_t seq_offset = (batch_offset + seq) * hidden_size;
            for (size_t h = 0; h < std::min(hidden_size, size_t(10)); ++h) {
                std::cout << data[seq_offset + h] << " ";
            }
            std::cout << "]\n";
        }
    }
    std::cout << " ]" << std::endl;
}

inline void print_tensor(std::string name, ov::Tensor tensor) {
    std::cout << name;
    std::cout << " " << tensor.get_shape().to_string();
    if (tensor.get_element_type() == ov::element::i32) {
        print_tensor<int>(tensor);
    } else if (tensor.get_element_type() == ov::element::i64) {
        print_tensor<int64_t>(tensor);
    } else if (tensor.get_element_type() == ov::element::f32) {
        print_tensor<float>(tensor);
    } else if (tensor.get_element_type() == ov::element::boolean) {
        print_tensor<bool>(tensor);
    } else if (tensor.get_element_type() == ov::element::f16) {
        print_tensor<ov::float16>(tensor);
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

/// @brief Read an npy file created in Python:
/// with open('ndarray.npy', 'wb') as file:
///     np.save(file, ndarray)
inline ov::Tensor from_npy(const std::filesystem::path& npy) {
    std::ifstream fstream{npy, std::ios::binary};
    fstream.seekg(0, std::ios_base::end);
    OPENVINO_ASSERT(fstream.good());
    auto full_file_size = static_cast<std::size_t>(fstream.tellg());
    fstream.seekg(0, std::ios_base::beg);

    std::string magic_string(6, ' ');
    fstream.read(&magic_string[0], magic_string.size());
    OPENVINO_ASSERT(magic_string == "\x93NUMPY");

    fstream.ignore(2);
    unsigned short header_size;
    fstream.read((char*)&header_size, sizeof(header_size));

    std::string header(header_size, ' ');
    fstream.read(&header[0], header.size());

    int idx, from, to;

    // Verify fortran order is false
    const std::string fortran_key = "'fortran_order':";
    idx = header.find(fortran_key);
    OPENVINO_ASSERT(idx != -1);

    from = header.find_last_of(' ', idx + fortran_key.size()) + 1;
    to = header.find(',', from);
    auto fortran_value = header.substr(from, to - from);
    OPENVINO_ASSERT(fortran_value == "False");

    // Verify array shape matches the input's
    const std::string shape_key = "'shape':";
    idx = header.find(shape_key);
    OPENVINO_ASSERT(idx != -1);

    from = header.find('(', idx + shape_key.size()) + 1;
    to = header.find(')', from);

    std::string shape_data = header.substr(from, to - from);
    ov::Shape _shape;

    if (!shape_data.empty()) {
        shape_data.erase(std::remove(shape_data.begin(), shape_data.end(), ','), shape_data.end());

        std::istringstream shape_data_stream(shape_data);
        size_t value;
        while (shape_data_stream >> value) {
            _shape.push_back(value);
        }
    }

    // Verify array data type matches input's
    std::string dataTypeKey = "'descr':";
    idx = header.find(dataTypeKey);
    OPENVINO_ASSERT(-1 != idx);

    from = header.find('\'', idx + dataTypeKey.size()) + 1;
    to = header.find('\'', from);
    std::string type;
    type = header.substr(from, to - from);

    size_t _size = 0;
    _size = full_file_size - static_cast<std::size_t>(fstream.tellg());
    ov::element::Type tensor_type;
    if ("<f4" == type) {
        tensor_type = ov::element::f32;
    } else if ("|u1" == type) {
        tensor_type = ov::element::u8;
    } else if ("<i8" == type) {
        tensor_type = ov::element::i64;
    } else if ("|b1" == type) {
        tensor_type = ov::element::boolean;
    } else {
        OPENVINO_THROW("Not implemented dtype");
    }
    OPENVINO_ASSERT(_size == ov::shape_size(_shape) * tensor_type.size());
    ov::Tensor tensor{tensor_type, _shape};
    fstream.read((char*)tensor.data(), _size);
    OPENVINO_ASSERT(fstream.gcount() == _size);
    return tensor;
}
