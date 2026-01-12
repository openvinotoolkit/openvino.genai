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

inline void save_matrix_as_numpy(const std::vector<std::vector<float>>& matrix, const std::string& filename) {
    if (matrix.empty() || matrix[0].empty()) {
        return;
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // NumPy header format
    const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    const uint8_t major_version = 1;
    const uint8_t minor_version = 0;

    file.write(reinterpret_cast<const char*>(magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&major_version), 1);
    file.write(reinterpret_cast<const char*>(&minor_version), 1);

    // Create header dict string
    const size_t rows = matrix.size();
    const size_t cols = matrix[0].size();
    std::ostringstream header;
    header << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << rows << ", " << cols << "), }";

    std::string header_str = header.str();
    // Pad to make total header size (including length field) a multiple of 64 bytes
    size_t header_len = header_str.size();
    size_t total_header_size = 10 + 2 + header_len;  // 6 (magic) + 2 (version) + 2 (header_len) + header
    size_t padding = (64 - (total_header_size % 64)) % 64;
    header_str.append(padding, ' ');
    header_str.push_back('\n');
    header_len = header_str.size();

    // Write header length (little-endian uint16)
    uint16_t header_len_le = static_cast<uint16_t>(header_len);
    file.write(reinterpret_cast<const char*>(&header_len_le), 2);

    // Write header
    file.write(header_str.c_str(), header_len);

    // Write data in row-major order (C order)
    for (const auto& row : matrix) {
        file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    }

    file.close();
    std::cout << "Saved matrix [" << rows << ", " << cols << "] to " << filename << std::endl;
}

inline void save_vector_of_tensors_as_np(std::vector<ov::Tensor> tensors, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // NumPy header format
    const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    const uint8_t major_version = 1;
    const uint8_t minor_version = 0;

    file.write(reinterpret_cast<const char*>(magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&major_version), 1);
    file.write(reinterpret_cast<const char*>(&minor_version), 1);

    // Create header dict string
    const size_t head_size = tensors.size();
    std::ostringstream header;
    header << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << head_size << ", ";
    const ov::Shape& first_shape = tensors[0].get_shape();
    for (size_t i = 0; i < first_shape.size(); ++i) {
        header << first_shape[i];
        if (i < first_shape.size() - 1) {
            header << ", ";
        }
    }
    header << "), }";
    std::string header_str = header.str();
    // Pad to make total header size (including length field) a multiple of 64 bytes
    size_t header_len = header_str.size();
    size_t total_header_size = 10 + 2 + header_len;  // 6 (magic) + 2 (version) + 2 (header_len) + header
    size_t padding = (64 - (total_header_size % 64)) % 64;
    header_str.append(padding, ' ');
    header_str.push_back('\n');
    header_len = header_str.size();

    // Write header length (little-endian uint16)
    uint16_t header_len_le = static_cast<uint16_t>(header_len);
    file.write(reinterpret_cast<const char*>(&header_len_le), 2);
    // Write header
    file.write(header_str.c_str(), header_len);
    // Write data in row-major order (C order)
    for (const auto& tensor : tensors) {
        const ov::Shape& shape = tensor.get_shape();
        size_t total_size = 1;
        for (const auto& dim : shape) {
            total_size *= dim;
        }
        const float* data = tensor.data<float>();
        file.write(reinterpret_cast<const char*>(data), total_size * sizeof(float));
    }

    file.close();
    std::cout << "Saved vector of tensors to " << filename << std::endl;
}
