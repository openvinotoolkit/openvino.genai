
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "utils.hpp"

#include <cstdlib>
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error "Missing the <filesystem> header."
#endif
#include "json_utils.hpp"
#include <fstream>
#include <openvino/runtime/tensor.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

bool readFileToString(const std::string& filename, std::string& content) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        content.clear();
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    content = buffer.str();
    file.close();
    return true;
}

std::string get_data_path() {
    const char* env_name = "DATA_DIR";
    const std::string default_path = "./test_data"; 

    const char* env_p = std::getenv(env_name);
    fs::path base_path;

    if (env_p != nullptr && std::string(env_p) != "") {
        base_path = env_p;
    } else {
        base_path = default_path;
    }

    if (!fs::exists(base_path)) {
        throw std::runtime_error(
            "Data directory does not exist: " + base_path.string() +
            "\nPlease set the DATA_DIR environment variable to point to the correct data directory.");
    }

    if (!fs::is_directory(base_path)) {
        throw std::runtime_error(
            "Path exists but is not a directory: " + base_path.string() +
            "\nPlease set the DATA_DIR environment variable to point to the correct data directory.");
    }

    return fs::absolute(base_path).string();
}

std::string get_model_path() {
    const char* env_name = "MODEL_DIR";
    const std::string default_path = "./test_models";

    const char* env_p = std::getenv(env_name);
    fs::path base_path;

    if (env_p != nullptr && std::string(env_p) != "") {
        base_path = env_p;
    } else {
        base_path = default_path;
    }

    if (!fs::exists(base_path)) {
        throw std::runtime_error(
            "Model directory does not exist: " + base_path.string() +
            "\nPlease set the MODEL_DIR environment variable to point to the correct model directory.");
    }

    if (!fs::is_directory(base_path)) {
        throw std::runtime_error(
            "Path exists but is not a directory: " + base_path.string() +
            "\nPlease set the MODEL_DIR environment variable to point to the correct model directory.");
    }

    return fs::absolute(base_path).string();
}

bool check_env_variable(const std::string& var_name) {
    const char* env_p = std::getenv(var_name.c_str());
    if (env_p != nullptr && (std::string(env_p) == "true" || std::string(env_p) == "TRUE" ||
                             std::string(env_p) == "1" || std::string(env_p) == "True")) {
        return true;
    }

    return false;
}

bool check_file_exists(const std::string& path) {
    return fs::exists(path) && fs::is_regular_file(path);
}

ov::Tensor load_tensor_from_file(const std::string& meta_data_path) {
    std::ifstream meta_data_file(meta_data_path);
    if (!meta_data_file.is_open()) {
        OPENVINO_THROW("Failed to open tensor meta data file: " + meta_data_path);
    }
    std::string dtype;
    std::vector<size_t> shape;

    nlohmann::json parsed = nlohmann::json::parse(meta_data_file);
    ov::genai::utils::read_json_param(parsed, "dtype", dtype);
    ov::genai::utils::read_json_param(parsed, "shape", shape);

    ov::element::Type element_type;
    if (dtype == "torch.float32") {
        element_type = ov::element::f32;
    } else if (dtype == "torch.int64") {
        element_type = ov::element::i64;
    } else {
        OPENVINO_THROW("Unsupported data type: " + dtype);
    }
    ov::Shape ov_shape(shape.begin(), shape.end());
    ov::Tensor tensor(element_type, ov_shape);
    char *data_ptr = static_cast<char *>(tensor.data());

    std::string data_file_path = meta_data_path.substr(0, meta_data_path.find_last_of('.')) + ".bin";
    std::ifstream data_file(data_file_path, std::ios::binary);
    if (!data_file.is_open()) {
        OPENVINO_THROW("Failed to open tensor data file: " + data_file_path);
    }
    data_file.read(data_ptr, tensor.get_byte_size());
    data_file.close();
    return tensor;
}

std::string get_test_file_path(const std::string& filename) {
    return (fs::path(get_data_path()) / filename).string();
}
