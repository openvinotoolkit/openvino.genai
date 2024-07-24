// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include <openvino/openvino.hpp>

#include <openvino/runtime/tensor.hpp>

template <typename T>
void print_array(T * array, size_t size) {
    std::cout << " => [ ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << " ] " << std::endl;
}

void print_tensor(std::string name, ov::Tensor tensor) {
    std::cout << name;
    if (tensor.get_element_type() == ov::element::i32) {
        print_array(tensor.data<int>(), tensor.get_size());
    } else if (tensor.get_element_type() == ov::element::i64) {
        print_array(tensor.data<int64_t>(), tensor.get_size());
    } else if (tensor.get_element_type() == ov::element::f32) {
        print_array(tensor.data<float>(), tensor.get_size());
    } else if (tensor.get_element_type() == ov::element::boolean) {
        print_array(tensor.data<bool>(), tensor.get_size());
    }
}

std::string join(const std::vector<std::string>& listOfStrings, const std::string delimiter) {
    std::stringstream ss;
    auto it = listOfStrings.cbegin();
    if (it == listOfStrings.end()) {
        return "";
    }
    for (; it != (listOfStrings.end() - 1); ++it) {
        ss << *it << delimiter;
    }
    if (it != listOfStrings.end()) {
        ss << *it;
    }
    return ss.str();
}

template <typename PropertyExtractor>
static void read_properties(PropertyExtractor&& property_extractor, std::vector<std::string>& output_configuration_values) {
    auto key = std::string("SUPPORTED_PROPERTIES");  // ov::supported_properties;
    std::vector<ov::PropertyName> supported_config_keys;
    try {
        ov::Any value = property_extractor(key);
        supported_config_keys = value.as<std::vector<ov::PropertyName>>();
    } catch (...) {
        std::cout << "Exception thrown from OpenVINO when requesting model property: " << key << std::endl;
        return;
    }

    for (auto& key : supported_config_keys) {
        if (key == "SUPPORTED_PROPERTIES")
            continue;
        std::string value;
        try {
            ov::Any param_value = property_extractor(key);
            value = param_value.as<std::string>();
        } catch (...) {
            std::cout << "WARNING: Exception thrown from OpenVINO when requesting model property: " << key << std::endl;
            continue;
        }
        output_configuration_values.emplace_back(join({key, value}, ": "));
    }
    std::sort(output_configuration_values.begin(), output_configuration_values.end());
}
