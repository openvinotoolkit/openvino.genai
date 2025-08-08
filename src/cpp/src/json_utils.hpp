
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <set>
#include <optional>

#include <nlohmann/json.hpp>

namespace ov {
namespace genai {
namespace utils {

/// @brief reads value to param if T argument type is aligned with value stores in json
/// if types are not compatible leave param unchanged
template <typename T>
void read_json_param(const nlohmann::json& data, const std::string& name, T& param) {
    if (data.contains(name)) {
        if (data[name].is_number() || data[name].is_boolean() || data[name].is_string() || data[name].is_object()) {
            param = data[name].get<T>();
        }
    } else if (name.find(".") != std::string::npos) {
        size_t delimiter_pos = name.find(".");
        std::string key = name.substr(0, delimiter_pos);
        if (!data.contains(key)) {
            return;
        }
        std::string rest_key = name.substr(delimiter_pos + 1);

        read_json_param(data[key], rest_key, param);
    }
}

template <typename V>
void read_json_param(const nlohmann::json& data, const std::string& name, std::vector<V>& param) {
    if (data.contains(name) && data[name].is_array()) {
        param.resize(0);
        for (const auto elem : data[name]) {
            param.push_back(elem.get<V>());
        }
    }
}

template <typename V>
void read_json_param(const nlohmann::json& data, const std::string& name, std::set<V>& param) {
    if (data.contains(name) && data[name].is_array()) {
        for (const auto elem : data[name]) {
            param.insert(elem.get<V>());
        }
    }
}

template <typename T>
void read_json_param(const nlohmann::json& data, const std::string& name, std::optional<T>& param) {
    if (data.contains(name) && !data[name].is_null()) {
        T value;
        read_json_param(data, name, value);
        param = value;
    }
}

}  // namespace utils
}  // namespace genai
}  // namespace ov
