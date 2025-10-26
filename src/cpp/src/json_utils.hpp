
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <set>
#include <optional>

#include <nlohmann/json.hpp>

#include <openvino/core/except.hpp>
#include <openvino/core/any.hpp>

#include "openvino/genai/json_container.hpp"

namespace nlohmann {

template<>
struct adl_serializer<ov::genai::JsonContainer> {
    static void to_json(ordered_json& json, const ov::genai::JsonContainer& container) {
        auto json_value_ptr = static_cast<const ordered_json*>(container._get_json_value_ptr());
        json = *json_value_ptr;
    }
    
    static ov::genai::JsonContainer from_json(const ordered_json& json) {
        return ov::genai::JsonContainer::from_json_string(json.dump());
    }
};

} // namespace nlohmann

namespace ov {
namespace genai {
namespace utils {

template<typename, typename = void>
constexpr bool is_std_array = false;

template<typename T, std::size_t N>
constexpr bool is_std_array<std::array<T, N>> = true;

/// @brief reads value to param if T argument type is aligned with value stores in json
/// if types are not compatible leave param unchanged
template <typename T>
void read_json_param(const nlohmann::json& data, const std::string& name, T& param) {
    if (data.contains(name)) {
        if (data[name].is_number() || data[name].is_boolean() || data[name].is_string() || data[name].is_object()
            || (is_std_array<T> && data[name].is_array())
        ) {
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

inline nlohmann::ordered_json any_map_to_json(const ov::AnyMap& any_map);

inline nlohmann::ordered_json any_to_json(const ov::Any& value) {
    if (value.is<std::string>()) {
        return value.as<std::string>();
    } else if (value.is<int>()) {
        return value.as<int>();
    } else if (value.is<int64_t>()) {
        return value.as<int64_t>();
    } else if (value.is<float>()) {
        return value.as<float>();
    } else if (value.is<double>()) {
        return value.as<double>();
    } else if (value.is<bool>()) {
        return value.as<bool>();
    } else if (value.is<ov::AnyMap>()) {
        return any_map_to_json(value.as<ov::AnyMap>());
    } else if (value.is<std::vector<std::string>>()) {
        return value.as<std::vector<std::string>>();
    } else if (value.is<std::vector<int64_t>>()) {
        return value.as<std::vector<int64_t>>();
    } else if (value.is<std::vector<float>>()) {
        return value.as<std::vector<float>>();
    } else if (value.is<std::vector<double>>()) {
        return value.as<std::vector<double>>();
    } else if (value.is<std::vector<bool>>()) {
        return value.as<std::vector<bool>>();
    } else if (value.is<std::vector<ov::AnyMap>>()) {
        nlohmann::ordered_json array_json = nlohmann::ordered_json::array();
        for (const auto& map : value.as<std::vector<ov::AnyMap>>()) {
            array_json.push_back(any_map_to_json(map));
        }
        return array_json;
    } else if (value.is<ov::genai::JsonContainer>()) {
        return value.as<ov::genai::JsonContainer>();
    } else {
        OPENVINO_THROW("Failed to convert Any to JSON, unsupported type: ", value.type_info().name());
    }
}

inline nlohmann::ordered_json any_map_to_json(const ov::AnyMap& any_map) {
    nlohmann::ordered_json object_json = nlohmann::ordered_json::object();
    for (const auto& [key, value] : any_map) {
        try {
            object_json[key] = any_to_json(value);
        } catch (const std::exception& e) {
            OPENVINO_THROW("Failed to convert AnyMap to JSON for key: ", key, "\n", e.what());
        }
    }
    return object_json;
}



}  // namespace utils
}  // namespace genai
}  // namespace ov
