// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/openvino.hpp>
#include <nlohmann/json.hpp>

namespace ov {
namespace genai {
namespace utils {

Tensor init_attention_mask(Tensor& position_ids);

void print_tensor(const ov::Tensor& tensor);

int64_t argmax(const ov::Tensor& logits, const size_t batch_idx);

void initialize_position_ids(ov::Tensor& position_ids, const ov::Tensor& attention_mask, int64_t start_pos = 0);

ov::Tensor extend_attention(ov::Tensor attention_mask);

void update_position_ids(ov::Tensor&& position_ids, const ov::Tensor&& attention_mask);

template <typename>
struct json_type_traits {};

template <>
struct json_type_traits<int> { static constexpr auto json_value_t = nlohmann::json::value_t::number_integer; };

template <>
struct json_type_traits<int64_t> { static constexpr auto json_value_t = nlohmann::json::value_t::number_integer; };

template <>
struct json_type_traits<size_t> { static constexpr auto json_value_t = nlohmann::json::value_t::number_unsigned; };

template <>
struct json_type_traits<float> { static constexpr auto json_value_t = nlohmann::json::value_t::number_float; };

template <>
struct json_type_traits<std::string> { static constexpr auto json_value_t = nlohmann::json::value_t::string; };

template <>
struct json_type_traits<bool> { static constexpr auto json_value_t = nlohmann::json::value_t::boolean; };

/// @brief reads value to param if T argument type is aligned with value stores in json
/// if types are not compatible leave param unchanged
template <typename T>
void read_json_param(const nlohmann::json& data, const std::string& name, T& param) {
    if (data.contains(name)) {
        if constexpr (std::is_integral_v<T>) {
            if (data[name].is_number_integer() || data[name].is_number_unsigned()) {
                param = data[name].get<T>();
            }
        } else if (data[name].type() == json_type_traits<T>::json_value_t) {
            param = data[name].get<T>();
        }
    }
}

template <typename T>
void read_anymap_param(const ov::AnyMap& config_map, const std::string& name, T& param) {
    if (config_map.count(name)) {
        param = config_map.at(name).as<T>();
    }
}

std::tuple<int64_t, int64_t, int64_t> get_special_tokens_from_config_json(const std::filesystem::path& config_path);

}  // namespace utils
}  // namespace genai
}  // namespace ov
