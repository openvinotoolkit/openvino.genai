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

bool is_xml(const std::string& path);

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

template <typename T>
void read_json_param(const nlohmann::json& data, const std::string& name, T& param) {
    if (data.contains(name) && data[name].type() == json_type_traits<T>::json_value_t) {
        param = data[name];
    }
}

template <typename T>
void read_anymap_param(const ov::AnyMap& config_map, const std::string& name, T& param) {
    if (config_map.count(name)) {
        param = config_map.at(name).as<T>();
    }
}

const char* get_tokenizers_env_name();

// const char* OV_TOKENIZERS_ENV_NAME = "OPENVINO_TOKENIZERS_PATH_GENAI";

class GenAIEnvManager {
public:
    GenAIEnvManager(const std::string& path);
    ~GenAIEnvManager();
private:
    bool was_already_set;
};

}  // namespace utils
}  // namespace genai
}  // namespace ov

