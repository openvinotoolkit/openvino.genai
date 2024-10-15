// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nlohmann/json.hpp>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/processor_config.hpp"

namespace ov {
namespace genai {
namespace utils {

Tensor init_attention_mask(const Tensor& position_ids);

void print_tensor(const ov::Tensor& tensor);

int64_t argmax(const ov::Tensor& logits, const size_t batch_idx);

void initialize_position_ids(ov::Tensor& position_ids, const ov::Tensor& attention_mask, int64_t start_pos = 0);

ov::Tensor extend_attention(ov::Tensor attention_mask);

void update_position_ids(ov::Tensor&& position_ids, const ov::Tensor&& attention_mask);

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

template <typename T>
void read_anymap_param(const ov::AnyMap& config_map, const std::string& name, T& param) {
    auto it = config_map.find(name);
    if (it != config_map.end()) {
        param = it->second.as<T>();
    }
}

const std::string STREAMER_ARG_NAME = "streamer";
const std::string CONFIG_ARG_NAME = "generation_config";
const std::string DRAFT_MODEL_ARG_NAME = "draft_model";

template<typename Config=ov::genai::GenerationConfig>
Config from_config_json_if_exists(const std::filesystem::path& model_path, const char config_name[]="generation_config.json") {
    auto config_file_path = model_path / config_name;
    if (std::filesystem::exists(config_file_path)) {
        return Config{(config_file_path).string()};
    } else {
        return Config{};
    }
}

ov::genai::StreamerVariant get_streamer_from_map(const ov::AnyMap& config_map);

ov::genai::OptionalGenerationConfig get_config_from_map(const ov::AnyMap& config_map);

ProcessorConfig from_any_map(
    const ov::AnyMap& config_map,
    const ProcessorConfig& initial
);

std::pair<ov::AnyMap, ov::AnyMap> split_core_complile_config(const ov::AnyMap& plugin_config);

ov::genai::TokenizedInputs subtract_chat_tokenized_inputs(const ov::genai::TokenizedInputs& minuend, const ov::genai::TokenizedInputs& subtrahend);

void slice_matmul_statefull_model(std::shared_ptr<ov::Model> model);
}  // namespace utils
}  // namespace genai
}  // namespace ov
