
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <yaml-cpp/yaml.h>

// Helper functions to create YAML nodes for inputs and outputs.
// source: empty means no source field.
inline YAML::Node input_node(const std::string& name,
                             const std::string& type,
                             const std::string& source = std::string()) {
    YAML::Node input_node;
    input_node["name"] = name;
    input_node["type"] = type;
    if (!source.empty()) {
        input_node["source"] = source;
    }
    return input_node;
}

inline YAML::Node output_node(const std::string& name, const std::string& type) {
    YAML::Node output_node;
    output_node["name"] = name;
    output_node["type"] = type;
    return output_node;
}

namespace TEST_MODEL {
std::string get_device();

// Return model full path for Qwen2.5-VL-3B-Instruct INT4 model.
std::string Qwen2_5_VL_3B_Instruct_INT4();

// Return model full path for Z-Image-Turbo-fp16-ov model.
std::string ZImage_Turbo_fp16_ov();

std::string Wan_2_1();

std::string Qwen3_5();

std::string Qwen3_5_0_8B();

std::string Qwen3_Omni_4B_Instruct_Multilingual();

// Return yaml content string for Qwen2.5-VL-3B-Instruct model pipeline configuration.
std::string get_qwen2_5_vl_config_yaml(const std::string& model_path, const std::string& device = "CPU");
};  // namespace TEST_MODEL