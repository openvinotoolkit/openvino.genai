
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

namespace utils {

std::string get_input_arg(int argc,
                          char* argv[],
                          const std::string& key,
                          const std::string& default_value = std::string());

std::string any_to_string(const ov::Any& value);

bool readFileToString(const std::string& filename, std::string& content);

// Check if name contains any of the keys in the keys vector
bool contains_key(const std::string& name, const std::vector<std::string>& keys);

// Find parameter module from yaml config file
YAML::Node find_param_module_in_yaml(const std::filesystem::path& cfg_yaml_path);

}  // namespace utils
