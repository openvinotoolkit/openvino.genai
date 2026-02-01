
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace utils {

std::string get_input_arg(int argc, char* argv[], const std::string& key, const std::string& default_value = std::string());

bool readFileToString(const std::string& filename, std::string& content);

// Check if name contains any of the keys in the keys vector
bool contain_key(const std::string& name, const std::vector<std::string>& keys);

// Find parameter module from yaml config file
YAML::Node find_param_module_in_yaml(const std::filesystem::path& cfg_yaml_path);

}  // namespace utils
