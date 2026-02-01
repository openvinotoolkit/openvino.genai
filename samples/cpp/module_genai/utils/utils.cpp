
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "utils.hpp"

#include <filesystem>
#include <fstream>
#include <openvino/runtime/tensor.hpp>
#include <sstream>
#include <string>

#include "yaml-cpp/yaml.h"

namespace utils {

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

YAML::Node find_param_module_in_yaml(const std::filesystem::path& cfg_yaml_path) {
    YAML::Node config = YAML::LoadFile(cfg_yaml_path.string());
    auto pipeline_modules = config["pipeline_modules"];
    // loop pipeline_modules to find a node with type "ParameterModule"
    for (const auto& module : pipeline_modules) {
        if (module.second["type"] && module.second["type"].as<std::string>() == "ParameterModule") {
            return module.second["outputs"];
        }
    }
    throw std::runtime_error("Could not find ParameterModule in config YAML.");
}

bool contain_key(const std::string& name, const std::vector<std::string>& keys) {
    for (const auto& key : keys) {
        if (name.find(key) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::string get_input_arg(int argc, char* argv[], const std::string& key, const std::string& default_value) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == key && i + 1 < argc) {
            return argv[i + 1];
        }
    }
    return default_value;
}

}  // namespace utils