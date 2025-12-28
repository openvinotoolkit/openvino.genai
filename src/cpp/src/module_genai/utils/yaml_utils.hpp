// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "module_genai/module.hpp"

namespace ov {
namespace genai {

namespace module {
namespace utils {

std::pair<std::string, std::string> parse_source(const std::string& source);

PipelineModuleDesc load_config(const std::string& cfg_path);

PipelineModuleDesc load_config_from_string(const std::string& content);

void yaml_cfg_auto_padding(YAML::Node& config_node);

std::ostream& operator<<(std::ostream& os, const IBaseModuleDesc::PTR& desc);

}  // namespace utils
}  // namespace module
}  // namespace genai
}  // namespace ov