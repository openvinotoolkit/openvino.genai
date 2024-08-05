// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/op/constant.hpp"
#include "openvino/pass/graph_rewrite.hpp"

#define DEBUG_PRINT(X) do { std::cerr << "[ DEBUG ] " << X << "\n"; } while(false)

using Adapter = std::vector<std::shared_ptr<ov::op::v0::Constant>>;
using AdapterMap = std::map<std::string, Adapter>;
using LoRAPrefixes = std::map<std::string, std::string>;

std::map<std::string, AdapterMap> load_lora_adapter(const std::string& adapter_file_path, const float alpha, const LoRAPrefixes& prefixes);
void apply_lora_adapter(std::shared_ptr<ov::Model> model, const AdapterMap& adapter_map);