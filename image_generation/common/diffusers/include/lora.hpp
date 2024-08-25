// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define GENAI_NEW_LORA 1

#if GENAI_NEW_LORA

#include <openvino/genai/lora_adapter.hpp>

#else

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/op/constant.hpp"
#include "openvino/pass/graph_rewrite.hpp"

#define DEBUG_PRINT(X) do { std::cerr << "[ DEBUG ] " << X << "\n"; } while(false)


class InsertLoRA : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertLoRA", "0");

    using LoRAMap = std::map<std::string, std::shared_ptr<ov::op::v0::Constant>>;

    explicit InsertLoRA(LoRAMap& lora_map);

    ~InsertLoRA () {
        DEBUG_PRINT("Applied: " << applied);
    }

private:
    LoRAMap* m_lora_map;
    size_t applied = 0;
};

std::map<std::string, InsertLoRA::LoRAMap>
read_lora_adapters(const std::string& filename, const float alpha = 0.75f);

using Adapter = std::vector<std::shared_ptr<ov::op::v0::Constant>>;
using AdapterMap = std::map<std::string, Adapter>;
using LoRAPrefixes = std::map<std::string, std::string>;

std::map<std::string, AdapterMap> load_lora_adapter(const std::string& adapter_file_path, const float alpha, const LoRAPrefixes& prefixes);
void apply_lora_adapter(std::shared_ptr<ov::Model> model, const AdapterMap& adapter_map);

#endif