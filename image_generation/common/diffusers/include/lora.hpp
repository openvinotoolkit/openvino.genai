// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/op/constant.hpp"
#include "openvino/pass/graph_rewrite.hpp"

class InsertLoRA : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("InsertLoRA", "0");

    using LoRAMap = std::map<std::string, std::shared_ptr<ov::op::v0::Constant>>;

    explicit InsertLoRA(LoRAMap& lora_map);

private:
    LoRAMap* m_lora_map;
};

std::map<std::string, InsertLoRA::LoRAMap>
read_lora_adapters(const std::string& filename, const float alpha = 0.75f);
