// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>

#include "openvino/genai/scheduler_config.hpp"
#include "openvino/genai/generation_config.hpp"

#include "visual_language/inputs_embedder.hpp"

namespace ov {
namespace genai {

struct ModelDesc {
    std::string device;
    ov::genai::SchedulerConfig scheduler_config;
    ov::AnyMap properties;
    ov::genai::GenerationConfig generation_config;
    std::shared_ptr<ov::Model> model = nullptr;
    std::shared_ptr<ov::genai::InputsEmbedder> inputs_embedder;
    ov::genai::Tokenizer tokenizer;

    ModelDesc(const std::shared_ptr<ov::Model>& model,
              const ov::genai::Tokenizer& tokenizer,
              const std::string& device = {},
              const ov::AnyMap& properties = {},
              const ov::genai::SchedulerConfig& scheduler_config = {},
              const ov::genai::GenerationConfig& generation_config = {})
        : model(model),
          tokenizer(tokenizer),
          device(device),
          properties(properties),
          scheduler_config(scheduler_config),
          generation_config(generation_config) {}

    ModelDesc(const std::shared_ptr<ov::Model>& model,
              std::shared_ptr<ov::genai::InputsEmbedder> inputs_embedder,
              const std::string& device = {},
              const ov::AnyMap& properties = {},
              const ov::genai::SchedulerConfig& scheduler_config = {},
              const ov::genai::GenerationConfig& generation_config = {})
        : model(model),
          inputs_embedder(inputs_embedder),
          device(device),
          properties(properties),
          scheduler_config(scheduler_config),
          generation_config(generation_config) {}

    ModelDesc() = default;
};

ov::genai::ModelDesc get_draft_model_from_config(const ov::AnyMap& config);

ov::genai::ModelDesc extract_draft_model_from_config(ov::AnyMap& config);

}  // namespace genai
}  // namespace ov