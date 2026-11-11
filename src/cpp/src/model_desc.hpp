// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <openvino/runtime/properties.hpp>

#include "openvino/genai/scheduler_config.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"

namespace ov {
class Model;

namespace genai {
class InputsEmbedder;

struct ModelDesc {
    std::shared_ptr<ov::Model> model = nullptr;
    ov::genai::Tokenizer tokenizer;
    std::shared_ptr<ov::genai::InputsEmbedder> inputs_embedder;
    std::string device;
    ov::AnyMap properties;
    ov::genai::SchedulerConfig scheduler_config;
    ov::genai::GenerationConfig generation_config;

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
              const ov::genai::Tokenizer& tokenizer,
              std::shared_ptr<ov::genai::InputsEmbedder> inputs_embedder,
              const std::string& device = {},
              const ov::AnyMap& properties = {},
              const ov::genai::SchedulerConfig& scheduler_config = {},
              const ov::genai::GenerationConfig& generation_config = {})
        : model(model),
          tokenizer(tokenizer),
          inputs_embedder(std::move(inputs_embedder)),
          device(device),
          properties(properties),
          scheduler_config(scheduler_config),
          generation_config(generation_config) {}

    ModelDesc() = default;
};

ov::genai::ModelDesc get_draft_model_from_config(const ov::AnyMap& config);

ov::genai::ModelDesc extract_draft_model_from_config(ov::AnyMap& config);

/// Converts a GGUF file with MTP layers once and returns its draft. When target is set, it
/// receives the target model, which shares weight buffers with the draft.
ov::genai::ModelDesc gguf_mtp_draft_model(const std::filesystem::path& models_path,
                                          const std::string& device,
                                          const ov::AnyMap& properties,
                                          const std::shared_ptr<std::shared_ptr<ov::Model>>& target);

/// Replaces speculative_type and draft_model_path with a draft_model descriptor created
/// with the main model device and properties.
ov::AnyMap resolve_draft_model_request(const std::filesystem::path& models_path,
                                       const std::string& device,
                                       const ov::AnyMap& properties);

}  // namespace genai
}  // namespace ov
