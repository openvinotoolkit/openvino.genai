// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "structured_output_controller.hpp"

namespace ov {
namespace genai {

std::unordered_map<std::string, StructuredOutputController::BackendFactory>&
StructuredOutputController::get_backend_registry() {
    static std::unordered_map<std::string, BackendFactory> registry;
    return registry;
}

std::string& StructuredOutputController::get_default_backend_name() {
    static std::string default_backend = "xgrammar";
    return default_backend;
}

void StructuredOutputController::register_backend(const std::string& name, BackendFactory factory) {
    get_backend_registry()[name] = std::move(factory);
}

void StructuredOutputController::set_default_backend(const std::string& name) {
    if (get_backend_registry().find(name) == get_backend_registry().end()) {
        OPENVINO_THROW("Cannot set default backend to unregistered backend: " + name);
    }

    get_default_backend_name() = name;
}

StructuredOutputController::StructuredOutputController(const ov::genai::Tokenizer& tokenizer,
                                                       std::optional<int> vocab_size)
    : m_tokenizer(tokenizer), m_vocab_size(vocab_size) {}

std::shared_ptr<LogitTransformers::ILogitTransformer>
StructuredOutputController::get_logits_transformer(const ov::genai::GenerationConfig& sampling_parameters) {
    auto& guided_gen_config = sampling_parameters.structured_output_config;
    if (!guided_gen_config.has_value()) {
        OPENVINO_THROW("Structured output is not enabled in the provided GenerationConfig.");
    }
    std::string backend_name = (*guided_gen_config).backend.value_or(get_default_backend_name());
    
    auto& registry = get_backend_registry();
    auto factory_it = registry.find(backend_name);
    if (factory_it == registry.end()) {
        OPENVINO_THROW("Structured output backend not found: " + backend_name);
    }
    return std::move(factory_it->second(m_tokenizer, m_vocab_size, sampling_parameters));
}

} // namespace genai
} // namespace ov
