// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "structured_output_controller.hpp"
#include <xgrammar/xgrammar.h>

namespace ov {
namespace genai {

StructuredOutputController::StructuredOutputController(const Tokenizer& tokenizer,
                                                       std::optional<int> vocab_size)
    : m_impl(nullptr) {
    // Initialize xgrammar backend if needed.
}

LogitTransformers::IStructuredOutputBaseLogitTransformer get_json_schema_logits_transformer(const std::string& json_schema) {
    return m_impl->get_json_schema_logits_transformer(json_schema);
}

} // namespace genai
} // namespace ov