// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "structured_output_controller.hpp"
#include <xgrammar/xgrammar.h>

namespace ov {
namespace genai {

StructuredOutputController::StructuredOutputController(const Tokenizer& tokenizer,
                                                       std::optional<int> vocab_size)
    : m_impl(nullptr) {}

LogitTransformers::IStructuredOutputBaseLogitTransformer get_json_schema_logits_transformer(const GenerationConfig& sampling_parameters) {
    return m_impl->get_logits_transformer(sampling_parameters);
}

} // namespace genai
} // namespace ov
