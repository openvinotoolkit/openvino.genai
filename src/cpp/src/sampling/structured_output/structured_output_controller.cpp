// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "structured_output_controller.hpp"
#include <xgrammar/xgrammar.h>

namespace ov {
namespace genai {

StructuredOutputController::StructuredOutputController(const ov::genai::Tokenizer& tokenizer,
                                                       std::optional<int> vocab_size)
    : m_impl(nullptr) {}

std::shared_ptr<LogitTransformers::ILogitTransformer>
StructuredOutputController::get_logits_transformer(const ov::genai::GenerationConfig& sampling_parameters) {
    return m_impl->get_logits_transformer(sampling_parameters);
}

} // namespace genai
} // namespace ov
