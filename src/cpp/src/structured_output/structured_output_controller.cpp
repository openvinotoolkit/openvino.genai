// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "structured_output_controller.hpp"
#include <xgrammar/xgrammar.h>

namespace ov {
namespace genai {

StructuredOutputController::StructuredOutputController(const Tokenizer& tokenizer)
    : m_impl(nullptr) {
    // Initialize xgrammar backend if needed.
}

void StructuredOutputController::render_output(const std::string &data) {
    m_impl->render_output(data);
}

} // namespace genai
} // namespace ov