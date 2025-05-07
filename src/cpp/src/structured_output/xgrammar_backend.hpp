// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <xgrammar/xgrammar.h>
#include "structured_output_controller.hpp"


namespace ov {
namespace genai {


class XGrammarStructuredOutput : public IStructuredOutputBaseImpl {
public:
    XGrammarStructuredOutput(const Tokenizer& tokenizer) {

    }

    void render_output(const std::string& data) {
        // Process the structured output using xgrammar functions.
        xgrammar::process(data);
    }
};


} // namespace genai
}// namespace ov
