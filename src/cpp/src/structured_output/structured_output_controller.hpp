// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <memory>

#include "openvino/genai/tokenizer.hpp"

namespace ov {
namespace genai {

class IStructuredOutputBaseImpl {
public:
    virtual ~IStructuredOutputBaseImpl() = default;
    virtual void render_output(const std::string& data) = 0;
};

class StructuredOutputController {
public:
    StructuredOutputController(const Tokenizer& tokenizer);
    void render_output(const std::string& data);

private:
    std::unique_ptr<IStructuredOutputBaseImpl> m_impl;
};

} // namespace genai
} // namespace ov
