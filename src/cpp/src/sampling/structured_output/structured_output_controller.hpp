// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <memory>
#include <optional>
#include <vector>

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "sampling/logit_transformers.hpp"

namespace ov {
namespace genai {

class IStructuredOutputImpl {
public:
    virtual ~IStructuredOutputImpl() = default;
    virtual std::shared_ptr<ov::genai::LogitTransformers::ILogitTransformer>
        get_logits_transformer(const ov::genai::GenerationConfig& sampling_parameters) = 0;

};

class StructuredOutputController {
public:
    StructuredOutputController(const ov::genai::Tokenizer& tokenizer,
                              std::optional<int> vocab_size=std::nullopt);
    std::shared_ptr<ov::genai::LogitTransformers::ILogitTransformer>
        get_logits_transformer(const ov::genai::GenerationConfig& sampling_parameters);

private:
    std::unique_ptr<IStructuredOutputImpl> m_impl;
};

} // namespace genai
} // namespace ov
