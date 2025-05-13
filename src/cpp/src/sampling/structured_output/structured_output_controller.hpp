// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <memory>
#include <optional>
#include <vector>

#include "openvino/genai/tokenizer.hpp"
#include "sampling/logit_processor.hpp"

namespace ov {
namespace genai {

namespace LogitTransformers {
//class ILogitTransformer {
//public:
//    virtual void apply(Logits& logits) = 0;
//
//    virtual bool is_applicable(size_t generated_tokens_cnt = 0) {
//        return true;
//    }
//};


class IStructuredOutputBaseLogitTransformer: public ILogitTransformer {
public:
    void register_sampled_token(const TokenIds& input_ids);
//private:
//    std::vector<TokenIds> m_sampled_tokens;
};
} // namespace LogitTransformers

class IStructuredOutputBaseImpl {
public:
    virtual ~IStructuredOutputBaseImpl() = default;
    virtual LogitTransformers::IStructuredOutputBaseLogitTransformer get_logits_transformer(const GenerationConfig& sampling_parameters) = 0;
};

class StructuredOutputController {
public:
    StructuredOutputController(const Tokenizer& tokenizer,
                               std::optional<int> vocab_size=std::nullopt);
    LogitTransformers::IStructuredOutputBaseLogitTransformer get_logits_transformer(const GenerationConfig& sampling_parameters);

private:
    std::unique_ptr<IStructuredOutputBaseImpl> m_impl;
};

} // namespace genai
} // namespace ov
