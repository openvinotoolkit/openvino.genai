// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <memory>

#include "openvino/genai/tokenizer.hpp"

namespace ov {
namespace genai {

namespace LogitTransformers {

class ILogitTransformer;
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
}
} // namespace LogitTransformers

class IStructuredOutputBaseImpl {
public:
    virtual ~IStructuredOutputBaseImpl() = default;
    virtual void render_output(const std::string& data) = 0;
    virtual LogitTransformers::IStructuredOutputBaseLogitTransformer get_json_logits_transformer(const std::string& json_schema) = 0;
};

class StructuredOutputController {
public:
    StructuredOutputController(const Tokenizer& tokenizer,
                               std::optional<int> vocab_size=std::nullopt);
    LogitTransformers::IStructuredOutputBaseLogitTransformer get_json_schema_logits_transformer(const std::string& json_schema);

private:
    std::unique_ptr<IStructuredOutputBaseImpl> m_impl;
};

} // namespace genai
} // namespace ov
