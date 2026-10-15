// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gguf_tokenizer.hpp"
#include "openvino/genai/gguf.hpp"
#include "tokenizer/tokenizer_impl.hpp"

namespace ov::genai {

class GGUFTokenizerImpl : public Tokenizer::TokenizerImpl {
public:
    GGUFTokenizerImpl(const std::filesystem::path& gguf_path, const ov::AnyMap& properties);
    GGUFTokenizerImpl(const GGUFTokenizerParameters& parameters, const ov::AnyMap& properties);

private:
    void finalize_gguf_tokenizer(const std::shared_ptr<ov::Model>& tokenizer,
                                 const std::shared_ptr<ov::Model>& detokenizer,
                                 const std::map<std::string, GGUFMetaData>& config,
                                 const ov::AnyMap& properties,
                                 const std::filesystem::path& save_dir);
};

}  // namespace ov::genai
