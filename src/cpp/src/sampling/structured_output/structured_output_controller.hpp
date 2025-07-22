// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <memory>
#include <optional>
#include <vector>
#include <unordered_map>
#include <functional>

#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "sampling/logit_transformers.hpp"

namespace ov {
namespace genai {

/**
 * Below is the structure of the structured output generation system:
 * 
 * Example below is for XGrammar backend. XGrammarLogitsTransformer, XGrammarStructuredOutput can replace with any other 
 * structured output backend implementation.
 * 
 * +---------------------------+    uses   +--------------------------+   implements  +-----------------------+
 * | XGrammarLogitsTransformer |---------->| XGrammarStructuredOutput |-------------->| IStructuredOutputImpl |
 * +---------------------------+           +--------------------------+               +-----------------------+
 *                                                                                             ↑
 *                                                                                             | holds/used by
 *                                                                                             |
 *                                                                                 +----------------------------+
 *                                                                                 | StructuredOutputController |
 *                                                                                 +----------------------------+
 */

/**
 * @brief Helper interface for structured output implementations.
 *
 * IStructuredOutputImpl is an interface for creating instances of logit transformers,
 * such as XGrammarLogitsTransformer. This abstraction allows StructuredOutputController
 * to avoid direct dependencies on specific logit transformer implementations, keeping
 * the logic encapsulated within each IStructuredOutputImpl implementation. This design
 * also simplifies the process of extending support for new structured output backends.
 */
class IStructuredOutputImpl {
public:
    virtual ~IStructuredOutputImpl() = default;
    virtual std::shared_ptr<ov::genai::LogitTransformers::ILogitTransformer>
        get_logits_transformer(const ov::genai::GenerationConfig& sampling_parameters) = 0;
};

/**
 * @brief Orchestrates structured output generation.
 *
 * StructuredOutputController manages the selection and instantiation of structured output
 * backends via the IStructuredOutputImpl interface. Instances of this class are created for every LogitProcessor.
 * It registers backend factories, manages backend selection, and provides access to logit transformers
 * for structured output generation. This design enables easy extension to new backends and
 * keeps backend-specific logic encapsulated.
 */
class StructuredOutputController {
public:
    using BackendFactory = std::function<std::unique_ptr<ov::genai::IStructuredOutputImpl>(
        const ov::genai::Tokenizer&, std::optional<int>)>;

    StructuredOutputController(const ov::genai::Tokenizer& tokenizer,
                              std::optional<int> vocab_size=std::nullopt);


    std::shared_ptr<ov::genai::LogitTransformers::ILogitTransformer>
        get_logits_transformer(const ov::genai::GenerationConfig& sampling_parameters);

    static void register_backend(const std::string& name, BackendFactory factory);
    static void set_default_backend(const std::string& name);
    static std::string& get_default_backend_name();
    static std::unordered_map<std::string, BackendFactory>& get_backend_registry();
    
    std::pair<std::map<std::string, float>, std::vector<float>> get_times() const;
    void clear_compile_times();
private:
    std::map<std::string, float> m_init_grammar_compiler_times;
    std::vector<float> m_grammar_compile_times;
    std::unordered_map<std::string, std::unique_ptr<IStructuredOutputImpl>> m_impls;
    const ov::genai::Tokenizer& m_tokenizer;
    std::optional<int> m_vocab_size;
    std::mutex m_mutex;
};

} // namespace genai
} // namespace ov
