// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// #pragma once

#include <openvino/openvino.hpp>
#include <openvino/core/any.hpp>
#include "generation_config.hpp"
#include "llm_tokenizer.hpp"
#include "streamer_base.hpp"
#include <filesystem>
#include <optional>

using namespace std;

class Tokenizer; // forward declaration

namespace ov {


using StreamerVariant = std::variant<std::monostate, std::function<void (std::string)>, std::shared_ptr<StreamerBase>>;
using OptionalGenerationConfig = std::optional<GenerationConfig>;

class EncodedResults {
public:
    std::vector<std::vector<int64_t>> tokens;
    std::vector<float> scores;
};

class DecodedResults {
public:
    std::vector<std::string> texts;
    std::vector<float> scores;
};

class LLMPipeline {
public:
    LLMPipeline(
        std::string& model_path,
        std::string& tokenizer_path,  // todo: make possible to specify tokenizers with ov::Model, ov::CompiledModel, etc. 
        std::string& detokenizer_path,
        std::string device="CPU",
        const ov::AnyMap& plugin_config={}
    );
    
    LLMPipeline(std::string& path, std::string device="CPU", const ov::AnyMap& plugin_config={});
    
    ~LLMPipeline();
    
    EncodedResults generate(ov::Tensor input_ids, std::optional<ov::Tensor> attention_mask, OptionalGenerationConfig generation_config);
    std::string generate(std::string text, OptionalGenerationConfig generation_config);
    DecodedResults generate(std::vector<std::string> text, OptionalGenerationConfig generation_config);

    std::string operator()(std::string text, OptionalGenerationConfig generation_config);
    DecodedResults operator()(std::vector<std::string> text, OptionalGenerationConfig generation_config);
    DecodedResults operator()(std::initializer_list<std::string> text, OptionalGenerationConfig generation_config);

    // generate with streamers
    std::string generate(std::string text, OptionalGenerationConfig generation_config, StreamerVariant streamer);
    std::string operator()(std::string text, OptionalGenerationConfig generation_config, StreamerVariant streamer);
    std::string operator()(std::string text, StreamerVariant streamer);
    
    ov::Tokenizer get_tokenizer();
    GenerationConfig get_generation_config() const;
    void set_generation_config(const GenerationConfig& generation_config);

    void start_chat();
    void finish_chat();
    void reset_state();
    std::string apply_chat_template(std::string prompt, std::string role = "user") const;
private:
    class LLMPipelineImpl;
    std::unique_ptr<LLMPipelineImpl> m_pimpl;
};

} // namespace ov
