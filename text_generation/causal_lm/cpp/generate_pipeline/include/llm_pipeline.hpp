// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// #pragma once

#include <openvino/openvino.hpp>
#include <openvino/core/any.hpp>
#include "generation_config.hpp"
#include "llm_tokenizer.hpp"
#include "streamer_base.hpp"
#include <filesystem>

using namespace std;

class Tokenizer; // forward declaration

namespace ov {

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
    
    GenerationConfig generation_config() const;

    std::string operator()(std::string text);

    std::string operator()(std::string text, GenerationConfig sampling_parameters);

    DecodedResults operator()(std::vector<std::string> text, GenerationConfig sampling_parameters);

    DecodedResults operator()(std::initializer_list<std::string> text, GenerationConfig sampling_parameters);

    EncodedResults generate(ov::Tensor input_ids, ov::Tensor attention_mask, GenerationConfig sampling_params);

    EncodedResults generate(ov::Tensor input_ids, ov::Tensor attention_mask);

    EncodedResults generate(ov::Tensor input_ids, GenerationConfig sampling_params);

    EncodedResults generate(ov::Tensor input_ids);

    ov::Tokenizer get_tokenizer();

    std::string apply_chat_template(std::string prompt, std::string role = "user") const;

    void set_streamer(std::function<void (std::string)> callback);
    void set_streamer(std::shared_ptr<StreamerBase> streamer);
    void set_streamer();
    void start_chat();
    void finish_chat();
    void reset_state();
    void set_default_config(const GenerationConfig& generation_config);
    // void set_default_config(const AnyMap& generation_config_map);

private:
    class LLMPipelineImpl;
    std::unique_ptr<LLMPipelineImpl> m_pimpl;

    std::string call(std::string text);
    std::string call(std::string text, GenerationConfig generation_config);
    DecodedResults call(std::vector<std::string> text, GenerationConfig sampling_parameters);
};

} // namespace ov
