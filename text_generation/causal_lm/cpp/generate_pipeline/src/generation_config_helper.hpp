// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "generation_config.hpp"

namespace ov {


class GenerationConfigHelper {
public:
    GenerationConfig config;

    GenerationConfigHelper() = default;
    
    GenerationConfigHelper(const GenerationConfig& config): config(config) {};

    size_t get_max_new_tokens(size_t prompt_length = 0);
    
    // template <typename T>
    // static GenerationConfig assistive_decoding(T& assistant_model) {
    //     GenerationConfig assistive;
    //     assistive.assistant_model(assistant_model);
    //     return assistive;
    // }

    bool is_greedy_sampling() const;

    bool is_beam_search() const;

    bool is_multimomial() const;

    bool is_speculative() const;


    // // for speculative decoding
    // void set_assistant_model(const ov::InferRequest& assistant_model) {
    //     this->assistant_model = assistant_model;
    //     is_assistant_request_defined = true;
    // }

    // void set_assistant_model(ov::CompiledModel& assistant_model) {
    //     this->assistant_model = assistant_model.create_infer_request();
    //     is_assistant_request_defined = true;
    // }

    // void set_assistant_model(const std::shared_ptr<const ov::Model>& assistant_model) {
    //     m_assistant_ov_model = assistant_model;
    //     is_assistant_ov_defined = true;
    // }

    // void set_assistant_model(std::string assistant_model) {
    //     auto is_xml = [](std::string path) -> bool { return path.compare(path.length() - 4, 4, ".xml") == 0;};
    //     if (!is_xml(assistant_model))
    //         assistant_model += "/openvino_model.xml";

    //     m_assistant_ov_model = ov::Core().read_model(assistant_model);
    //     is_assistant_ov_defined = true;
    // }

    ov::InferRequest get_assistant_model(std::string device="CPU", const ov::AnyMap& config={});
    
    // void set_num_assistant_tokens(int64_t num_assistant_tokens) {
    //     this->num_assistant_tokens = num_assistant_tokens;
    // }

    // for Assistive/Speculative decoding
    ov::InferRequest assistant_model;
    size_t num_assistant_tokens = 5;
    size_t seq_len_axis = 2;
private:

    std::shared_ptr<const ov::Model> m_assistant_ov_model;
    bool is_assistant_request_defined = false;
    bool is_assistant_ov_defined = false;
};

} // namespace ov