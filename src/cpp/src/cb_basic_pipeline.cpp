// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/cb_basic_pipeline.hpp"

ov::Tensor ov::genai::BasicPipeline::encode(const std::string& prompt) {
    return m_tokenizer.encode(prompt).input_ids;
}

std::vector<ov::Tensor> ov::genai::BasicPipeline::encode(const std::vector<std::string>& prompts) {
    std::vector<ov::Tensor> tensors;
    const size_t prompt_num = prompts.size();
    tensors.resize(prompt_num);
    for (size_t i = 0; i < prompt_num; ++i) {
        tensors[i] = m_tokenizer.encode(prompts[i]).input_ids;
    }
    return tensors;
}

std::string ov::genai::BasicPipeline::decode(const std::vector<int64_t>& line) {
    return m_tokenizer.decode(line);
}

std::vector<ov::genai::GenerationResult>
ov::genai::BasicPipeline::process_generated_sequences(
    const std::vector<ov::genai::GenerationHandle>& generations, 
    std::vector<ov::genai::GenerationConfig> sampling_params) {
    std::vector<ov::genai::GenerationResult> results;
    results.reserve(sampling_params.size());

    for (size_t generation_idx = 0; generation_idx < generations.size(); ++generation_idx) {
        const auto& generation = generations[generation_idx];
        ov::genai::GenerationResult result;
        result.m_request_id = 1;
        std::vector<ov::genai::GenerationOutput> generation_outputs = generation->read_all();
        std::sort(generation_outputs.begin(), generation_outputs.end(), [=] (ov::genai::GenerationOutput& r1, ov::genai::GenerationOutput& r2) {
            return r1.score > r2.score;
        });

        auto num_outputs = std::min(sampling_params[generation_idx].num_return_sequences, generation_outputs.size());
        for (size_t generation_output_idx = 0; generation_output_idx < num_outputs; ++generation_output_idx) {
            const auto& generation_output = generation_outputs[generation_output_idx];
            std::string output_text = decode(generation_output.generated_token_ids);
            result.m_generation_ids.push_back(output_text);
            result.m_scores.push_back(generation_output.score);
        }
        result.m_status = generation->get_status();
        results.push_back(result);
    }
    return results;
} 

ov::genai::Tokenizer ov::genai::BasicPipeline::get_tokenizer() {
    return m_tokenizer;
};

// ov::genai::GenerationConfig ov::genai::BasicPipeline::get_config() const {
//     return m_generation_config;
// };

// ov::genai::GenerationHandle
// ov::genai::BasicPipeline::add_request(uint64_t request_id, std::string prompt, ov::genai::GenerationConfig sampling_params) {
//     sampling_params.validate();
//     ov::Tensor encoded_prompt = encode(prompt);
//     return add_request(request_id, encoded_prompt, sampling_params);
// };

std::vector<ov::genai::GenerationResult>
ov::genai::BasicPipeline::generate(
    const std::vector<std::string>& prompts,
    std::vector<ov::genai::GenerationConfig> sampling_params) {
    std::vector<ov::Tensor> tokenized_prompts = encode(prompts);
    std::vector<ov::genai::GenerationHandle> sequences = generate_sequences(tokenized_prompts, sampling_params);
    std::vector<ov::genai::GenerationResult> results = process_generated_sequences(sequences, sampling_params);
    OPENVINO_ASSERT(results.size() == prompts.size());
    return results;
}

