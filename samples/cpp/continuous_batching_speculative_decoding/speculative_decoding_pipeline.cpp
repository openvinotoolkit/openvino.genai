// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speculative_decoding_pipeline.hpp"

inline size_t get_kv_cache_size(const std::shared_ptr<ov::Model>& model) {
    const auto& parameters = model->get_parameters();
    // extract num_kv_heads and head_size
    size_t kv_caches_inputs_offset = 2;
    ov::PartialShape k_shape = parameters[kv_caches_inputs_offset]->get_partial_shape();
    OPENVINO_ASSERT(k_shape.rank().get_length() == 3, "KV cache shape is expected to have rank 3, while shape is ", k_shape);
    size_t num_kv_heads = k_shape[1].get_length(), head_size = k_shape[2].get_length();
    return num_kv_heads * head_size;
}

SpeculativeDecodingPipeline::SpeculativeDecodingPipeline(
    const std::string& models_path,
    const std::string& speculative_models_path,
    size_t start_candidates_number,
    const ov::genai::SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& plugin_config) {
    m_tokenizer = ov::genai::Tokenizer(models_path, plugin_config);

    m_candidates_num = start_candidates_number;
    m_max_candidates_num = m_candidates_num * 2;
    m_is_speculative_mode = m_candidates_num > 0;

    ov::Core core;
    std::shared_ptr<ov::Model> model = ov::genai::read_model_and_apply_paged_attention(models_path, core),
                               assisting_model = ov::genai::read_model_and_apply_paged_attention(speculative_models_path, core);

    ov::genai::SchedulerConfig model_scheduler_config = scheduler_config,
                               assisting_scheduler_config = scheduler_config;
    if (m_is_speculative_mode) {
        // split KV cache to 2 caches for speculative and base parts
        size_t model_cache_size = get_kv_cache_size(model),
               assisting_cache_size = get_kv_cache_size(assisting_model);
        auto k = float(assisting_cache_size) / (model_cache_size + assisting_cache_size);
        auto cache_size = scheduler_config.num_kv_blocks;
        auto assisted_cache_size = size_t(cache_size * k);
        cache_size -= assisted_cache_size;
        model_scheduler_config.num_kv_blocks = cache_size;
        assisting_scheduler_config.num_kv_blocks = assisted_cache_size;
        m_speculative_pipeline = ov::genai::ContinuousBatchingPipeline(core, assisting_model, m_tokenizer, assisting_scheduler_config, device, plugin_config, false);
    }

    m_pipeline = ov::genai::ContinuousBatchingPipeline(core, model, m_tokenizer, model_scheduler_config, device, plugin_config, true);
}

void SpeculativeDecodingPipeline::step() {
    std::vector<ov::genai::ContinuousBatchingPipeline::GeneratedSequence> candidate_sequences;
    if (m_is_speculative_mode) {
        // find minimum(candidates_number, seq_len) to generate candidates
        size_t min_candidates_number = m_candidates_num;
        for (auto& request : m_to_generate_length) {
            if (request.second < min_candidates_number && request.second > 0) {
                min_candidates_number = request.second;
            }
        }
        // generate candidates by speculative model
        for (size_t i = 0; i < min_candidates_number; ++i) {
            m_speculative_pipeline.step();
        }

        // put candidates to model KV cache
        candidate_sequences = m_speculative_pipeline.get_generated_sequences();
        for (const auto& candidate : candidate_sequences) {
            m_pipeline.update_generated_sequence(candidate);
        }
    }

    // validate candidates and generate 1 new token
    m_pipeline.step();

    if (m_is_speculative_mode) {
        auto checked_sequences = m_pipeline.get_generated_sequences();
        size_t max_removed_token_cnt = 0;
        for (const auto& checked_sequence : checked_sequences) {
            auto update_result = m_speculative_pipeline.update_generated_sequence(checked_sequence);
            max_removed_token_cnt = std::max(max_removed_token_cnt, update_result.to_remove);
        }
        OPENVINO_ASSERT(m_candidates_num >= max_removed_token_cnt);
        auto num_matches = m_candidates_num - max_removed_token_cnt;
        update_strategy(num_matches);

        // update to generate tokens
        for (auto& request : m_to_generate_length) {
            if (request.second > num_matches) {
                request.second -= (num_matches + 1);
            } else {
                request.second = 0;
                m_speculative_pipeline.finish_request(request.first);
            }
        }
    }
}

std::vector<ov::genai::GenerationResult>
SpeculativeDecodingPipeline::generate(const std::vector<std::string>& prompts,
                                      const std::vector<ov::genai::GenerationConfig>& sampling_params) {
    OPENVINO_ASSERT(!m_pipeline.has_non_finished_requests(), "Generate cannot be called while ContinuousBatchingPipeline is already in running state. Use ContinuousBatchingPipeline::add_request");
    OPENVINO_ASSERT(prompts.size() == sampling_params.size());

    std::vector<ov::genai::GenerationHandle> generations, speculative_generations;
    for (size_t request_id = 0; request_id < prompts.size(); ++request_id) {
        generations.push_back(m_pipeline.add_request(request_id, prompts[request_id], sampling_params[request_id]));
        m_to_generate_length.insert({ request_id, sampling_params[request_id].max_new_tokens });
        if (m_is_speculative_mode) {
            auto assisting_sampling_params = sampling_params[request_id];
            assisting_sampling_params.max_new_tokens += m_max_candidates_num;
            assisting_sampling_params.min_new_tokens += m_max_candidates_num;
            speculative_generations.push_back(m_speculative_pipeline.add_request(request_id, prompts[request_id], assisting_sampling_params));
        }
    }

    while (m_pipeline.has_non_finished_requests()) {
        step();
    }
    if (m_is_speculative_mode) {
        // finish all speculative requests
        m_speculative_pipeline.finish_request(-1);
    }

    std::vector<ov::genai::EncodedGenerationResult> encoded_results;
    for (size_t generation_idx = 0; generation_idx < generations.size(); ++generation_idx) {
        const auto& generation = generations[generation_idx];
        ov::genai::EncodedGenerationResult result;
        result.m_request_id = 1;
        std::vector<ov::genai::GenerationOutput> generation_outputs = generation->read_all();
        std::sort(generation_outputs.begin(), generation_outputs.end(), [=] (ov::genai::GenerationOutput& r1, ov::genai::GenerationOutput& r2) {
            return r1.score > r2.score;
        });

        auto num_outputs = std::min(sampling_params[generation_idx].num_return_sequences, generation_outputs.size());
        for (size_t generation_output_idx = 0; generation_output_idx < num_outputs; ++generation_output_idx) {
            const auto& generation_output = generation_outputs[generation_output_idx];
            result.m_generation_ids.push_back(std::move(generation_output.generated_token_ids));
            result.m_scores.push_back(generation_output.score);
        }
        result.m_status = generation->get_status();
        encoded_results.push_back(std::move(result));
    }

    OPENVINO_ASSERT(encoded_results.size() == prompts.size());

    std::vector<ov::genai::GenerationResult> decoded_results;
    for (ov::genai::EncodedGenerationResult& res : encoded_results) {
        std::vector<std::string> generated;
        generated.reserve(res.m_generation_ids.size());
        for (size_t idx = 0; idx < res.m_generation_ids.size(); ++idx) {
            generated.push_back(m_tokenizer.decode(res.m_generation_ids.at(idx)));
        }
        decoded_results.push_back(ov::genai::GenerationResult{
            res.m_request_id,
            std::move(generated),
            std::move(res.m_scores),
            res.m_status
        });
    }
    return decoded_results;
}

void SpeculativeDecodingPipeline::update_strategy(const size_t num_matches) {
    // dynamically adjust number of generated candidates based on number of matches
    // we want to balance the benefits of getting candidates tokens correct with the
    // cost of forecasting incorrect candidates tokens.
    if (m_max_candidates_num == 0) {
        return;
    }
    if (num_matches == m_candidates_num) {
        m_candidates_num = std::min(m_candidates_num + 2, m_max_candidates_num);
    } else {
        m_candidates_num = std::max(int64_t(m_candidates_num) - 1, int64_t(1));
    }
}