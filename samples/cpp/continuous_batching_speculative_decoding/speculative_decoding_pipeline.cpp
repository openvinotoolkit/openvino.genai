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
    // std::cout << "=======STEP==================" << std::endl;
    std::vector<ov::genai::ContinuousBatchingPipeline::GeneratedSequence> candidate_sequences;
    if (m_is_speculative_mode) {
        // generate candidates using small model
        // std::cout << "num_candidates: " << candidates_number << std::endl;
        for (size_t i = 0; i < m_candidates_num; ++i) {
            auto start_time = std::chrono::system_clock::now();
            m_speculative_pipeline.step();
            auto end_time = std::chrono::system_clock::now();
            m_speculative_model_duration += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        }

        // put candidates to model cache
        candidate_sequences = m_speculative_pipeline.get_generated_sequences();
        // todo: remove debug code
        // for (const auto& s : candidate_sequences) {
        //     std::cout << "ASSISTANT: ";
        //     for (const auto& d : s.token_ids) {
        //         std::cout << d << " ";
        //     }
        //     std::cout << std::endl;
            // for (const auto& d : s.log_probs) {
            //     std::cout << d << " ";
            // }
            // std::cout << std::endl;
            // std::cout << decode(s.token_ids) << std::endl;
        // }

        for (const auto& candidate : candidate_sequences) {
            m_pipeline.update_generated_sequence(candidate);
        }
    }

    // validate candidates and generate 1 new token
    m_pipeline.step();

    if (m_is_speculative_mode) {
        // todo: iefode: remove debug prints
        auto checked_sequences = m_pipeline.get_generated_sequences();
        // todo: remove debug code
        // for (const auto& s : checked_sequences) {
        //     std::cout << "MODEL:     ";
        //     for (const auto& d : s.token_ids) {
        //         std::cout << d << " ";
        //     }
        //     std::cout << std::endl;
            // for (const auto& d : s.log_probs) {
            //     std::cout << d << " ";
            // }
            // std::cout << std::endl;
            // std::cout << decode(s.token_ids) << std::endl;
            // std::cout << std::endl;
        // }

        ov::genai::ContinuousBatchingPipeline::UpdateSeqResult update_result;
        for (const auto& checked_sequence : checked_sequences) {
            update_result = m_speculative_pipeline.update_generated_sequence(checked_sequence);
        }

        OPENVINO_ASSERT(m_candidates_num >= update_result.to_remove);
        // if (update_result.to_remove) {
        //     std::cout << "to_remove: " << update_result.to_remove << std::endl;
        // }
        update_strategy(m_candidates_num - update_result.to_remove);
        // std::cout << "=========================" << std::endl;
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
        m_speculative_pipeline.finish_all_requests();
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

inline size_t get_median(std::vector<size_t> values) {
    const auto size = values.size();
    if (size == 0) {
        return 0;
    }
    size_t offset = values.size() / 2;

    auto it = values.begin() + offset;
    std::nth_element(values.begin(), it, values.end());

    if (size % 2 != 0) {
        return *it;
    }
    auto it_1 = values.begin() + offset - 1;
    std::nth_element(values.begin(), it_1, values.end());
    return (*it + *it_1) / 2;
}


void SpeculativeDecodingPipeline::update_strategy(size_t num_matches) {
    // std::cout << "num_matches: " << num_matches << " m_candidates_num: " << m_candidates_num << std::endl;
    if (m_max_candidates_num == 0) {
        return;
    }

    if (m_max_matches < num_matches) {
        m_max_matches = num_matches;
    }
    if (num_matches == m_candidates_num) {
        m_candidates_num = std::min(std::max(m_candidates_num + 1, m_max_matches), m_max_candidates_num);
    } else {
        m_candidates_num = num_matches > 0 ? num_matches : std::max(get_median(m_matches_info), size_t(1));
    }
    m_matches_info.push_back(num_matches);
}