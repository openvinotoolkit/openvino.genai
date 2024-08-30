// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "prompt_lookup_pipeline.hpp"

PromptLookupPipeline::PromptLookupPipeline(const std::string& models_path,
    size_t candidates_number,
    size_t ngram_size,
    const ov::genai::SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& plugin_config) {
    ov::genai::Tokenizer tokenizer(models_path);
    PromptLookupPipeline(models_path, candidates_number, max_ngram_size, tokenizer, scheduler_config, device, plugin_config);
};

PromptLookupPipeline::PromptLookupPipeline(const std::string& models_path,
                                size_t candidates_number,
                                size_t ngram_size,
                                const ov::genai::Tokenizer& tokenizer,
                                const ov::genai::SchedulerConfig& scheduler_config,
                                const std::string& device,
                                const ov::AnyMap& plugin_config) {
    m_tokenizer = tokenizer;
    set_k(candidates_number);
    max_ngram_size = ngram_size;

    model_pipeline = ov::genai::ContinuousBatchingPipeline(models_path, m_tokenizer, scheduler_config, device, plugin_config);
    model_pipeline.enable_validation_mode();
}

ov::genai::PipelineMetrics PromptLookupPipeline::get_metrics() const {
    return model_pipeline.get_metrics();
}

void PromptLookupPipeline::step() {
    std::cout << "=======STEP==================" << std::endl;
    bool is_updated = false;
    if (is_speculative_mode) {
        // predict tokens using prompt
        std::cout << "num_candidates: " << candidates_number << std::endl;
        for (const auto& whole_input : model_pipeline.get_prompts_with_generated_tokens()) {
            auto updated_input = whole_input;
            const auto& input_ids = whole_input.token_ids;
            const size_t input_length = input_ids.size();
            for (int32_t ngram_size = max_ngram_size; ngram_size > 0; ngram_size--) {
                std::vector<int64_t> ngram = std::vector<int64_t>{input_ids.cend() - ngram_size, input_ids.cend()};
                std::cout << "ngram: " << std::endl;
                for (const auto& a : ngram) {
                    std::cout << a;
                }
                std::cout << std::endl;

                // find ngram match in input_ids
                size_t ngram_i = 0;
                for (size_t input_i = 0; input_i < input_length - ngram_size; input_i++) {
                    if (ngram[ngram_i] != input_ids[input_i]) {
                        ngram_i = 0;
                        continue;
                    }
                    ngram_i++;

                    if (ngram_i < ngram_size) {
                        continue;
                    }

                    // match found with the end at input_i
                    size_t avaliable_num_pred = std::min(input_length - (input_i + 1), candidates_number);

                    // return candidates with length of avaliable_num_pred
                    std::vector<int64_t> candidate{input_ids.cbegin() + input_i + 1,
                                                   input_ids.cbegin() + input_i + 1 + avaliable_num_pred};
                    updated_input.token_ids = candidate;
                    updated_input.log_probs = std::vector<float>(candidate.size(), 0);

                    model_pipeline.update_generated_sequence(updated_input);
                    break;
                }
                if (whole_input.token_ids != updated_input.token_ids) {
                    is_updated = true;
                    break;
                }
            }
        }

        // put candidates to model cache
        auto candidate_sequences = model_pipeline.get_generated_sequences();
        // todo: remove debug code
        for (const auto& s : candidate_sequences) {
            std::cout << "ASSISTANT: ";
            for (const auto& d : s.token_ids) {
                std::cout << d << " ";
            }
            // std::cout << std::endl;
            // for (const auto& d : s.log_probs) {
            //     std::cout << d << " ";
            // }
            std::cout << std::endl;
            std::cout << decode(s.token_ids) << std::endl;
        }
    }

    const auto gen_seq_before = model_pipeline.get_generated_sequences();

    // validate candidates and generate 1 new token
    model_pipeline.step();

    if (is_speculative_mode && is_updated) {
        // todo: remove debug code
        for (const auto& s : model_pipeline.get_generated_sequences()) {
            std::cout << "MODEL:     ";
            for (const auto& d : s.token_ids) {
                std::cout << d << " ";
            }
            // std::cout << std::endl;
            // for (const auto& d : s.log_probs) {
            //     std::cout << d << " ";
            // }
            std::cout << std::endl;
            std::cout << decode(s.token_ids) << std::endl;
            std::cout << std::endl;
        }

        // todo: iefode: remove debug prints
        for (const auto& gen_seq_after : model_pipeline.get_generated_sequences()) {
            const auto& candidate_seq = gen_seq_before[gen_seq_after.request_id];
            size_t before_len = candidate_seq.token_ids.size(),
                   after_len = gen_seq_after.token_ids.size();
            size_t dist = is_updated ? (after_len <= before_len ? (before_len - after_len) : candidates_number) : 0;
            update_strategy(dist);
        }
        // ov::genai::ContinuousBatchingPipeline::UpdateSeqResult update_result;
        // for (const auto& checked_sequence : checked_sequences) {
        //     update_result = assisting_pipeline.update_generated_sequence(checked_sequence);
        // }

        // OPENVINO_ASSERT(candidates_number >= update_result.to_remove);
        // if (update_result.to_remove) {
        //     std::cout << "to_remove: " << update_result.to_remove << std::endl;
        // }
        // update_strategy(candidates_number - update_result.to_remove);
        // std::cout << "=========================" << std::endl;
    }
}

void PromptLookupPipeline::update_strategy(size_t num_matches) {
    std::cout << "num_matches: " << num_matches << std::endl;
    max_matches = std::max(max_matches, num_matches);
    avg_matches += num_matches;
    if (max_candidates_number == 0) {
        return;
    }
    if (num_matches == candidates_number) {
        candidates_number = std::min(candidates_number + 2, max_candidates_number);
    } else {
        candidates_number = std::max(int64_t(candidates_number) - 1, int64_t(1));
    }
}


void PromptLookupPipeline::set_k(size_t new_default_k) {
    candidates_number = new_default_k;
    max_candidates_number = new_default_k * 2;
    is_speculative_mode = candidates_number > 0;
}

bool PromptLookupPipeline::has_non_finished_requests() {
    return model_pipeline.has_non_finished_requests();
}


std::vector<ov::genai::GenerationHandle>
PromptLookupPipeline::generate_sequences(
    const std::vector<ov::Tensor> prompts,
    std::vector<ov::genai::GenerationConfig> sampling_params) {
    OPENVINO_ASSERT(!has_non_finished_requests(), "Generate cannot be called while ContinuousBatchingPipeline is already in running state. Use ContinuousBatchingPipeline::add_request");
    OPENVINO_ASSERT(prompts.size() == sampling_params.size());

    std::vector<ov::genai::GenerationHandle> generations, assisting_generations;
    for (size_t request_id = 0; request_id < prompts.size(); ++request_id) {
        generations.push_back(model_pipeline.add_request(request_id, prompts[request_id], sampling_params[request_id]));
    }

    while (has_non_finished_requests()) {
        step();
        infer_cnt++;
    }

    return generations;
}