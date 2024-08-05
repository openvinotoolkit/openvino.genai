// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speculative_decoding_pipeline.hpp"

SpeculativeDecodingPipeline::SpeculativeDecodingPipeline(const std::string& models_path,
    const std::string& assisting_model_path,
    const ov::genai::SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& plugin_config) {
    ov::genai::Tokenizer tokenizer(models_path);
    SpeculativeDecodingPipeline(models_path, assisting_model_path, tokenizer, scheduler_config, device, plugin_config);
};

SpeculativeDecodingPipeline::SpeculativeDecodingPipeline(const std::string& models_path,
                                const std::string& assisting_model_path,
                                const ov::genai::Tokenizer& tokenizer,
                                const ov::genai::SchedulerConfig& scheduler_config,
                                const std::string& device,
                                const ov::AnyMap& plugin_config) {
    m_tokenizer = tokenizer;
    ov::genai::SchedulerConfig model_scheduler_config = scheduler_config;
    ov::genai::SchedulerConfig assisting_scheduler_config = scheduler_config;
    {
        auto cache_size = scheduler_config.cache_size;
        auto assisted_cache_size = cache_size * 0.45;
        cache_size -= assisted_cache_size;
        model_scheduler_config.cache_size = cache_size;
        assisting_scheduler_config.cache_size = assisted_cache_size;
    }
    model_pipeline = ov::genai::ContinuousBatchingPipeline(models_path, m_tokenizer, model_scheduler_config, device, plugin_config);
    model_pipeline.enable_validation_mode();
    assisting_pipeline = ov::genai::ContinuousBatchingPipeline(assisting_model_path, m_tokenizer, assisting_scheduler_config, device, plugin_config);
}

ov::genai::PipelineMetrics SpeculativeDecodingPipeline::get_metrics() const {
    return model_pipeline.get_metrics();
}

void SpeculativeDecodingPipeline::step() {
    std::vector<ov::genai::ContinuousBatchingPipeline::GeneratedSequence> candidate_sequences;
    if (is_speculative_mode) {
        // generate candidates using small model
        // std::cout << "num_candidates: " << candidates_number << std::endl;
        for (size_t i = 0; i < candidates_number; ++i) {
            assisting_pipeline.step();
        }

        // put candidates to model cache
        candidate_sequences = assisting_pipeline.get_generated_sequences();
        // todo: remove debug code
        // for (const auto& s : candidate_sequences) {
        //     std::cout << "ASSISTANT: ";
        //     for (const auto& d : s.token_ids) {
        //         std::cout << d << " ";
        //     }
        //     std::cout << std::endl;
        //     for (const auto& d : s.log_probs) {
        //         std::cout << d << " ";
        //     }
        //     std::cout << std::endl;
        //     std::cout << decode(s.token_ids) << std::endl;
        // }

        for (const auto& candidate : candidate_sequences) {
            model_pipeline.update_generated_sequence(candidate);
        }
    }

    // validate candidates and generate 1 new token
    model_pipeline.step();

    if (is_speculative_mode) {
        // todo: iefode: remove debug prints
        auto checked_sequences = model_pipeline.get_generated_sequences();
        // todo: remove debug code
        // for (const auto& s : checked_sequences) {
        //     std::cout << "MODEL:     ";
        //     for (const auto& d : s.token_ids) {
        //         std::cout << d << " ";
        //     }
        //     std::cout << std::endl;
        //     for (const auto& d : s.log_probs) {
        //         std::cout << d << " ";
        //     }
        //     std::cout << std::endl;
        //     std::cout << decode(s.token_ids) << std::endl;
        //     std::cout << std::endl;
        // }

        ov::genai::ContinuousBatchingPipeline::UpdateSeqResult update_result;
        for (const auto& checked_sequence : checked_sequences) {
            update_result = assisting_pipeline.update_generated_sequence(checked_sequence);
        }

        if (candidates_number < update_result.to_remove)
            auto a = 0;

        OPENVINO_ASSERT(candidates_number >= update_result.to_remove);
        update_strategy(candidates_number - update_result.to_remove);
    }
}

void SpeculativeDecodingPipeline::update_strategy(size_t num_matches) {
    // std::cout << "num_matches: " << num_matches << std::endl;
    if (num_matches == candidates_number) {
        candidates_number = std::min(candidates_number + 2, max_candidates_number);
    } else {
        candidates_number = std::max(int64_t(candidates_number) - 1, int64_t(1));
    }
}


void SpeculativeDecodingPipeline::set_k(size_t new_default_k) {
    candidates_number = new_default_k;
    max_candidates_number = new_default_k * 2;
    is_speculative_mode = candidates_number > 0;
}

bool SpeculativeDecodingPipeline::has_non_finished_requests() {
    return model_pipeline.has_non_finished_requests();
}


std::vector<ov::genai::GenerationHandle>
SpeculativeDecodingPipeline::generate_sequences(
    const std::vector<ov::Tensor> prompts,
    std::vector<ov::genai::GenerationConfig> sampling_params) {
    OPENVINO_ASSERT(!has_non_finished_requests(), "Generate cannot be called while ContinuousBatchingPipeline is already in running state. Use ContinuousBatchingPipeline::add_request");
    OPENVINO_ASSERT(prompts.size() == sampling_params.size());

    std::vector<ov::genai::GenerationHandle> generations, assisting_generations;
    for (size_t request_id = 0; request_id < prompts.size(); ++request_id) {
        generations.push_back(model_pipeline.add_request(request_id, prompts[request_id], sampling_params[request_id]));
        auto assisting_sampling_params = sampling_params[request_id];
        assisting_sampling_params.max_new_tokens += max_candidates_number;
        assisting_sampling_params.min_new_tokens += max_candidates_number;
        assisting_generations.push_back(assisting_pipeline.add_request(request_id, prompts[request_id], assisting_sampling_params));
        // assisting_generations.push_back(assisting_pipeline.add_request(request_id, prompts[request_id], sampling_params[request_id]));
    }

    while (has_non_finished_requests()) {
        step();
    }
    assisting_pipeline.finish_all_requests();

    return generations;
}
