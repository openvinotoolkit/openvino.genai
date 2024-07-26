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
        auto assisted_cache_size = cache_size * 0.25;
        cache_size -= assisted_cache_size;
        model_scheduler_config.cache_size = cache_size;
        assisting_scheduler_config.cache_size = assisted_cache_size;
    }
    // todo: iefode: tokenizer is not needed for CB
    model_pipeline = ov::genai::ContinuousBatchingPipeline(models_path, m_tokenizer, model_scheduler_config, device, plugin_config);
    assisting_pipeline = ov::genai::ContinuousBatchingPipeline(assisting_model_path, m_tokenizer, assisting_scheduler_config, device, plugin_config);
}

ov::genai::PipelineMetrics SpeculativeDecodingPipeline::get_metrics() const {
    return model_pipeline.get_metrics();
}

void SpeculativeDecodingPipeline::step() {
    ContinuousBatchingPipeline::GeneratedTokensMap candidate_sequences;
    std::cout << "K: " << k << std::endl;
    if (is_speculative_mode) {
        // generate candidates using small model
        for (size_t i = 0; i < k; ++i) {
            if (!assisting_pipeline.has_non_finished_requests()) {
                break;
            }
            assisting_pipeline.step();
        }

        // put candidates to model cache
        candidate_sequences = assisting_pipeline.get_generated_sequences();
        for (const auto& request : candidate_sequences) {
            const auto& request_id = request.first;
            for (const auto& sequence : request.second) {
                model_pipeline.update_generated_sequence(sequence.second.first, sequence.second.second, request_id, sequence.first);
            }
        }
    }

    // validate candidates and generate 1 new token
    model_pipeline.step();

    if (is_speculative_mode) {
        size_t max_removed_token_cnt = 0;
        auto checked_sequences = model_pipeline.get_generated_sequences();
        for (const auto& request : checked_sequences) {
            auto& request_id = request.first;
            for (const auto& sequence : request.second) {
                const auto& sequence_id = sequence.first;
                const auto& generated_sequence = sequence.second;
                const auto& candidate_sequence = candidate_sequences[request_id].at(sequence_id);
                const auto generated_sequence_size = generated_sequence.first.size();
                const auto candidate_sequence_size = candidate_sequence.first.size();
                if (generated_sequence_size <= candidate_sequence_size) {
                    const auto dist = candidate_sequence_size - generated_sequence_size + 1;
                    assisting_pipeline.remove_tokens_from_sequences(dist, request_id, sequence_id);
                    max_removed_token_cnt = std::max(max_removed_token_cnt, dist);
                }
                assisting_pipeline.update_generated_sequence(generated_sequence.first, generated_sequence.second, request_id, sequence_id);
            }
        }

        if (max_removed_token_cnt > 0) {
            k = k > max_removed_token_cnt ? k - max_removed_token_cnt : 1;
        } else {
            k = default_k;
        }
    }
}

void SpeculativeDecodingPipeline::set_k(size_t new_default_k) {
    default_k = new_default_k;
    k = default_k;
    is_speculative_mode = k > 0;
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
        assisting_generations.push_back(assisting_pipeline.add_request(request_id, prompts[request_id], sampling_params[request_id]));
    }

    while (has_non_finished_requests()) {
        step();
    }

    return generations;
}
