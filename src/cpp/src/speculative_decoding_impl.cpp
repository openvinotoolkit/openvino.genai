// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "text_callback_streamer.hpp"
#include "speculative_decoding_impl.hpp"
#include "paged_attention_transformations.hpp"

namespace ov::genai {
template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

ContinuousBatchingPipeline::SpeculativeDecodingImpl::SpeculativeDecodingImpl(
    const std::string& main_models_path,
    const std::string& draft_models_path,
    const Tokenizer& tokenizer,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& plugin_config) {
    ov::Core core;
    std::shared_ptr<ov::Model> main_model = core.read_model(main_models_path + "/openvino_model.xml"),
                               draft_model = core.read_model(draft_models_path + "/openvino_model.xml");

    apply_paged_attention_transformations(main_model, scheduler_config.use_cache_eviction);
    apply_paged_attention_transformations(draft_model, scheduler_config.use_cache_eviction);

    ov::genai::SchedulerConfig main_scheduler_config = scheduler_config,
                               draft_scheduler_config = scheduler_config;
    
    // split KV cache to 2 caches for main and draft models
    size_t main_model_cache_size = get_kv_cache_size(main_model),
           draft_model_cache_size = get_kv_cache_size(draft_model);
    auto k = static_cast<float>(draft_model_cache_size) / (main_model_cache_size + draft_model_cache_size);

    size_t main_cache_size = scheduler_config.cache_size * (1 - k),
           draft_cache_size = scheduler_config.cache_size * k;
    if (draft_cache_size == 0) {
        main_cache_size -= main_cache_size > 1 ? 1 : 0;
        draft_cache_size = 1;
    }

    main_scheduler_config.cache_size = main_cache_size;
    draft_scheduler_config.cache_size = draft_cache_size;

    DeviceConfig main_device_config(core, main_scheduler_config, device, plugin_config),
                 draft_device_config(core, draft_scheduler_config, device, plugin_config);

    set_kv_cache_type_and_shape(main_model, main_device_config);
    set_kv_cache_type_and_shape(draft_model, draft_device_config);

    m_main_pipeline = std::make_shared<ContinuousBatchingImpl>(core, main_model, tokenizer, main_device_config, main_scheduler_config, device, plugin_config, true);
    m_draft_pipeline = std::make_shared<ContinuousBatchingImpl>(core, draft_model, tokenizer, draft_device_config, draft_scheduler_config, device, plugin_config, false);
}

GenerationHandle
ContinuousBatchingPipeline::SpeculativeDecodingImpl::add_request(uint64_t request_id,
                                                                 const ov::Tensor& input_ids,
                                                                 ov::genai::GenerationConfig sampling_params) {
    m_draft_pipeline->add_request(request_id, input_ids, sampling_params);
    return m_main_pipeline->add_request(request_id, input_ids, sampling_params);
};

GenerationHandle
ContinuousBatchingPipeline::SpeculativeDecodingImpl::add_request(uint64_t request_id,
                                                                 const std::string& prompt,
                                                                 ov::genai::GenerationConfig sampling_params) {
    m_draft_pipeline->add_request(request_id, prompt, sampling_params);
    return m_main_pipeline->add_request(request_id, prompt, sampling_params);
}

bool ContinuousBatchingPipeline::SpeculativeDecodingImpl::has_non_finished_requests() {
    return m_main_pipeline->has_non_finished_requests();
}

void ContinuousBatchingPipeline::SpeculativeDecodingImpl::update_strategy(size_t num_matches) {
    // dynamically adjust number of generated candidates based on number of matches
    // we want to balance the benefits of getting candidates tokens correct with the
    // cost of forecasting incorrect candidates tokens.
    if (m_num_candidates == 0) {
        return;
    }
    if (num_matches == m_num_candidates) {
        m_num_candidates = std::min(m_num_candidates + 2, m_max_num_candidates);
    } else {
        m_num_candidates = std::max(int64_t(m_num_candidates) - 1, int64_t(1));
    }
}

void ContinuousBatchingPipeline::SpeculativeDecodingImpl::step() {
    std::vector<ov::genai::ContinuousBatchingPipeline::ContinuousBatchingImpl::GeneratedSequence> candidate_sequences;
    // find minimum(candidates_number, seq_len) to generate candidates
    size_t min_candidates_number = m_num_candidates;
    for (auto& request : m_to_generate_length) {
        if (request.second < min_candidates_number && request.second > 0) {
            min_candidates_number = request.second;
        }
    }
    // generate candidates by speculative model
    for (size_t i = 0; i < min_candidates_number; ++i) {
        m_draft_pipeline->step();
    }

    // put candidates to model KV cache
    candidate_sequences = m_draft_pipeline->get_generated_sequences();
    for (const auto& candidate : candidate_sequences) {
        m_main_pipeline->update_generated_sequence(candidate);
    }

    // validate candidates and generate 1 new token
    m_main_pipeline->step();

    auto checked_sequences = m_main_pipeline->get_generated_sequences();
    size_t max_removed_token_cnt = 0;
    for (const auto& checked_sequence : checked_sequences) {
        auto update_result = m_draft_pipeline->update_generated_sequence(checked_sequence);
        max_removed_token_cnt = std::max(max_removed_token_cnt, update_result.to_remove);
    }
    OPENVINO_ASSERT(m_max_num_candidates >= max_removed_token_cnt);
    auto num_matches = m_max_num_candidates - max_removed_token_cnt;
    update_strategy(num_matches);

    // update to generate tokens
    for (auto& request : m_to_generate_length) {
        if (request.second > num_matches) {
            request.second -= (num_matches + 1);
        } else {
            request.second = 0;
            m_draft_pipeline->finish_request(request.first);
        }
    }
}

std::vector<EncodedGenerationResult>
ContinuousBatchingPipeline::SpeculativeDecodingImpl::generate(const std::vector<ov::Tensor>& input_ids,
                                                              const std::vector<GenerationConfig>& sampling_params,
                                                              const StreamerVariant& streamer) {
    OPENVINO_ASSERT(!has_non_finished_requests(), "Generate cannot be called while ContinuousBatchingPipeline is already in running state. Use ContinuousBatchingPipeline::add_request");
    OPENVINO_ASSERT(input_ids.size() == sampling_params.size());
    const std::shared_ptr<StreamerBase>& streamer_ptr = std::visit(overloaded{
        [](std::monostate) -> std::shared_ptr<StreamerBase> {
            return nullptr;
        },
        [](const std::shared_ptr<StreamerBase>& streamer) {
            return streamer;
        },
        [this](const std::function<bool(std::string)>& streamer) -> std::shared_ptr<StreamerBase> {
            return std::make_unique<TextCallbackStreamer>(m_tokenizer, streamer);
        }
    }, streamer);

    std::vector<GenerationHandle> main_generations, draft_generations;
    for (size_t request_id = 0; request_id < input_ids.size(); ++request_id) {
        OPENVINO_ASSERT(1 == input_ids[request_id].get_shape().at(0), "Use multiple tensors to pass a batch.");
        main_generations.push_back(m_main_pipeline->add_request(request_id, input_ids[request_id], sampling_params[request_id]));

        auto draft_sampling_params = sampling_params[request_id];
        draft_sampling_params.max_new_tokens = SIZE_MAX;
        draft_sampling_params.min_new_tokens = 0;
        draft_generations.push_back(m_draft_pipeline->add_request(request_id, input_ids[request_id], draft_sampling_params));
    }

    std::vector<EncodedGenerationResult> results;
    results.reserve(input_ids.size());

    bool continue_generation = true;
    while (has_non_finished_requests() && continue_generation) {
        step();
            if (streamer_ptr) {
            std::unordered_map<uint64_t, GenerationOutput> token = main_generations.at(0).get()->back();
            OPENVINO_ASSERT(1 == token.size());
            OPENVINO_ASSERT(1 == token.begin()->second.generated_ids.size());
            continue_generation = !streamer_ptr->put(token.begin()->second.generated_ids.at(0));
        }
    }
    if (streamer_ptr) {
        streamer_ptr->end();
    }
    draft_generations.clear();

    for (size_t generation_idx = 0; generation_idx < main_generations.size(); ++generation_idx) {
        const auto& generation = main_generations[generation_idx];
        EncodedGenerationResult result;
        result.m_request_id = 1;
        std::vector<GenerationOutput> generation_outputs = generation->read_all();
        std::sort(generation_outputs.begin(), generation_outputs.end(), [=] (GenerationOutput& r1, GenerationOutput& r2) {
            return r1.score > r2.score;
        });

        auto num_outputs = std::min(sampling_params[generation_idx].num_return_sequences, generation_outputs.size());
        for (size_t generation_output_idx = 0; generation_output_idx < num_outputs; ++generation_output_idx) {
            const auto& generation_output = generation_outputs[generation_output_idx];
            result.m_generation_ids.push_back(std::move(generation_output.generated_ids));
            result.m_scores.push_back(generation_output.score);
        }
        result.m_status = generation->get_status();
        results.push_back(std::move(result));
    }

    OPENVINO_ASSERT(results.size() == input_ids.size());
    return results;
}

// std::vector<GenerationResult>
// ContinuousBatchingPipeline::SpeculativeDecodingImpl::generate(const std::vector<std::string>& prompts,
//                                                               std::vector<ov::genai::GenerationConfig> sampling_params,
//                                                               const StreamerVariant& streamer) {
//     return m_main_pipeline->generate(prompts, sampling_params, streamer);
// }

}
