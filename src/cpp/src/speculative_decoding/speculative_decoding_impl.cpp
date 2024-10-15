// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "text_callback_streamer.hpp"
#include "speculative_decoding_impl.hpp"
#include "utils/ov_utils.hpp"


namespace ov::genai {
template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

ContinuousBatchingPipeline::SpeculativeDecodingImpl::SpeculativeDecodingImpl(
    const std::string& main_models_path,
    const SchedulerConfig& main_scheduler_config,
    const std::string& main_device,
    const ov::AnyMap& main_plugin_config,
    const ov::genai::ModelDesc draft_model_desc,
    const ov::AnyMap& tokenizer_plugin_config) {
    ov::Core core;
    std::string openvino_model_name = "/openvino_model.xml",
                draft_model_path = draft_model_desc.model_path;

    std::shared_ptr<ov::Model> main_model = core.read_model(main_models_path + openvino_model_name),
                               draft_model = core.read_model(draft_model_path + openvino_model_name);

    utils::apply_paged_attention_transformations(main_model, main_scheduler_config.use_cache_eviction);
    utils::apply_paged_attention_transformations(draft_model, main_scheduler_config.use_cache_eviction);

    std::string draft_device = draft_model_desc.device;
    bool is_draft_device_undefined = false;
    if (draft_device.empty() || draft_device == main_device) {
        draft_device = main_device;
        is_draft_device_undefined = true;
    }

    ov::genai::SchedulerConfig main_scheduler_config_updated = main_scheduler_config,
                               draft_scheduler_config = is_draft_device_undefined ? main_scheduler_config : draft_model_desc.scheduler_config;
    if (is_draft_device_undefined) {
        // split KV cache to 2 caches for main and draft models
        size_t main_model_cache_size = utils::get_kv_cache_size(main_model),
            draft_model_cache_size = utils::get_kv_cache_size(draft_model);
        auto k = static_cast<float>(draft_model_cache_size) / (main_model_cache_size + draft_model_cache_size);

        size_t main_cache_size = main_scheduler_config.cache_size * (1 - k),
               draft_cache_size = main_scheduler_config.cache_size * k;
        if (draft_cache_size == 0) {
            main_cache_size -= main_cache_size > 1 ? 1 : 0;
            draft_cache_size = 1;
        }

        main_scheduler_config_updated.cache_size = main_cache_size;
        draft_scheduler_config.cache_size = draft_cache_size;
    }

    ov::AnyMap draft_plugin_config = is_draft_device_undefined ? main_plugin_config : draft_model_desc.plugin_config;

    DeviceConfig main_device_config(core, main_scheduler_config, main_device, main_plugin_config),
                 draft_device_config(core, draft_scheduler_config, draft_device, draft_plugin_config);

    utils::set_kv_cache_type_and_shape(main_model, main_device_config);
    utils::set_kv_cache_type_and_shape(draft_model, draft_device_config);

    // main and draft model can have different tokenizers
    // to do: support retokenization: 154103
    Tokenizer main_model_tokenizer(main_models_path, tokenizer_plugin_config),
              draft_model_tokenizer(draft_model_path, tokenizer_plugin_config);
    
    m_tokenizer = main_model_tokenizer;

    // to create `main_pipeline` with enabled validation_mode and `draft_pipeline` with disabled validation mode
    m_main_pipeline = std::make_shared<ContinuousBatchingForSpeculativeDecodingImpl>(core, main_model, main_model_tokenizer, main_device_config, main_scheduler_config, main_device, main_plugin_config, true);
    m_draft_pipeline = std::make_shared<ContinuousBatchingForSpeculativeDecodingImpl>(core, draft_model, draft_model_tokenizer, draft_device_config, draft_scheduler_config, draft_device, draft_plugin_config, false);
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

void print_generated_request(const ov::genai::GeneratedRequests& requests) {
    for (const auto& request : requests) {
        for (const auto& sequence : request.second) {
            std::cout << "request_id: " << request.first << " | sequence_id: " << sequence.first << " | ";
            for (const auto& token_id : sequence.second.token_ids) {
                std::cout << token_id << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void ContinuousBatchingPipeline::SpeculativeDecodingImpl::step() {
    // to make first inference to process prompt once and generate first token by main model
    if (m_main_pipeline->is_pipeline_not_started()) {
        m_draft_pipeline->step();
        m_main_pipeline->step();
        for (const auto& prompt_generated_tokens : m_main_pipeline->get_generated_requests()) {
            m_draft_pipeline->update_request(prompt_generated_tokens.first, prompt_generated_tokens.second, true);
        }
    }

    // generate candidates by draft model
    static ManualTimer draft_timer("speculative_decoding: draft_model: multistep()");
    draft_timer.start();
    m_draft_pipeline->multistep();
    draft_timer.end();
    m_sd_metrics.draft_duration += draft_timer.get_duration_ms();

    // to generate num_matches statistic
    std::map<int64_t, UpdateRequestResult> update_sequence_info;
    // put candidates to model KV cache
    auto draft_generated_requests = m_draft_pipeline->get_generated_requests();
    for (const auto& candidate : m_draft_pipeline->get_generated_requests()) {
        auto update_result = m_main_pipeline->update_request(candidate.first, candidate.second, false);
        update_sequence_info.insert({{candidate.first, update_result}});
    }

    static ManualTimer main_timer("speculative_decoding: main_model: step()");
    main_timer.start();
    m_main_pipeline->step();
    main_timer.end();
    m_sd_metrics.main_duration += main_timer.get_duration_ms();

    auto main_generated_requests = m_main_pipeline->get_generated_requests();
    for (const auto& checked_sequence : main_generated_requests) {
        auto update_result = m_draft_pipeline->update_request(checked_sequence.first, checked_sequence.second, true);
        update_sequence_info[checked_sequence.first].removed_tokens_cnt = update_result.removed_tokens_cnt;
    }

    // finish draft request if the generation was complited
    for (const auto& draft_request : draft_generated_requests) {
        auto request_id = draft_request.first;
        if (!main_generated_requests.count(request_id)) {
            m_draft_pipeline->finish_request(request_id);
        }
        auto updated_seq_info = update_sequence_info[request_id];
        float acceptance_rate = 1 - static_cast<float>(updated_seq_info.removed_tokens_cnt) / updated_seq_info.inserted_tokens_cnt; 
        m_sd_metrics.update_acceptance_rate(request_id, acceptance_rate);
        m_sd_metrics.update_draft_accepted_tokens(request_id, (updated_seq_info.inserted_tokens_cnt - updated_seq_info.removed_tokens_cnt));
    }
}

std::vector<EncodedGenerationResult>
ContinuousBatchingPipeline::SpeculativeDecodingImpl::generate(const std::vector<ov::Tensor>& input_ids,
                                                              const std::vector<GenerationConfig>& sampling_params,
                                                              const StreamerVariant& streamer) {                                                  
    static ManualTimer timer("speculative_decoding: generate()");
    timer.start();
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
        // set the parameters do not stop draft generation without stopping of the same request for main pipeline
        draft_sampling_params.max_new_tokens = SIZE_MAX - 1;
        draft_sampling_params.min_new_tokens = SIZE_MAX - 1;
        draft_sampling_params.ignore_eos = true;
        draft_generations.push_back(m_draft_pipeline->add_request(request_id, input_ids[request_id], draft_sampling_params));
        // decrease generation len to generate last token by main model
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
            m_sd_metrics.set_generated_len(generation_idx, generation_outputs[generation_output_idx].generated_ids.size());
            result.m_generation_ids.push_back(std::move(generation_output.generated_ids));
            result.m_scores.push_back(generation_output.score);
        }
        result.m_status = generation->get_status();
        results.push_back(std::move(result));
    }

    OPENVINO_ASSERT(results.size() == input_ids.size());
    timer.end();
    m_sd_metrics.total_duration = timer.get_duration_ms();
    return results;
}

SpeculativeDecodingMetrics
ContinuousBatchingPipeline::SpeculativeDecodingImpl::get_speculative_decoding_metrics() {
    return m_sd_metrics;
};
}
