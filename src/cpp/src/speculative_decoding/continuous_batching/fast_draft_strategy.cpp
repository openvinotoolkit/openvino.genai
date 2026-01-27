// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <thread>

#include "openvino/genai/text_streamer.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "fast_draft_strategy.hpp"
#include "continuous_batching/paged_attention_transformations.hpp"
#include "utils.hpp"


namespace ov::genai {
template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

bool are_tokenizers_equal(Tokenizer& lhs, Tokenizer& rhs) {
    std::string test_string = "Could you please tell me something about OpenVINO.GenAI?";
    ov::Tensor encoded_string_lhs = lhs.encode(test_string).input_ids,
               encoded_string_rhs = rhs.encode(test_string).input_ids;
    
    ov::Shape shape_lhs = encoded_string_lhs.get_shape(),
              shape_rhs = encoded_string_rhs.get_shape();

    return shape_lhs == shape_rhs && lhs.get_eos_token_id() == rhs.get_eos_token_id() &&
           lhs.get_bos_token_id() == rhs.get_bos_token_id() && lhs.get_pad_token_id() == rhs.get_pad_token_id();
}

std::pair<ov::genai::SchedulerConfig, ov::genai::SchedulerConfig>
ContinuousBatchingPipeline::SpeculativeDecodingImpl::init_speculative_models(const ov::genai::ModelDesc& main_model_desc, const ov::genai::ModelDesc& draft_model_desc) {
    auto main_model = main_model_desc.model;
    auto draft_model = draft_model_desc.model;
    OPENVINO_ASSERT(main_model != nullptr, "Main model cannot be null");
    OPENVINO_ASSERT(draft_model != nullptr, "Draft model cannot be null");

    auto main_scheduler_config = main_model_desc.scheduler_config;
    bool allow_score_aggregation = true;
    bool allow_xattention = false;

    ov::pass::SDPAToPagedAttention(main_model_desc.scheduler_config.use_cache_eviction,
                                   main_model_desc.scheduler_config.use_cache_eviction,
                                   allow_score_aggregation,
                                   allow_xattention).run_on_model(main_model);
    ov::pass::SDPAToPagedAttention(main_model_desc.scheduler_config.use_cache_eviction,
                                   main_model_desc.scheduler_config.use_cache_eviction,
                                   allow_score_aggregation,
                                   allow_xattention).run_on_model(draft_model);

    utils::apply_gather_before_matmul_transformation(main_model);
    utils::apply_gather_before_matmul_transformation(draft_model);

    bool is_draft_scheduler_undefined = draft_model_desc.scheduler_config == SchedulerConfig();

    ov::genai::SchedulerConfig main_scheduler_config_updated = main_scheduler_config,
                               draft_scheduler_config = is_draft_scheduler_undefined ? main_scheduler_config : draft_model_desc.scheduler_config;

    if (is_draft_scheduler_undefined) {
        // split KV cache to 2 caches for main and draft models
        auto compute_total_hidden_size = [] (const std::shared_ptr<ov::Model>& model) -> size_t {
            size_t total_hidden_size = 0;
            for (const auto& param_ptr : model->get_parameters()) {
                const auto& name = param_ptr->get_friendly_name();
                if (name.find("key_cache.") == 0) {
                    auto pa_op = param_ptr->get_output_target_inputs(0).begin()->get_node();
                    const auto& rt_info = pa_op->get_rt_info();
                    total_hidden_size += rt_info.at("num_k_heads").as<int>() * rt_info.at("k_head_size").as<int>() +
                                         rt_info.at("num_v_heads").as<int>() * rt_info.at("v_head_size").as<int>();
                }
            }
            return total_hidden_size;
        };
        float main_model_hidden_size = compute_total_hidden_size(main_model),
              draft_model_hidden_size = compute_total_hidden_size(draft_model);
        auto k = draft_model_hidden_size / (main_model_hidden_size + draft_model_hidden_size);

        // TODO: work with KV blocks as it will be more precise instead of GBs
        size_t main_cache_size = std::ceil(main_scheduler_config.cache_size * (1.f - k)),
               draft_cache_size = main_scheduler_config.cache_size - main_cache_size;
        if (draft_cache_size == 0 && main_cache_size > 0) {
            main_cache_size -= (main_cache_size > 1 ? 1 : 0);
            draft_cache_size = 1;
        }

        main_scheduler_config_updated.cache_size = main_cache_size;
        draft_scheduler_config.cache_size = draft_cache_size;
    } else {
        draft_scheduler_config.dynamic_split_fuse = main_scheduler_config_updated.dynamic_split_fuse;
        draft_scheduler_config.max_num_batched_tokens = main_scheduler_config_updated.max_num_batched_tokens;
    }

    return std::make_pair(main_scheduler_config_updated, draft_scheduler_config);
}

ContinuousBatchingPipeline::SpeculativeDecodingImpl::SpeculativeDecodingImpl(const ov::genai::ModelDesc& main_model_desc,
                                                                             const ov::genai::ModelDesc& draft_model_desc) {
    auto scheduler_configs = init_speculative_models(main_model_desc, draft_model_desc);

    auto main_device = main_model_desc.device;
    std::string draft_device = draft_model_desc.device.empty() ? main_model_desc.device : draft_model_desc.device;
    // main and draft model can have different tokenizers
    // to do: support retokenization: 154103
    Tokenizer main_model_tokenizer = main_model_desc.tokenizer;
    Tokenizer draft_model_tokenizer = draft_model_desc.tokenizer;

    // todo: remove this condition after support of CVS-154103
    OPENVINO_ASSERT(are_tokenizers_equal(main_model_tokenizer, draft_model_tokenizer), "Tokenizers for draft and main models are different!");
    m_tokenizer = main_model_tokenizer;
    ov::AnyMap draft_properties = draft_model_desc.properties.empty() ? main_model_desc.properties : draft_model_desc.properties;
    // to create `main_pipeline` with enabled validation_mode and `draft_pipeline` with disabled validation mode
    m_main_pipeline = std::make_shared<ContinuousBatchingForSpeculativeDecodingImpl>(
        main_model_desc.model, main_model_tokenizer, main_model_desc.generation_config,
        scheduler_configs.first, main_device, main_model_desc.properties, true);
    m_draft_pipeline = std::make_shared<ContinuousBatchingForSpeculativeDecodingImpl>(
        draft_model_desc.model, draft_model_tokenizer, draft_model_desc.generation_config,
        scheduler_configs.second, draft_device, draft_properties, false);

    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    m_draft_pipeline->raw_perf_metrics.m_inference_durations =  {{ MicroSeconds(0.0f) }};
}

GenerationHandle
ContinuousBatchingPipeline::SpeculativeDecodingImpl::add_request(uint64_t request_id,
                                                                 const ov::Tensor& input_ids,
                                                                 const ov::genai::GenerationConfig& sampling_params,
                                                                 std::optional<ov::Tensor> token_type_ids) {
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    auto draft_sampling_params = sampling_params;
    draft_sampling_params.ignore_eos = true;
    draft_sampling_params.stop_strings = {};
    m_draft_generations.insert({request_id, m_draft_pipeline->add_request(request_id, input_ids, draft_sampling_params, token_type_ids)});
    return m_main_pipeline->add_request(request_id, input_ids, sampling_params, token_type_ids);
}

GenerationHandle
ContinuousBatchingPipeline::SpeculativeDecodingImpl::add_request(uint64_t request_id,
                                                                 const std::string& prompt,
                                                                 const ov::genai::GenerationConfig& sampling_params) {
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    auto draft_sampling_params = sampling_params;
    draft_sampling_params.ignore_eos = true;
    draft_sampling_params.stop_strings = {};
    m_draft_generations.insert({request_id, m_draft_pipeline->add_request(request_id, prompt, draft_sampling_params)});
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
    // this blocks adding new requests during step as it may break coherence between main and draft models
    std::lock_guard<std::mutex> lock{m_draft_generations_mutex};

    auto& raw_perf_counters = m_perf_metrics.raw_metrics;
    auto& main_raw_perf_counters = m_perf_metrics.main_model_metrics.raw_metrics;

    const auto step_start = std::chrono::steady_clock::now();

    m_draft_pipeline->pull_awaiting_requests(true);
    m_main_pipeline->pull_awaiting_requests();

    // generate candidates by draft model
    const auto draft_start = std::chrono::steady_clock::now();
    m_draft_pipeline->multistep();
    const auto draft_end = std::chrono::steady_clock::now();
    m_sd_metrics.draft_duration += PerfMetrics::get_microsec(draft_end - draft_start) / 1e6;
    m_pipeline_metrics = m_main_pipeline->get_metrics();

    // to generate num_matches statistic
    std::map<int64_t, UpdateRequestResult> update_sequence_info;
    // put candidates to model KV cache
    auto draft_generated_requests = m_draft_pipeline->get_generated_requests();
    for (const auto& candidate : m_draft_pipeline->get_generated_requests()) {
        auto update_result = m_main_pipeline->update_request(candidate.first, candidate.second, false);
        update_sequence_info.insert({{candidate.first, update_result}});
    }

    const auto main_start = std::chrono::steady_clock::now();
    m_main_pipeline->step();
    const auto main_end = std::chrono::steady_clock::now();
    const auto main_duration = PerfMetrics::get_microsec(main_end - main_start);
    m_sd_metrics.main_duration += main_duration / 1e6;
    m_pipeline_metrics = m_main_pipeline->get_metrics();

    auto main_generated_requests = m_main_pipeline->get_generated_requests();
    for (const auto& checked_sequence : main_generated_requests) {
        auto update_result = m_draft_pipeline->update_request(checked_sequence.first, checked_sequence.second, true);
        update_sequence_info[checked_sequence.first].removed_tokens_cnt = update_result.removed_tokens_cnt;
    }

    // finish draft request if the generation was completed
    for (const auto& draft_request : draft_generated_requests) {
        auto request_id = draft_request.first;
        if (!main_generated_requests.count(request_id)) {
            m_draft_pipeline->finish_request(request_id);
            // remove draft_generation_handle from queue
            m_draft_generations.erase(request_id);
        }
        auto updated_seq_info = update_sequence_info[request_id];
        m_sd_metrics.update_draft_generated_len(request_id, updated_seq_info.inserted_tokens_cnt);

        // several prompt phase
        if (updated_seq_info.inserted_tokens_cnt == 0 || main_generated_requests.empty()) {
            continue;
        }
        float acceptance_rate = 1 - static_cast<float>(updated_seq_info.removed_tokens_cnt) / updated_seq_info.inserted_tokens_cnt;
        m_sd_metrics.update_acceptance_rate(request_id, acceptance_rate * 100);
        m_sd_metrics.update_draft_accepted_tokens(request_id, (updated_seq_info.inserted_tokens_cnt - updated_seq_info.removed_tokens_cnt));
    }

    const auto step_end = std::chrono::steady_clock::now();
    const auto step_microsec_duration = PerfMetrics::get_microsec(step_end - step_start);

    // update perf metrics
    const auto num_generated_tokens = m_main_pipeline->get_processed_tokens_per_iteration();
    if (num_generated_tokens > 0) {
        raw_perf_counters.m_token_infer_durations.emplace_back(step_microsec_duration);
        raw_perf_counters.m_inference_durations[0] += MicroSeconds(step_microsec_duration);
        raw_perf_counters.m_new_token_times.emplace_back(main_end);
        raw_perf_counters.m_batch_sizes.emplace_back(num_generated_tokens);

        auto m_main_pipeline_metrics = m_main_pipeline->get_metrics();
        main_raw_perf_counters.m_durations.push_back(MicroSeconds(main_duration));
        main_raw_perf_counters.m_inference_durations[0] += MicroSeconds(m_main_pipeline_metrics.inference_duration);
        main_raw_perf_counters.m_batch_sizes.push_back(num_generated_tokens); // or should be processed + generated
        m_sd_metrics.update_generated_len(num_generated_tokens);
    }

    if (main_generated_requests.empty() && utils::env_setup_for_print_debug_info()) {
        m_sd_metrics.print(true);
        m_sd_metrics.clean_up();
    }
}



std::vector<EncodedGenerationResult>
ContinuousBatchingPipeline::SpeculativeDecodingImpl::generate(const std::vector<ov::Tensor>& input_ids,
                                                              const std::vector<GenerationConfig>& sampling_params,
                                                              const StreamerVariant& streamer,
                                                              const std::optional<std::vector<ov::Tensor>>& token_type_ids,
                                                              const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids) {
    GenerateStrategy strategy;
    strategy.prepare_request = [this](size_t,
                                  const ov::Tensor& in_ids,
                                  GenerationConfig& main_cfg,
                                  GenerationConfig& draft_cfg,
                                  ov::Tensor& main_in,
                                  ov::Tensor& draft_in) {
        if (main_cfg.assistant_confidence_threshold == 0.f) {
            if (main_cfg.num_assistant_tokens == 0) {
                main_cfg.num_assistant_tokens = m_main_pipeline->default_num_assistant_tokens;
            }
        }
        draft_cfg.ignore_eos = true;
        draft_cfg.stop_strings = {};
        main_in = in_ids;
        draft_in = in_ids;
    };
    strategy.check_streaming = [](const std::shared_ptr<ThreadedStreamerWrapper>& streamer_ptr,
                                  const std::vector<ov::Tensor>& input_ids,
                                  const std::vector<GenerationConfig>& sampling_params) {
        OPENVINO_ASSERT(!streamer_ptr->has_callback() ||
                        (input_ids.size() == 1 &&
                         (sampling_params[0].is_greedy_decoding() || sampling_params[0].is_multinomial())),
                        "Streaming only supports batch size=1 with greedy/multinomial");
    };
    strategy.start_timer = [](){ return std::chrono::steady_clock::now(); };
    strategy.stop_timer  = [](const TimePoint& start){
        return PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start);
    };

    return generate_common(this, input_ids, sampling_params, streamer, token_type_ids, strategy);
}

SpeculativeDecodingMetrics
ContinuousBatchingPipeline::SpeculativeDecodingImpl::get_speculative_decoding_metrics() {
    std::lock_guard<std::mutex> lock{m_draft_generations_mutex};
    return m_sd_metrics;
};

void ContinuousBatchingPipeline::SpeculativeDecodingImpl::drop_requests() {
    m_draft_pipeline->finish_request();
    m_main_pipeline->finish_request();
}


bool ContinuousBatchingPipeline::SpeculativeDecodingImpl::is_requests_empty() {
    return m_main_pipeline->is_requests_empty() && m_draft_pipeline->is_requests_empty();
}

std::vector<SequenceGroup::Ptr> ContinuousBatchingPipeline::SpeculativeDecodingImpl::get_awaiting_requests() {
    auto main_awaiting_requests = m_main_pipeline->get_awaiting_requests();
    auto draft_awaiting_requests = m_draft_pipeline->get_awaiting_requests();
    OPENVINO_ASSERT(main_awaiting_requests.size() == draft_awaiting_requests.size());
    return main_awaiting_requests;
}

}
