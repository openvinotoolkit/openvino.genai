// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "text_callback_streamer.hpp"
#include "continuous_batching_impl.hpp"
#include "utils.hpp"
#include "utils/paged_attention_transformations.hpp"
#include "lora_helper.hpp"

namespace ov::genai {
template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

ContinuousBatchingPipeline::ContinuousBatchingImpl::ContinuousBatchingImpl(
    const std::shared_ptr<ov::Model>& model,
    const Tokenizer& tokenizer,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& properties,
    const ov::genai::GenerationConfig& generation_config,
    bool is_validation_mode_enabled) {
    m_tokenizer = tokenizer;
    m_generation_config = generation_config;
    m_is_validation_mode_enabled = is_validation_mode_enabled;

    ov::Core core = utils::singleton_core();
    DeviceConfig device_config(core, scheduler_config, device, properties);

    bool is_need_per_layer_cache_control = scheduler_config.use_cache_eviction;
    utils::apply_paged_attention_transformations(model, device_config, is_need_per_layer_cache_control);

    initialize_pipeline(model, scheduler_config, properties, device_config, core);
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::_pull_awaiting_requests() {
    std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
    m_requests.insert(m_requests.end(), m_awaiting_requests.begin(), m_awaiting_requests.end());
    m_awaiting_requests.clear();
    m_pipeline_metrics.requests = m_requests.size();
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::initialize_pipeline(
    std::shared_ptr<ov::Model> model,
    const SchedulerConfig& scheduler_config,
    const ov::AnyMap& properties,
    const DeviceConfig& device_config,
    ov::Core& core) {
    ov::CompiledModel compiled_model;

    // apply LoRA
    if (auto filtered_properties = extract_adapters_from_properties(properties, &m_generation_config.adapters)) {
        m_generation_config.adapters->set_tensor_name_prefix("base_model.model.model.");
        m_adapter_controller = AdapterController(model, *m_generation_config.adapters, device_config.get_device());   // TODO: Make the prefix name configurable
        compiled_model = core.compile_model(model, device_config.get_device(), *filtered_properties);
    } else {
        compiled_model = core.compile_model(model, device_config.get_device(), properties);
    }

    ov::genai::utils::print_compiled_model_properties(compiled_model, "LLM with Paged Attention");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // setup KV caches
    m_cache_manager = std::make_shared<CacheManager>(device_config, infer_request, core);

    SchedulerConfig updated_config = scheduler_config;
    // update KV blocks number in scheduler config
    if (scheduler_config.num_kv_blocks != device_config.get_num_kv_blocks()) {
        updated_config.num_kv_blocks = device_config.get_num_kv_blocks();
    }

    bool can_use_partial_preemption = true;
    if (device_config.get_device().find("GPU") != std::string::npos && !updated_config.dynamic_split_fuse) {
        // in case of executing a `vLLM-like` pipeline, it's better not to use partial eviction on the GPU,
        // as it may lead to performance slowdown
        can_use_partial_preemption = false;
    }
    m_scheduler = std::make_shared<Scheduler>(device_config.get_block_size(), m_cache_manager, updated_config, device_config.get_num_layers(), can_use_partial_preemption);

    // model runner
    bool is_use_cache_eviction = m_scheduler->get_config().use_cache_eviction;
    m_model_runner = std::make_shared<ModelRunner>(infer_request, m_scheduler->get_block_size(), device_config.get_num_layers(), is_use_cache_eviction);

    // sampler
    m_sampler = std::make_shared<Sampler>(m_tokenizer);
    m_sampler->set_seed(m_generation_config.rng_seed);

    // If eos_token_id was not provided, take value
    if (m_generation_config.eos_token_id == -1)
        m_generation_config.set_eos_token_id(m_tokenizer.get_eos_token_id());
};


GenerationHandle
ContinuousBatchingPipeline::ContinuousBatchingImpl::add_request(uint64_t request_id,
                                                               const ov::Tensor& input_ids,
                                                               ov::genai::GenerationConfig sampling_params) {
    // If eos_token_id was not provided, take value from default m_generation_config
    if (sampling_params.eos_token_id == -1)
        sampling_params.set_eos_token_id(m_generation_config.eos_token_id);
    sampling_params.validate();

    SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(request_id, input_ids,
                                                                        sampling_params,
                                                                        m_scheduler->get_block_size());

    if (m_scheduler->get_config().enable_prefix_caching) {
        m_scheduler->restore_cached_blocks(sequence_group);
    }

    {
        std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
        m_awaiting_requests.push_back(sequence_group);
    }

    return std::make_shared<GenerationHandleImpl>(sequence_group->get_generation_stream(), sampling_params);
};

GenerationHandle
ContinuousBatchingPipeline::ContinuousBatchingImpl::add_request(uint64_t request_id,
                                                                const std::string& prompt,
                                                                ov::genai::GenerationConfig sampling_params) {
    static ManualTimer timer("tokenize");
    timer.start();
    ov::Tensor input_ids = m_tokenizer.encode(prompt).input_ids;
    timer.end();

    return add_request(request_id, input_ids, sampling_params);
}

bool ContinuousBatchingPipeline::ContinuousBatchingImpl::has_non_finished_requests() {
    std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
    return !m_awaiting_requests.empty() || !m_requests.empty();
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::step() {
    static ManualTimer step_timer("step()");
    step_timer.start();

    _pull_awaiting_requests();

    Scheduler::Output scheduler_output;
    {
        static ManualTimer scheduling_timer("scheduling");
        scheduling_timer.start();
        scheduler_output = m_scheduler->schedule(m_requests);
        scheduling_timer.end();

        m_pipeline_metrics.scheduled_requests = scheduler_output.m_scheduled_sequence_groups_ids.size();
        m_pipeline_metrics.cache_usage = scheduler_output.m_cache_usage;
        m_pipeline_metrics.max_cache_usage = std::max(m_pipeline_metrics.max_cache_usage, scheduler_output.m_cache_usage);
        _register_step_cache_usage(scheduler_output.m_cache_usage);
        m_pipeline_metrics.avg_cache_usage = _get_current_running_average_cache_usage();

        static ManualTimer copy_blocks_timer("scheduling");
        copy_blocks_timer.start();
        m_cache_manager->copy_blocks(scheduler_output.m_block_copy_map);
        copy_blocks_timer.end();
    }

    // if no tokens were scheduled, we are out of memory => free all requests and return
    if (scheduler_output.m_total_num_scheduled_tokens == 0) {
        for (size_t i = 0; i < m_requests.size(); ++i) {
            SequenceGroup::Ptr sequence_group = m_requests[i];
            if (!sequence_group->is_waiting()) {
                sequence_group->set_out_of_memory();
                sequence_group->notify_handle();
            }
        }
        _free_non_running_requests();
        return;
    }

    ov::Tensor logits;
    {
        static ManualTimer timer("forward");
        timer.start();
        logits = m_model_runner->forward(m_requests, scheduler_output);
        timer.end();
    }

#ifdef DEBUG_CACHE_STATE_DUMP
    CacheStateDumper dumper(CacheStateDumper::get_run_id_for_generation_step(step_count, "before_eviction"));
    dumper.dump_cache_state(*m_scheduler, m_requests, step_count);
#endif

    // evict unimportant blocks from KV cache, if requested
    const auto& sched_config = m_scheduler->get_config();
    if (sched_config.use_cache_eviction) {
        _maybe_evict_cache_blocks(sched_config);
    }

#ifdef DEBUG_CACHE_STATE_DUMP
    CacheStateDumper dumper_after(CacheStateDumper::get_run_id_for_generation_step(step_count, "eviction"));
    dumper_after.dump_cache_state(*m_scheduler, m_requests, step_count);
    step_count++;
#endif

    // process generation_config.echo parameter
    _fill_prompt_log_probs(m_requests, logits);

    SamplerOutput sampler_output;
    {
        static ManualTimer timer("sample");
        timer.start();
        sampler_output = m_sampler->sample(m_requests, logits, m_is_validation_mode_enabled);
        timer.end();
    }

    // process sampler_output (e.g. fork or drop sequences from BlockScheduler)
    {
        static ManualTimer free_fork_timer("fork / free sequence");
        free_fork_timer.start();

        for (const auto& pair : sampler_output.m_forked_sequences) {
            uint64_t parent_id = pair.first;
            const std::list<uint64_t>& child_ids = pair.second;
            for (auto& child_id : child_ids)
                m_scheduler->fork_sequence(parent_id, child_id);
        }

        for (auto seq_id : sampler_output.m_dropped_sequences)
            m_scheduler->free_sequence(seq_id);

        free_fork_timer.end();
    }

    // notify requests dropped by handle
    {
        static ManualTimer report_tokens_timer("notify requests dropped by handle");
        report_tokens_timer.start();
        _notify_requests_dropped_by_handle();
        report_tokens_timer.end();
    }

    // free non running requests for current step

    {
        static ManualTimer clean_up_requests_timer("free non running requests");
        clean_up_requests_timer.start();
        _free_non_running_requests();
        clean_up_requests_timer.end();
    }

    step_timer.end();
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::set_adapters(const std::optional<AdapterConfig>& adapters) {
    if (m_adapter_controller) {
        m_adapter_controller->apply(m_model_runner->get_infer_request(), adapters);
    }
}

std::vector<EncodedGenerationResult>
ContinuousBatchingPipeline::ContinuousBatchingImpl::generate(const std::vector<ov::Tensor>& input_ids,
                                                             const std::vector<GenerationConfig>& sampling_params,
                                                             const StreamerVariant& streamer) {
    OPENVINO_ASSERT(!has_non_finished_requests(), "Generate cannot be called while ContinuousBatchingPipeline is already in running state. Use ContinuousBatchingPipeline::add_request");
    OPENVINO_ASSERT(input_ids.size() == sampling_params.size());

    // checks that all requests has the same LoRA adapters property value
    for (size_t i = 1; i < sampling_params.size(); ++i) {
        OPENVINO_ASSERT(sampling_params[i - 1].adapters == sampling_params[i].adapters,
            "LoRA adapters value must be the same for all requests");
    }
    set_adapters(sampling_params[0].adapters);

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

    auto drop_requests = [&] () {
        for (const std::shared_ptr<ov::genai::SequenceGroup> request : m_requests) {
            for (const auto& sequence: request->get_sequences()) {
                if (m_scheduler->has_block_table(sequence->get_id())) {
                    m_scheduler->free_sequence(sequence->get_id());
                }
            }
            m_sampler->clear_request_info(request->get_request_id());
        }
        m_requests.clear();
    };

    OPENVINO_ASSERT(streamer_ptr == nullptr || input_ids.size() == 1 && sampling_params[0].num_return_sequences == 1 &&
        (sampling_params[0].is_greedy_decoding() || sampling_params[0].is_multinomial()),
        "Currently streaming is possible only with batch size=1 and only for greedy or multinomial decoding");

    std::vector<GenerationHandle> generations;
    for (size_t request_id = 0; request_id < input_ids.size(); ++request_id) {
        OPENVINO_ASSERT(1 == input_ids[request_id].get_shape().at(0), "Use multiple tensors to pass a batch.");
        generations.push_back(add_request(request_id, input_ids[request_id], sampling_params[request_id]));
    }
    auto all_requests = m_awaiting_requests; // we need to store all requests to get results from them once generation has finished

    bool continue_generation = true;
    while (has_non_finished_requests() && continue_generation) {
        try {
            step();
        } catch (...) {
            drop_requests(); // remove all requests from pipeline state in case of exception
            throw;
        }

        auto & generation = generations.at(0);
        if (streamer_ptr && generation->can_read()) {
            std::unordered_map<uint64_t, GenerationOutput> token = generation->back();
            for (const auto& gen_token : token.begin()->second.generated_ids) {
                continue_generation = !streamer_ptr->put(gen_token);
                if (!continue_generation) {
                    generation->drop();
                    break;
                }
            }
        }
    }

    if (streamer_ptr) { // push streamer's cache
        streamer_ptr->end();
    }

    if (!continue_generation) {
        drop_requests();
    } else {
        OPENVINO_ASSERT(m_requests.empty(), "Internal error: current request is supposed to be dropped within step() function as completed");
    }

    std::vector<EncodedGenerationResult> results;
    results.reserve(all_requests.size());

    for (size_t request_id = 0; request_id < all_requests.size(); ++request_id) {
        const auto& request = all_requests[request_id];
        auto sampling_params = request->get_sampling_parameters();
        const auto& sequences = request->get_finished_sequences();
        size_t num_outputs = std::min(sampling_params.num_return_sequences, sequences.size());

        EncodedGenerationResult result;
        result.m_request_id = request_id;
        result.m_generation_ids.resize(num_outputs);
        result.m_scores.resize(num_outputs);

        for (size_t i = 0; i < num_outputs; ++i) {
            const auto & sequence = sequences[i];
            const float score = sampling_params.is_beam_search() ? sequence->get_beam_search_score(sampling_params) : sequence->get_cumulative_log_prob();
            const auto & generated_ids = sequence->get_generated_ids();

            if (sampling_params.echo)
                result.m_generation_ids[i] = request->get_prompt_ids();
            std::copy(generated_ids.begin(), generated_ids.end(), std::back_inserter(result.m_generation_ids[i]));
            result.m_scores[i] = score;
        }

        result.m_status = generations[request_id]->get_status();
        results.push_back(std::move(result));
    }

    OPENVINO_ASSERT(results.size() == input_ids.size());
    return results;
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::get_infer_duration(float& duration, int& number) {
    m_model_runner->get_infer_duration(duration, number);
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::reset_infer_duration() {
    m_model_runner->reset_infer_duration();
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::_free_non_running_requests() {
    std::vector<SequenceGroup::Ptr>::iterator requests_iterator = m_requests.begin();
    while (requests_iterator != m_requests.end()) {
        const auto& request = *requests_iterator;
        if(request->has_finished() || request->out_of_memory() || request->handle_dropped()) {
            for (const auto& sequence: request->get_sequences()) {
                if (m_scheduler->has_block_table(sequence->get_id())) {
                    m_scheduler->free_sequence(sequence->get_id());
                }
            }
            m_sampler->clear_request_info(request->get_request_id());
            requests_iterator = m_requests.erase(requests_iterator);
        } else {
            requests_iterator++;
        }
    }
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::_notify_requests_dropped_by_handle() {
    // Notify the last time by pushing empty output
    // This causes read() to unblock by adding anything to the queue
    for (SequenceGroup::Ptr& request : m_requests) {
        if (request->handle_dropped())
            request->push_empty_outputs();
    }
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::_register_step_cache_usage(float step_cache_usage) {
    if (m_previous_step_cache_usages.size() >= AVG_CACHE_USAGE_WINDOW_SIZE_IN_STEPS) {
        m_previous_step_cache_usages.pop_front();
    }
    m_previous_step_cache_usages.push_back(step_cache_usage);
}

float ContinuousBatchingPipeline::ContinuousBatchingImpl::_get_current_running_average_cache_usage() const {
    return std::accumulate(m_previous_step_cache_usages.begin(), m_previous_step_cache_usages.end(), 0.0) / m_previous_step_cache_usages.size();
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::_maybe_evict_cache_blocks(const SchedulerConfig& sched_config) {
    std::unordered_map<SequenceGroup::Ptr, size_t> seq_group_to_num_blocks_evicted_map;
    auto sequence_attention_scores = m_model_runner->get_last_attention_scores();
    for (auto& seq_id_and_attention_scores : sequence_attention_scores) {
        auto seq_id = seq_id_and_attention_scores.first;
        const auto& attention_scores_for_all_decoder_layers = seq_id_and_attention_scores.second;
        if (m_seq_group_id_to_cache_eviction_algo_map.find(seq_id) == m_seq_group_id_to_cache_eviction_algo_map.end()) {
            auto num_decoder_layers = attention_scores_for_all_decoder_layers.size();

            m_seq_group_id_to_cache_eviction_algo_map[seq_id] = CacheEvictionAlgorithm(sched_config.cache_eviction_config, m_scheduler->get_block_size(), num_decoder_layers);
        }
        auto& cache_eviction_algo = m_seq_group_id_to_cache_eviction_algo_map[seq_id];

        cache_eviction_algo.register_new_token_scores(attention_scores_for_all_decoder_layers);
        auto logical_blocks_to_evict = cache_eviction_algo.evict_logical_blocks();

        m_scheduler->free_blocks_from_sequence(seq_id, logical_blocks_to_evict);

        auto seq_group_ptr_it = std::find_if(m_requests.begin(), m_requests.end(), [seq_id](const SequenceGroup::Ptr& val) { return val->has_sequence_with_id(seq_id); });
        OPENVINO_ASSERT(seq_group_ptr_it != m_requests.end(), "could not find sequence group with sequence ", seq_id);
        auto seq_group_ptr = *seq_group_ptr_it;
        size_t num_blocks_evicted = logical_blocks_to_evict[0].size();

        if (seq_group_to_num_blocks_evicted_map.find(seq_group_ptr) != seq_group_to_num_blocks_evicted_map.end()) {
            OPENVINO_ASSERT(seq_group_to_num_blocks_evicted_map[seq_group_ptr] == num_blocks_evicted, "internal error - each sequence in the same group must have the same number of blocks evicted");
        } else {
            seq_group_to_num_blocks_evicted_map[seq_group_ptr] = num_blocks_evicted;
        }

    }
    for (const auto& seq_group_ptr_and_num_blocks_evicted : seq_group_to_num_blocks_evicted_map) {
        // Assuming that the evicted blocks are always full (since they by design are only selected from intermediate-age blocks)
        auto seq_group_ptr = seq_group_ptr_and_num_blocks_evicted.first;
        auto num_blocks_evicted = seq_group_ptr_and_num_blocks_evicted.second;
        seq_group_ptr->register_token_eviction(num_blocks_evicted * m_scheduler->get_block_size());
    }
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::_fill_prompt_log_probs(std::vector<SequenceGroup::Ptr>& sequence_groups, ov::Tensor& logits) {
    const float * logits_data = logits.data<float>();
    ov::Shape logits_shape = logits.get_shape();
    OPENVINO_ASSERT(logits_shape.size() == 3);
    size_t batch_seq_len = logits_shape[1], vocab_size = logits_shape[2];
    for (size_t sequence_group_id = 0, currently_processed_tokens = 0; sequence_group_id < sequence_groups.size(); ++sequence_group_id) {
        SequenceGroup::Ptr sequence_group = sequence_groups[sequence_group_id];
        // requests not scheduled, in decoding phase or not echoing are not processed
        if (!sequence_group->is_scheduled() || sequence_group->get_context_len() > sequence_group->get_prompt_len() ||
            !sequence_group->get_sampling_parameters().echo)
            continue;

        size_t num_running_sequences = sequence_group->num_running_seqs();
        OPENVINO_ASSERT(num_running_sequences == 1);
        size_t actual_seq_len = sequence_group->get_num_scheduled_tokens();
        size_t padded_amount_of_processed_tokens = std::max(actual_seq_len, batch_seq_len);

        const float * sequence_group_logits_data = logits_data + vocab_size * currently_processed_tokens;

        size_t num_prompt_tokens_processed = sequence_group->get_num_processed_tokens();
        OPENVINO_ASSERT(num_prompt_tokens_processed + actual_seq_len <= sequence_group->get_prompt_len());

        // if we processed the whole prompt we don't include last logprob as it will be processed by the sampler (it's already completion)
        // otherwise we include it as it will be used in the next part of the prompt
        int exclude_last_logprob = 1;
        if (num_prompt_tokens_processed + actual_seq_len < sequence_group->get_prompt_len())
            exclude_last_logprob = 0;

        // if we start processing the prompt we add "fake" log prob for the first position (begin of sequence)
        if (num_prompt_tokens_processed == 0)
            sequence_group->append_prompt_log_prob(1.0);

        for (int token_logits_offset = 0, token_id_offset = num_prompt_tokens_processed + 1;
             token_logits_offset < actual_seq_len - exclude_last_logprob;
             token_logits_offset++, token_id_offset++) {

            const float* token_logits = (sequence_group_logits_data + token_logits_offset * vocab_size);
            int64_t token_id = sequence_group->get_prompt_ids()[token_id_offset];
            float token_logit = token_logits[token_id];

            // find max value for log softmax
            float max_value = -std::numeric_limits<float>::infinity();
            size_t max_index = 0;
            for (size_t i = 0; i < vocab_size; ++i) {
                if (token_logits[i] > max_value) {
                    max_value = token_logits[i];
                    max_index = i;
                }
            }

            // apply log softmax to token logit
            float log_sum = std::log(std::accumulate(
                token_logits, token_logits + vocab_size, 0.0f, [max_value](float accumulated, float to_add) {
                    return accumulated + std::exp(to_add - max_value);
            }));

            sequence_group->append_prompt_log_prob(token_logit - max_value - log_sum);
        }
        currently_processed_tokens += padded_amount_of_processed_tokens * num_running_sequences;
        // For max_new_tokens == 0, we don't reach sampling so need to notify handle separately
        if(sequence_group->get_sampling_parameters().max_new_tokens == 0) {
            sequence_group->notify_handle_echo_only();
        }
    }
}
}
