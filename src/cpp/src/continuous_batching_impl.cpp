// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "text_callback_streamer.hpp"
#include "continuous_batching_impl.hpp"
#include "utils.hpp"
#include "utils/paged_attention_transformations.hpp"

namespace ov::genai {
template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

ContinuousBatchingPipeline::ContinuousBatchingImpl::ContinuousBatchingImpl(
    const std::string& models_path,
    const Tokenizer& tokenizer,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& plugin_config) {
    m_tokenizer = tokenizer;

    ov::Core core;

    auto [core_plugin_config, compile_plugin_config] = ov::genai::utils::split_core_complile_config(plugin_config);
    core.set_property(core_plugin_config);

    // The model can be compiled for GPU as well
    std::shared_ptr<ov::Model> model = core.read_model(models_path + "/openvino_model.xml");

    DeviceConfig device_config(core, scheduler_config, device, compile_plugin_config);

    bool is_need_per_layer_cache_control = scheduler_config.use_cache_eviction;
    utils::apply_paged_attention_transformations(model, device_config, is_need_per_layer_cache_control);

    init(model, scheduler_config, compile_plugin_config, device_config, core);
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::_pull_awaiting_requests() {
    std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
    m_requests.insert(m_requests.end(), m_awaiting_requests.begin(), m_awaiting_requests.end());
    m_awaiting_requests.clear();
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::init(
    std::shared_ptr<ov::Model> model,
    const SchedulerConfig& scheduler_config,
    const ov::AnyMap& plugin_config,
    const DeviceConfig& device_config,
    ov::Core& core) {
    ov::InferRequest infer_request = core.compile_model(model, device_config.get_device(), plugin_config).create_infer_request();

    // setup KV caches
    m_cache_manager = std::make_shared<CacheManager>(device_config, core);
    for (size_t decoder_layer_id = 0; decoder_layer_id < device_config.get_num_layers(); ++decoder_layer_id) {
        infer_request.set_tensor(std::string("key_cache.") + std::to_string(decoder_layer_id), m_cache_manager->get_key_cache(decoder_layer_id));
        infer_request.set_tensor(std::string("value_cache.") + std::to_string(decoder_layer_id), m_cache_manager->get_value_cache(decoder_layer_id));
    }

    SchedulerConfig updated_config = scheduler_config;
    // update KV number in scheduler config
    if (scheduler_config.num_kv_blocks != device_config.get_num_kv_blocks()) {
        updated_config.num_kv_blocks = device_config.get_num_kv_blocks();
    }

    bool can_use_partial_preemption = true;
    if (device_config.get_device().find("GPU") != std::string::npos && !updated_config.dynamic_split_fuse) {
        // in case of executing a `vLLM-like` pipeline, it's better not to use partial eviction on the GPU,
        // as it may lead to performance slowdown
        can_use_partial_preemption = false;
    }

    m_scheduler = std::make_shared<Scheduler>(updated_config, device_config.get_num_layers(), can_use_partial_preemption);
    // and finally create model runner
    bool is_use_cache_eviction = m_scheduler->get_config().use_cache_eviction;
    if (is_use_cache_eviction) {
        m_model_runner = std::make_shared<ModelRunner>(infer_request, updated_config, device_config.get_num_layers(), true);
    } else {
        m_model_runner = std::make_shared<ModelRunner>(infer_request, updated_config, device_config.get_num_layers());
    }
    m_sampler = std::make_shared<Sampler>(m_tokenizer);
    m_sampler->set_seed(m_generation_config.rng_seed);
};


GenerationHandle
ContinuousBatchingPipeline::ContinuousBatchingImpl::add_request(uint64_t request_id,
                                                               const ov::Tensor& input_ids,
                                                               ov::genai::GenerationConfig sampling_params) {
    sampling_params.set_eos_token_id(m_tokenizer.get_eos_token_id());
    sampling_params.validate();
    SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(request_id, input_ids,
                                                                        sampling_params,
                                                                        m_scheduler->get_config().block_size,
                                                                        m_scheduler->get_config().enable_prefix_caching);
    sequence_group->set_sequence_group_ptr(sequence_group);
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

    // Pull awaiting requests
    _pull_awaiting_requests();

    m_pipeline_metrics.requests = m_requests.size();

    Scheduler::Output scheduler_output;
    {
        static ManualTimer timer("scheduling");
        timer.start();
        m_scheduler->clean_empty_blocks(m_requests);
        scheduler_output = m_scheduler->schedule(m_requests);
        m_pipeline_metrics.scheduled_requests = scheduler_output.m_scheduled_sequence_groups_ids.size();
        m_pipeline_metrics.cache_usage = scheduler_output.m_cache_usage;
        m_pipeline_metrics.max_cache_usage =
            std::max(m_pipeline_metrics.max_cache_usage, scheduler_output.m_cache_usage);
        _register_step_cache_usage(scheduler_output.m_cache_usage);
        m_pipeline_metrics.avg_cache_usage = _get_current_running_average_cache_usage();
        m_cache_manager->copy_blocks(scheduler_output.m_block_copy_map);
        timer.end();
    }

    // if no tokens were scheduled, we are out of memory
    if (scheduler_output.m_total_num_scheduled_tokens == 0) {
        for (size_t i = 0; i < m_requests.size(); ++i) {
            SequenceGroup::Ptr sequence_group = m_requests[i];
            sequence_group->set_out_of_memory();
            sequence_group->notify_handle();
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

        ov::InferRequest infer_request = m_model_runner->get_infer_request();
        ov::CompiledModel compiled_model = infer_request.get_compiled_model();
        const bool is_profiling_enabled = compiled_model.get_property(ov::enable_profiling);

        // collect detailed statistic
        if (is_profiling_enabled) {
            std::vector<ov::ProfilingInfo> profiling_info = m_model_runner->get_infer_request().get_profiling_info();
            for (const ov::ProfilingInfo& info : profiling_info) {
                double current_time = info.real_time.count();
                if (info.node_type == "PagedAttentionExtension") {
                    m_perf.m_paged_attention_time_ms += current_time;
                } else if (info.node_type == "FullyConnected") {
                    m_perf.m_matmul_time_ms += current_time;
                }
                m_perf.m_infer_total_ms += current_time;
            }
        }
    }

#ifdef DEBUG_CACHE_STATE_DUMP

    CacheStateDumper dumper(CacheStateDumper::get_run_id_for_generation_step(step_count, "before_eviction"));
    dumper.dump_cache_state(*m_scheduler, m_requests, step_count);
#endif
    const auto& sched_config = m_scheduler->get_config();

    // evict unimportant blocks from KV cache, if requested
    if (sched_config.use_cache_eviction) {
        maybe_evict_cache_blocks(sched_config);
    }

#ifdef DEBUG_CACHE_STATE_DUMP
    CacheStateDumper dumper_after(CacheStateDumper::get_run_id_for_generation_step(step_count, "eviction"));
    dumper_after.dump_cache_state(*m_scheduler, m_requests, step_count);
    step_count++;
#endif

    SamplerOutput sampler_output;
    {
        static ManualTimer timer("sample");
        timer.start();
        sampler_output = m_sampler->sample(m_requests, logits, m_is_validation_mode_enabled);
        timer.end();
    }

    // process sampler_output (e.g. fork or drop sequences from BlockScheduler)
    {
        static ManualTimer timer("fork / free sequence");
        timer.start();

        for (const auto& pair : sampler_output.m_forked_sequences) {
            uint64_t parent_id = pair.first;
            const std::list<uint64_t>& child_ids = pair.second;
            for (auto& child_id : child_ids)
                m_scheduler->fork_sequence(parent_id, child_id);
        }

        for (auto seq_id : sampler_output.m_dropped_sequences)
            m_scheduler->free_sequence(seq_id);

        timer.end();
    }

    // notify requests dropped by handle
    {
        static ManualTimer timer("notify requests dropped by handle");
        timer.start();
        _notify_requests_dropped_by_handle();
        timer.end();
    }

    // free non running requests for current step

    {
        static ManualTimer timer("free non running requests");
        timer.start();
        _free_non_running_requests();
        timer.end();
    }

    step_timer.end();
}

std::vector<EncodedGenerationResult>
ContinuousBatchingPipeline::ContinuousBatchingImpl::generate(const std::vector<ov::Tensor>& input_ids,
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

    std::vector<GenerationHandle> generations;
    for (size_t request_id = 0; request_id < input_ids.size(); ++request_id) {
        OPENVINO_ASSERT(1 == input_ids[request_id].get_shape().at(0), "Use multiple tensors to pass a batch.");
        generations.push_back(add_request(request_id, input_ids[request_id], sampling_params[request_id]));
    }

    std::vector<EncodedGenerationResult> results;
    results.reserve(m_awaiting_requests.size());

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

    bool continue_generation = true, step_throws_exception = false;
    while (has_non_finished_requests() && continue_generation) {
        try {
            step();
        } catch (...) {
            drop_requests();
            throw;
        }
        if (streamer_ptr && generations.at(0)->can_read()) {
            std::unordered_map<uint64_t, GenerationOutput> token = generations.at(0).get()->back();
            OPENVINO_ASSERT(1 == token.size());
            OPENVINO_ASSERT(1 == token.begin()->second.generated_ids.size());
            continue_generation = !streamer_ptr->put(token.begin()->second.generated_ids.at(0));
        }
    }

    if (streamer_ptr) {
        streamer_ptr->end();
    }

    if (!continue_generation) {
        drop_requests();
    } else {
        OPENVINO_ASSERT(m_requests.empty(), "Internal error: current request is supposed to be dropped within step() function as completed");
    }

    for (size_t generation_idx = 0; generation_idx < generations.size(); ++generation_idx) {
        const auto& generation = generations[generation_idx];
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

void ContinuousBatchingPipeline::ContinuousBatchingImpl::maybe_evict_cache_blocks(const SchedulerConfig& sched_config) {
    std::unordered_map<SequenceGroup::Ptr, size_t> seq_group_to_num_blocks_evicted_map;
    auto sequence_attention_scores = m_model_runner->get_last_attention_scores();
    for (auto& seq_id_and_attention_scores : sequence_attention_scores) {
        auto seq_id = seq_id_and_attention_scores.first;
        const auto& attention_scores_for_all_decoder_layers = seq_id_and_attention_scores.second;
        if (m_seq_group_id_to_cache_eviction_algo_map.find(seq_id) == m_seq_group_id_to_cache_eviction_algo_map.end()) {
            auto num_decoder_layers = attention_scores_for_all_decoder_layers.size();

            m_seq_group_id_to_cache_eviction_algo_map[seq_id] = CacheEvictionAlgorithm(sched_config.cache_eviction_config, sched_config.block_size, num_decoder_layers);
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
        seq_group_ptr->register_token_eviction(num_blocks_evicted * sched_config.block_size);
    }
}

}
