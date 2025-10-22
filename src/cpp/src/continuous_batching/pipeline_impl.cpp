// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <thread>
#include <optional>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#elif !defined(_WIN32)
#include <sys/sysinfo.h>
#endif

#include "openvino/genai/text_streamer.hpp"
#include "continuous_batching/pipeline_impl.hpp"
#include "utils.hpp"
#include "continuous_batching/paged_attention_transformations.hpp"
#include "lora/helper.hpp"
#include "continuous_batching/cache_state_dumper.hpp"

namespace {

// Returns available RAM memory on system if possible, otherwise returns std::numeric_limits<std::streamsize>::max()
size_t get_available_cpu_memory() {
#ifdef __APPLE__ 
    int64_t memsize;
    size_t len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &len, NULL, 0) == 0) {
        return memsize;
    }
#endif 

#if !defined(_WIN32)
    std::string token;
    std::ifstream file("/proc/meminfo");
    if(file.is_open()) {
        while(file >> token) {
            if(token == "MemTotal:") {
                size_t mem;
                if(file >> mem) {
                    if (mem > std::numeric_limits<size_t>::max() / 1024) {
                        return std::numeric_limits<size_t>::max();
                    }
                    return mem * 1024;
                } else {
                    return std::numeric_limits<size_t>::max();
                }
            }
            file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    }
#endif
    return std::numeric_limits<size_t>::max();
}

} // namespace

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

    bool is_need_per_layer_cache_control = scheduler_config.use_cache_eviction;
    bool allow_cache_rotation = scheduler_config.cache_eviction_config.apply_rotation;
    bool allow_xattention = scheduler_config.use_sparse_attention && scheduler_config.sparse_attention_config.mode == SparseAttentionMode::XATTENTION;
    utils::apply_paged_attention_transformations(model, is_need_per_layer_cache_control, allow_cache_rotation, allow_xattention);
    utils::apply_gather_before_matmul_transformation(model);

    initialize_pipeline(model, scheduler_config, device, properties);
}

ContinuousBatchingPipeline::ContinuousBatchingImpl::ContinuousBatchingImpl(
    const std::shared_ptr<ov::Model>& model,
    std::shared_ptr<InputsEmbedder> inputs_embedder,
    const Tokenizer& tokenizer,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& properties,
    const ov::genai::GenerationConfig& generation_config,
    bool is_validation_mode_enabled) : ContinuousBatchingImpl(model, tokenizer, scheduler_config, device, properties, generation_config, is_validation_mode_enabled){
    m_inputs_embedder = inputs_embedder;
    m_model_runner->set_embedding_model(inputs_embedder->get_embedding_model());
    m_model_input_type = ModelInputType::EMBEDDINGS;
}

ContinuousBatchingPipeline::ContinuousBatchingImpl::~ContinuousBatchingImpl() {
    // manually release all blocks, which can re-initialize OpenVINO plugins during destruction
    if (m_model_runner) {
        m_model_runner->get_infer_request().get_compiled_model().release_memory();
    }

    if (m_scheduler) {
        m_scheduler->release();
    }
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
    const std::string& device,
    const ov::AnyMap& properties) {
    m_device = device;
    // apply LoRA
    auto filtered_properties = extract_adapters_from_properties(properties, &m_generation_config.adapters);
    if (m_generation_config.adapters) {
        m_generation_config.adapters->set_tensor_name_prefix("base_model.model.");
        m_adapter_controller = AdapterController(model, *m_generation_config.adapters, device);   // TODO: Make the prefix name configurable
    }
    // Extract sampler_num_threads property if exists and remove it from properties
    size_t sampler_num_threads = std::thread::hardware_concurrency();
    auto sampler_num_threads_it = filtered_properties->find("sampler_num_threads");
    if (sampler_num_threads_it != filtered_properties->end()) {
        sampler_num_threads = sampler_num_threads_it->second.as<size_t>();
        filtered_properties.fork().erase("sampler_num_threads");   // do not use iterator sampler_num_threads_it because a forked container may not be the same container
    }

    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(model, device, *filtered_properties);
    std::vector<std::string> execution_devices = compiled_model.get_property(ov::execution_devices);
    const bool all_gpu_device =
        std::all_of(execution_devices.begin(), execution_devices.end(), [&](const std::string& device) {
            return device.find("GPU") != std::string::npos;
        });
    OPENVINO_ASSERT(all_gpu_device || execution_devices.size() == 1,
                    "Continuous batching: execution device is expected to be single CPU / single GPU / multi GPUs");
    const std::string execution_device = execution_devices[0];

    ov::genai::utils::print_compiled_model_properties(compiled_model, "LLM with Paged Attention");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // Cache manager
    std::shared_ptr<CacheManager> cache_manager = std::make_shared<CacheManager>(infer_request);
    m_num_decoder_layers = cache_manager->get_num_decoder_layers();
    m_block_size = cache_manager->get_block_size();


    // Scheduler configuration
    SchedulerConfig normalized_config = scheduler_config;
    size_t total_mem_size;
    if (execution_device.find("GPU") != std::string::npos) {
        total_mem_size = utils::get_available_gpu_memory(execution_device, m_num_decoder_layers);
    } else {
        total_mem_size = get_available_cpu_memory();
    }
    if (normalized_config.num_kv_blocks == 0 && normalized_config.cache_size > 0) {
        size_t size_in_bytes = normalized_config.cache_size * 1024 * 1024 * 1024; // convert GBs to bytes
        OPENVINO_ASSERT(size_in_bytes <= total_mem_size, "Requested KV-cache size is larger than available memory size on the system.");
        normalized_config.num_kv_blocks = size_in_bytes / cache_manager->get_block_size_in_bytes();
    }
    if (normalized_config.num_kv_blocks > 0) {
        size_t size_in_bytes = cache_manager->get_block_size_in_bytes() * normalized_config.num_kv_blocks;
        OPENVINO_ASSERT(size_in_bytes <= total_mem_size, "Requested number of KV-blocks require more memory than available on the system.");
    }

    bool can_use_partial_preemption = true;
    if (execution_device.find("GPU") != std::string::npos && !normalized_config.dynamic_split_fuse) {
        // in case of executing a `vLLM-like` pipeline, it's better not to use partial eviction on the GPU,
        // as it may lead to performance slowdown
        can_use_partial_preemption = false;
    }

    // Scheduler and Model Runner instantiation
    bool is_use_xattention = scheduler_config.use_sparse_attention && scheduler_config.sparse_attention_config.mode == SparseAttentionMode::XATTENTION;
    bool is_use_cache_eviction = scheduler_config.use_cache_eviction;
    if (is_use_cache_eviction) {
        const auto& eviction_config = scheduler_config.cache_eviction_config;
        m_scheduler = std::make_shared<Scheduler>(m_block_size, cache_manager, normalized_config, m_num_decoder_layers, can_use_partial_preemption, eviction_config.snapkv_window_size);

        bool is_apply_rotation = eviction_config.apply_rotation;
        m_model_runner = std::make_shared<ModelRunner>(infer_request,
                                                       m_block_size,
                                                       m_num_decoder_layers,
                                                       /* collect_attention_scores = */ true,
                                                       /* is_use_per_layer_cache_control = */ true,
                                                       /* is_use_rotation_inputs = */ is_apply_rotation,
                                                       /* is_aggregate_attention_scores = */ true,
                                                       is_use_xattention);
        if (eviction_config.apply_rotation) {
            _prepare_rotation_data_storage(normalized_config, cache_manager->get_v_head_size(0));
        }
    } else {
        m_scheduler = std::make_shared<Scheduler>(m_block_size, cache_manager, normalized_config, m_num_decoder_layers, can_use_partial_preemption);
        m_model_runner =
            std::make_shared<ModelRunner>(infer_request, m_block_size, m_num_decoder_layers,
                                                       /* collect_attention_scores = */ false,
                                                       /* is_use_per_layer_cache_control = */ false,
                                                       /* is_use_rotation_inputs = */ false,
                                                       /* is_aggregate_attention_scores = */ false,
                                                       is_use_xattention);
    }

    m_sampler = std::make_shared<Sampler>(m_tokenizer, sampler_num_threads);
    m_sampler->set_seed(m_generation_config.rng_seed);

    // If eos_token_id was not provided, take value
    if (m_generation_config.eos_token_id == -1)
        m_generation_config.set_eos_token_id(m_tokenizer.get_eos_token_id());
};


void ContinuousBatchingPipeline::ContinuousBatchingImpl::_prepare_rotation_data_storage(const SchedulerConfig& normalized_config, size_t embedding_size) {
    m_rotation_deltas_stores.reserve(m_num_decoder_layers);
    ov::Shape rotation_deltas_store_shape{normalized_config.num_kv_blocks, 1}; // last dim can be later changed to BLOCK_SIZE for per-token granularity
    for (size_t i = 0; i < m_num_decoder_layers; i++) {
        ov::Tensor store(ov::element::i32, rotation_deltas_store_shape);
        std::memset(store.data(), 0, store.get_byte_size());
        m_rotation_deltas_stores.push_back(store);
    }

    size_t max_sequence_cache_occupation_length_in_blocks = normalized_config.max_num_batched_tokens / m_block_size  + 1;
    m_cache_rotation_calculator = std::make_shared<CacheRotationCalculator>(
        m_block_size,
        max_sequence_cache_occupation_length_in_blocks,
        embedding_size);
    auto rotation_trig_lut = ov::Tensor(ov::element::f32, ov::Shape{max_sequence_cache_occupation_length_in_blocks, embedding_size});
    float* rotation_trig_lut_data = rotation_trig_lut.data<float>();
    std::memset(rotation_trig_lut_data, 0, rotation_trig_lut.get_byte_size());

    const auto& cos_lut = m_cache_rotation_calculator->get_cos_lut();
    const auto& sin_lut = m_cache_rotation_calculator->get_sin_lut();


    for (size_t pos_idx = 0; pos_idx < max_sequence_cache_occupation_length_in_blocks; pos_idx++) {
        for (size_t embedding_pair_idx = 0; embedding_pair_idx < cos_lut[0].size(); embedding_pair_idx++) {
            rotation_trig_lut_data[pos_idx * embedding_size + embedding_pair_idx] = cos_lut[pos_idx][embedding_pair_idx];
            rotation_trig_lut_data[pos_idx * embedding_size + embedding_size / 2 + embedding_pair_idx] = sin_lut[pos_idx][embedding_pair_idx];
        }
    }

    m_model_runner->set_cache_rotation_trig_lut(std::move(rotation_trig_lut));
}

GenerationHandle
ContinuousBatchingPipeline::ContinuousBatchingImpl::add_request(
    uint64_t request_id,
    const ov::Tensor& input_ids,
    const ov::genai::GenerationConfig& sampling_params,
    std::optional<ov::Tensor> token_type_ids) {
    auto sampling_params_copy = sampling_params;
    // If stop_token_ids were not provided, take value from default m_generation_config
    if (sampling_params_copy.stop_token_ids.empty())
        sampling_params_copy.stop_token_ids = m_generation_config.stop_token_ids;
    // If eos_token_id was not provided, take value from default m_generation_config
    if (sampling_params_copy.eos_token_id == -1)
        sampling_params_copy.set_eos_token_id(m_generation_config.eos_token_id);
    sampling_params_copy.validate();
    size_t prompt_len;
    if (input_ids.get_shape().size() > 1) {
        prompt_len = input_ids.get_shape()[1];
    } else {
        prompt_len = input_ids.get_size();
    }
    OPENVINO_ASSERT(sampling_params_copy.max_length > prompt_len, "'max_length' must be greater than the number of prompt tokens");

    auto sequence_group = std::make_shared<SequenceGroup>(request_id, input_ids, sampling_params_copy, m_block_size, token_type_ids);

    if (m_scheduler->get_config().enable_prefix_caching) {
        m_scheduler->restore_cached_blocks(sequence_group);
    }

    {
        std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
        m_awaiting_requests.push_back(sequence_group);
    }

    return std::make_shared<GenerationHandleImpl>(sequence_group->get_generation_stream(), sampling_params_copy);
}

GenerationHandle
ContinuousBatchingPipeline::ContinuousBatchingImpl::add_request(uint64_t request_id,
                                                                const std::string& prompt,
                                                                const ov::genai::GenerationConfig& sampling_params) {
    ov::Tensor inputs;
    ov::genai::VLMPerfMetrics metrics;
    if (m_model_input_type == ModelInputType::TOKENS) {
        static ManualTimer timer("tokenize");
        timer.start();
        inputs = m_tokenizer.encode(prompt).input_ids;
        timer.end();
        return add_request(request_id, inputs, sampling_params);
    } else if (m_model_input_type == ModelInputType::EMBEDDINGS) {
        return ContinuousBatchingPipeline::IContinuousBatchingPipeline::add_request(request_id, prompt, {}, sampling_params);
    } else {
        OPENVINO_THROW("Unknown model input type.");
    }

    return add_request(request_id, inputs, sampling_params);
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

        const auto& sched_config = m_scheduler->get_config();
        if (sched_config.use_cache_eviction && sched_config.cache_eviction_config.apply_rotation) {
            _compute_cache_rotation_data(m_requests, scheduler_output);
            m_model_runner->set_cache_rotation_data(std::move(m_current_step_rotated_block_indices_per_sequence),
                                                    std::move(m_current_step_rotation_deltas));
        }

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
        const auto infer_start = std::chrono::steady_clock::now();
        timer.start();
        logits = m_model_runner->forward(m_requests, scheduler_output);
        const auto infer_end = std::chrono::steady_clock::now();
        m_pipeline_metrics.inference_duration = PerfMetrics::get_microsec(infer_end - infer_start);
        timer.end();
    }

#ifdef DEBUG_CACHE_STATE_DUMP
    CacheStateDumper dumper(CacheStateDumper::get_run_id_for_generation_step(step_count, "before_eviction"));
    dumper.dump_cache_state(*m_scheduler, m_requests, step_count);
#endif

    // evict unimportant blocks from KV cache, if requested
    const auto& sched_config = m_scheduler->get_config();
    if (sched_config.use_cache_eviction) {
        _maybe_evict_cache_blocks(sched_config, scheduler_output);
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
        m_batch_size = sampler_output.num_generated_tokens;
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
    
    // append embeddings for generated tokens
    if (m_model_input_type == ModelInputType::EMBEDDINGS)
        m_model_runner->append_embeddings(m_requests, scheduler_output);

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
                                                             const StreamerVariant& streamer,
                                                             const std::optional<std::vector<ov::Tensor>> token_type_ids) {

    _reset_cache_usage_statistics();
    ManualTimer generate_timer("generate()");
    generate_timer.start();

    OPENVINO_ASSERT(!has_non_finished_requests(), "Generate cannot be called while ContinuousBatchingPipeline is already in running state. Use ContinuousBatchingPipeline::add_request");
    OPENVINO_ASSERT(input_ids.size() == sampling_params.size());

    auto start_time =  std::chrono::steady_clock::now();
    PerfMetrics perf_metrics;
    auto& raw_perf_counters = perf_metrics.raw_metrics;
    raw_perf_counters.m_inference_durations =  {{ MicroSeconds(0.0f) }};

    // checks that all requests has the same LoRA adapters property value
    for (size_t i = 1; i < sampling_params.size(); ++i) {
        OPENVINO_ASSERT(sampling_params[i - 1].adapters == sampling_params[i].adapters,
            "LoRA adapters value must be the same for all requests");
    }
    set_adapters(sampling_params[0].adapters);

    const auto streamer_ptr = std::make_shared<ThreadedStreamerWrapper>(streamer, m_tokenizer);

    OPENVINO_ASSERT(!streamer_ptr->has_callback() || input_ids.size() == 1 && sampling_params[0].num_return_sequences == 1 &&
        (sampling_params[0].is_greedy_decoding() || sampling_params[0].is_multinomial()),
        "Currently streaming is possible only with batch size=1 and only for greedy or multinomial decoding");

    std::vector<GenerationHandle> generations;
    for (size_t request_id = 0; request_id < input_ids.size(); ++request_id) {
        OPENVINO_ASSERT(1 == input_ids[request_id].get_shape().at(0), "Use multiple tensors to pass a batch.");
        bool has_valid_token = token_type_ids.has_value() && request_id < token_type_ids->size();
        generations.push_back(
            add_request(request_id, input_ids[request_id], sampling_params[request_id], has_valid_token ? std::make_optional((*token_type_ids)[request_id]) : std::nullopt)
        );
    }

    auto all_requests = get_awaiting_requests(); // we need to store all requests to get results from them once generation has finished

    GenerationHandle& generation = generations.at(0);

    streamer_ptr->start();
    m_sampler->clear_structured_output_compile_times();
    while (has_non_finished_requests()) {
        try {
            const auto infer_start = std::chrono::steady_clock::now();
            step();
            
            // During prefill step (or steps if max_batch_size < prompt_len) we don't generate new tokens,
            // but still inference took place, so we need to add this time to the total inference duration.
            raw_perf_counters.m_inference_durations[0] += MicroSeconds(m_pipeline_metrics.inference_duration);
            if (m_batch_size > 0) {
                const auto infer_end = std::chrono::steady_clock::now();
                const auto infer_ms = PerfMetrics::get_microsec(infer_end - infer_start);
                raw_perf_counters.m_token_infer_durations.emplace_back(infer_ms);
                raw_perf_counters.m_new_token_times.emplace_back(infer_end);
                raw_perf_counters.m_batch_sizes.emplace_back(m_batch_size);
            }
        } catch (...) {
            drop_requests(); // remove all requests from pipeline state in case of exception
            streamer_ptr->end();
            std::rethrow_exception(std::current_exception());
        }
        stream_tokens(streamer_ptr, generation);
    }

    auto times = m_sampler->get_structured_output_times();
    perf_metrics.grammar_compiler_init_times = times.first;
    for (const auto& t: times.second) {
        raw_perf_counters.m_grammar_compile_times.emplace_back(t);
    }

    // waiting for competion of streaming
    streamer_ptr->end();

    OPENVINO_ASSERT(m_requests.empty(), "Internal error: current request is supposed to be dropped within step() function as completed");

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
        result.m_status = request->get_generation_stream()->get_status();

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

        // The same perf metrics for each sequence, only tokenization/detokenization will differ.
        perf_metrics.raw_metrics.generate_durations.clear();
        perf_metrics.raw_metrics.generate_durations.emplace_back(PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start_time));
        perf_metrics.num_input_tokens = request->get_prompt_len();
        perf_metrics.evaluate_statistics(start_time);

        result.perf_metrics = perf_metrics;
        results.push_back(std::move(result));
    }

    OPENVINO_ASSERT(results.size() == input_ids.size());

    generate_timer.end();
    
    const auto& scheduler_config = m_scheduler->get_config();
    // Clear KV-cache in case of dynamic cache allocation and no prefix caching
    if (!scheduler_config.enable_prefix_caching && scheduler_config.cache_size == 0 && scheduler_config.num_kv_blocks == 0) {
        m_scheduler->clear_kv_cache();
    }
    return results;
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::_free_non_running_requests() {
    std::vector<SequenceGroup::Ptr>::iterator requests_iterator = m_requests.begin();
    while (requests_iterator != m_requests.end()) {
        const auto& request = *requests_iterator;
        if(request->has_finished() || request->handle_stopped() || request->handle_cancelled()) {
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
        if (request->handle_stopped() || request->handle_cancelled())
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

void ContinuousBatchingPipeline::ContinuousBatchingImpl::_reset_cache_usage_statistics() {
    m_previous_step_cache_usages.clear();
    m_pipeline_metrics.max_cache_usage = 0.0;
    m_pipeline_metrics.avg_cache_usage = 0.0;
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::drop_requests() {
    for (const std::shared_ptr<ov::genai::SequenceGroup> request : m_requests) {
        for (const auto& sequence: request->get_sequences()) {
            if (m_scheduler->has_block_table(sequence->get_id())) {
                m_scheduler->free_sequence(sequence->get_id());
            }
        }
        m_sampler->clear_request_info(request->get_request_id());
    }
    m_requests.clear();
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::_compute_cache_rotation_data(const std::vector<SequenceGroup::Ptr>& sequence_groups,
        const Scheduler::Output& scheduler_output) {
    size_t num_sequence_groups = scheduler_output.m_scheduled_sequence_groups_ids.size();
    std::map<size_t, size_t> live_seq_ids_to_num_occupied_blocks;
    for (size_t i = 0; i < num_sequence_groups; ++i) {
        size_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
        SequenceGroup::CPtr sequence_group = sequence_groups[seq_group_id];
        std::vector<Sequence::CPtr> running_sequences = sequence_group->get_running_sequences();
        size_t num_running_sequences = running_sequences.size();

        for (size_t i = 0; i < num_running_sequences; ++i) {
            Sequence::CPtr sequence = running_sequences[i];
            size_t num_blocks = sequence_group->get_num_logical_blocks();
            size_t seq_id = sequence->get_id();
            OPENVINO_ASSERT(live_seq_ids_to_num_occupied_blocks.find(seq_id) == live_seq_ids_to_num_occupied_blocks.end(),
                    "duplicate seq_id ", seq_id, " among sequence groups");
            live_seq_ids_to_num_occupied_blocks[seq_id] = num_blocks;
        }
    }

    // necessary since we move from these members during previous steps
    m_current_step_rotated_block_indices_per_sequence.clear();
    m_current_step_rotated_block_indices_per_sequence.resize(m_num_decoder_layers);
    m_current_step_rotation_deltas.clear();

    std::vector<size_t> num_blocks_to_rotate_for_each_layer(m_num_decoder_layers, 0);


    for (const auto& seq_id_and_evicted_blocks : m_previous_evicted_block_logical_indices_per_sequence) {
        size_t seq_id = seq_id_and_evicted_blocks.first;
        // Skip sequences that, in the meanwhile before previous step's forward execution and now,
        // have left the cache (e.g. finished or were preempted)
        if (live_seq_ids_to_num_occupied_blocks.find(seq_id) == live_seq_ids_to_num_occupied_blocks.end()) {
            continue;
        }

        const auto& logical_blocks_to_evict = seq_id_and_evicted_blocks.second;

        for (size_t layer_idx = 0; layer_idx < logical_blocks_to_evict.size(); layer_idx++) {
            if (logical_blocks_to_evict[layer_idx].empty()) {
                continue;
            }
            size_t num_blocks_before_eviction = m_previous_num_blocks_before_eviction_per_sequence[seq_id];
            auto rotation_multipliers =
                m_cache_rotation_calculator->get_rotation_data(logical_blocks_to_evict[layer_idx],
                                                                       num_blocks_before_eviction);
            for (size_t i = 0; i < rotation_multipliers.size(); i++) {
                const auto& block_rotation_data = rotation_multipliers[i];

                m_current_step_rotated_block_indices_per_sequence[layer_idx][seq_id].push_back(
                    block_rotation_data.logical_block_idx);

                size_t block_offset = num_blocks_to_rotate_for_each_layer[layer_idx];
                auto rotation_deltas_tensor_data =
                    m_rotation_deltas_stores[layer_idx].data<int32_t>() + block_offset;
                for (size_t tok_idx = 0; tok_idx < m_block_size; tok_idx++) {
                   rotation_deltas_tensor_data[tok_idx] = block_rotation_data.rotation_delta / m_block_size;
                }
                num_blocks_to_rotate_for_each_layer[layer_idx] += 1;
            }
        }
    }
    // Select the previously filled rotation coefficients from the store tensor
    for (size_t i = 0; i < m_num_decoder_layers; i++) {
        m_current_step_rotation_deltas.emplace_back(
            m_rotation_deltas_stores[i],
            ov::Coordinate{0, 0},
            ov::Coordinate{num_blocks_to_rotate_for_each_layer[i], 1});
    }
}


void ContinuousBatchingPipeline::ContinuousBatchingImpl::_maybe_evict_cache_blocks(const SchedulerConfig& sched_config, const Scheduler::Output& scheduler_output) {
    std::unordered_map<SequenceGroup::Ptr, size_t> seq_group_to_num_blocks_evicted_map;
    auto sequence_attention_scores = m_model_runner->get_last_attention_scores();

    OPENVINO_ASSERT(!sequence_attention_scores.empty());
    size_t num_decoder_layers = sequence_attention_scores.begin()->second.size();

    m_previous_evicted_block_logical_indices_per_sequence.clear();
    m_previous_num_blocks_before_eviction_per_sequence.clear();

    for (auto& seq_id_and_attention_scores : sequence_attention_scores) {
        auto seq_id = seq_id_and_attention_scores.first;
        const auto& attention_scores_for_all_decoder_layers = seq_id_and_attention_scores.second;
        if (m_seq_group_id_to_cache_eviction_algo_map.find(seq_id) == m_seq_group_id_to_cache_eviction_algo_map.end()) {
            constexpr size_t MAX_POOL_WINDOW_SIZE = 7;
            m_seq_group_id_to_cache_eviction_algo_map[seq_id] = CacheEvictionAlgorithm(sched_config.cache_eviction_config, m_block_size, num_decoder_layers, MAX_POOL_WINDOW_SIZE);
        }
        auto& cache_eviction_algo = m_seq_group_id_to_cache_eviction_algo_map[seq_id];
        std::set<size_t> skip_set;
        if (scheduler_output.m_apply_sparse_attention_mask) {
            const auto& skip_map = scheduler_output.m_sparse_attention_skipped_logical_blocks;
            auto it = skip_map.find(seq_id);
            if (it != skip_map.end()) {
                skip_set = it->second;
            }
        }

        if (skip_set.empty()) {
            // For now, will only register token scores from the dense attention stages
            cache_eviction_algo.register_new_token_scores(attention_scores_for_all_decoder_layers, skip_set, scheduler_output.m_score_aggregation_windows.at(seq_id));
        }

        auto seq_group_ptr_it = std::find_if(m_requests.begin(), m_requests.end(), [seq_id](const SequenceGroup::Ptr& val) { return val->has_sequence_with_id(seq_id); });
        OPENVINO_ASSERT(seq_group_ptr_it != m_requests.end(), "could not find sequence group with sequence ", seq_id);
        auto seq_group_ptr = *seq_group_ptr_it;

         if (!seq_group_ptr->can_generate_tokens()) {
             // do not evict during prefill
             continue;
         }

        m_previous_num_blocks_before_eviction_per_sequence[seq_id] = seq_group_ptr->get_num_logical_blocks();

        auto logical_blocks_to_evict = cache_eviction_algo.evict_logical_blocks();
        m_previous_evicted_block_logical_indices_per_sequence[seq_id] = logical_blocks_to_evict;

        m_scheduler->free_blocks_from_sequence(seq_id, logical_blocks_to_evict);

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
        seq_group_ptr->register_token_eviction(num_blocks_evicted * m_block_size);
    }
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::_fill_prompt_log_probs(std::vector<SequenceGroup::Ptr>& sequence_groups, ov::Tensor& logits) {
    const float * logits_data = logits.data<float>();
    ov::Shape logits_shape = logits.get_shape();
    OPENVINO_ASSERT(logits_shape.size() == 3);
    size_t vocab_size = logits_shape[2];
    for (size_t sequence_group_id = 0, currently_processed_tokens = 0; sequence_group_id < sequence_groups.size(); ++sequence_group_id) {
        SequenceGroup::Ptr sequence_group = sequence_groups[sequence_group_id];
        // requests not scheduled, in decoding phase or not echoing are not processed
        if (!sequence_group->is_scheduled() || sequence_group->get_context_len() > sequence_group->get_prompt_len() ||
            !sequence_group->get_sampling_parameters().echo)
            continue;

        size_t num_running_sequences = sequence_group->num_running_seqs();
        OPENVINO_ASSERT(num_running_sequences == 1);
        size_t output_seq_len = sequence_group->get_output_seq_len();

        const float * sequence_group_logits_data = logits_data + vocab_size * currently_processed_tokens;

        size_t num_prompt_tokens_processed = sequence_group->get_num_processed_tokens();
        OPENVINO_ASSERT(num_prompt_tokens_processed + output_seq_len <= sequence_group->get_prompt_len());

        // if we processed the whole prompt we don't include last logprob as it will be processed by the sampler (it's already completion)
        // otherwise we include it as it will be used in the next part of the prompt
        int exclude_last_logprob = 1;
        if (num_prompt_tokens_processed + output_seq_len < sequence_group->get_prompt_len())
            exclude_last_logprob = 0;

        // if we start processing the prompt we add "fake" log prob for the first position (begin of sequence)
        if (num_prompt_tokens_processed == 0)
            sequence_group->append_prompt_log_prob(1.0);

        for (int token_logits_offset = 0, token_id_offset = num_prompt_tokens_processed + 1;
             token_logits_offset < output_seq_len - exclude_last_logprob;
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
        currently_processed_tokens += output_seq_len * num_running_sequences;
        // For max_new_tokens == 0, we don't reach sampling so need to notify handle separately
        if(sequence_group->get_max_new_tokens() == 0) {
            sequence_group->notify_handle_echo_only();
        }
    }
}

std::vector<SequenceGroup::Ptr> ContinuousBatchingPipeline::ContinuousBatchingImpl::get_awaiting_requests() {
    std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
    return m_awaiting_requests;
}
} // namespace ov::genai
