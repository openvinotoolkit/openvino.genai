// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "text_callback_streamer.hpp"
#include "continuous_batching_impl.hpp"
#include "paged_attention_transformations.hpp"

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

    // The model can be compiled for GPU as well
    std::shared_ptr<ov::Model> model = core.read_model(models_path + "/openvino_model.xml");

    DeviceConfig device_config(core, scheduler_config, device, plugin_config);

    apply_paged_attention_transformations(model, device_config);

    ov::InferRequest infer_request = core.compile_model(model, device_config.get_device(), plugin_config).create_infer_request();

    // setup KV caches
    m_cache_manager = std::make_shared<CacheManager>(device_config, core);
    for (size_t decoder_layer_id = 0; decoder_layer_id < device_config.get_num_layers(); ++decoder_layer_id) {
        infer_request.set_input_tensor(2 + decoder_layer_id * 2, m_cache_manager->get_key_cache(decoder_layer_id));
        infer_request.set_input_tensor(2 + decoder_layer_id * 2 + 1, m_cache_manager->get_value_cache(decoder_layer_id));
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

    m_scheduler = std::make_shared<Scheduler>(updated_config, can_use_partial_preemption);
    // and finally create model runner
    m_model_runner = std::make_shared<ModelRunner>(infer_request, updated_config);
    m_sampler = std::make_shared<Sampler>(m_tokenizer);
    m_sampler->set_seed(m_generation_config.rng_seed);

    // read default generation config
}

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
    {
        std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
        m_requests.insert(m_requests.end(), m_awaiting_requests.begin(), m_awaiting_requests.end());
        m_awaiting_requests.clear();
    }

    m_pipeline_metrics.requests = m_requests.size();
    Scheduler::Output scheduler_output;
    {
        static ManualTimer timer("scheduling");
        timer.start();
        scheduler_output = m_scheduler->schedule(m_requests);
        m_pipeline_metrics.scheduled_requests = scheduler_output.m_scheduled_sequence_groups_ids.size();
        m_pipeline_metrics.cache_usage = scheduler_output.m_cache_usage;
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

    SamplerOutput sampler_output;
    {
        static ManualTimer timer("sample");
        timer.start();
        sampler_output = m_sampler->sample(m_requests, logits);
        timer.end();
    }

    // process sampler_output (e.g. fork or drop sequences from BlockScheduler)
    {
        static ManualTimer timer("fork / free sequence");
        timer.start();

        for (const auto& pair : sampler_output.m_forked_sequences) {
            uint64_t parent_id = pair.first;
            const std::list<uint64_t>& child_ids = pair.second;
            for (auto & child_id : child_ids)
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

    bool continue_generation = true;
    while (has_non_finished_requests() && continue_generation) {
        step();
        if (streamer_ptr) {
            std::unordered_map<uint64_t, GenerationOutput> token = generations.at(0).get()->back();
            OPENVINO_ASSERT(1 == token.size());
            OPENVINO_ASSERT(1 == token.begin()->second.generated_ids.size());
            continue_generation = !streamer_ptr->put(token.begin()->second.generated_ids.at(0));
        }
    }
    if (streamer_ptr) {
        streamer_ptr->end();
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

std::vector<GenerationResult>
ContinuousBatchingPipeline::ContinuousBatchingImpl::generate(const std::vector<std::string>& prompts,
                                                             std::vector<ov::genai::GenerationConfig> sampling_params,
                                                             const StreamerVariant& streamer) {
    std::vector<ov::Tensor> input_ids;
    static ManualTimer timer("tokenize");
    if (m_is_chat_conversation) {
        OPENVINO_ASSERT(1 == prompts.size(), "Can't chat with multiple prompts");
        m_history.push_back({{"role", "user"}, {"content", prompts.at(0)}});
        constexpr bool add_generation_prompt = true;
        std::string history = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
        timer.start();
        input_ids.push_back(m_tokenizer.encode(history).input_ids);
        timer.end();
    } else {
        input_ids.reserve(prompts.size());
        for (const std::string& prompt : prompts) {
            timer.start();
            input_ids.push_back(m_tokenizer.encode(prompt).input_ids);
            timer.end();
        }
    }
    std::vector<EncodedGenerationResult> encoded = generate(input_ids, sampling_params, streamer);
    std::vector<GenerationResult> decoded;
    decoded.reserve(encoded.size());
    for (EncodedGenerationResult& res : encoded) {
        std::vector<std::string> generated;
        generated.reserve(res.m_generation_ids.size());
        for (size_t idx = 0; idx < res.m_generation_ids.size(); ++idx) {
            generated.push_back(m_tokenizer.decode(res.m_generation_ids.at(idx)));
            if (m_is_chat_conversation && 0 == idx) {
                m_history.push_back({{"role", "assistant"}, {"content", generated.back()}});
            }
        }
        decoded.push_back(GenerationResult{
            res.m_request_id,
            std::move(generated),
            std::move(res.m_scores),
            res.m_status
        });
    }
    return decoded;
}

void ContinuousBatchingPipeline::ContinuousBatchingImpl::_free_non_running_requests() {
    std::vector<SequenceGroup::Ptr>::iterator requests_iterator = m_requests.begin();
    while (requests_iterator != m_requests.end()) {
        const auto& request = *requests_iterator;
        if(request->has_finished() || request->out_of_memory() || request->handle_dropped()) {
            for (const auto& sequence: request->get_sequences()) {
                m_scheduler->free_sequence(sequence->get_id());
            }
            m_sampler->clear_beam_search_info(request->get_request_id());
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
}
