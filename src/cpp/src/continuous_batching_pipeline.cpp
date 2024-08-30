// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <mutex>
#include <memory>

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "cache_manager.hpp"
#include "sampler.hpp"
#include "model_runner.hpp"
#include "scheduler.hpp"
#include "text_callback_streamer.hpp"
#include "timer.hpp"
#include "debug_utils.hpp"

using namespace ov::genai;

template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

void apply_paged_attention_transformations(std::shared_ptr<ov::Model> model);
void set_type_and_shape_to_kv_cache(std::shared_ptr<ov::Model> model, DeviceConfig& device_config);

std::shared_ptr<ov::Model>
ov::genai::read_model_and_apply_paged_attention(const std::string& models_path, ov::Core& core) {
    auto model = core.read_model(models_path + "/openvino_model.xml");
    apply_paged_attention_transformations(model);
    return model;
}

class ContinuousBatchingPipeline::Impl {
    ov::genai::Tokenizer m_tokenizer;
    std::shared_ptr<Scheduler> m_scheduler;
    std::shared_ptr<CacheManager> m_cache_manager;
    std::shared_ptr<ModelRunner> m_model_runner;
    std::shared_ptr<Sampler> m_sampler;
    bool m_is_validation_mode_enabled = false;

    // TODO (mzegla): GenerationConfig is request specific object
    // and pipeline only uses default rng_seed. 
    ov::genai::GenerationConfig m_generation_config;

    PipelineMetrics m_pipeline_metrics;

    struct PerfTime {
        float m_paged_attention_time_ms = 0.0f;
        float m_matmul_time_ms = 0.0f;
        float m_infer_total_ms = 0.0f;

        ~PerfTime() {
            std::cout << "Inference requests aggregated statistic: " << std::endl;
            std::cout << "Paged attention % of inference execution: " << (m_paged_attention_time_ms / m_infer_total_ms) * 100 << std::endl;
            std::cout << "MatMul % of inference execution: " << (m_matmul_time_ms / m_infer_total_ms) * 100 << std::endl;
            std::cout << "Total inference execution secs: " << m_infer_total_ms / 1000. << std::endl;
            std::cout << std::endl;
        }
    } m_perf;

    // current requests to process
    std::vector<SequenceGroup::Ptr> m_requests;
    // requests added to the pipeline that will be added to m_requests in the next iteration
    std::vector<SequenceGroup::Ptr> m_awaiting_requests;
    // Mutex protecting access to m_awaiting_requests, so add_request and step methods can be called from different threads
    std::mutex m_awaiting_requests_mutex;
    bool m_is_chat_conversation = false;
    ChatHistory m_history;


    void _notify_requests_dropped_by_handle() {
        // Notify the last time by pushing empty output
        // This causes read() to unblock by adding anything to the queue
        for (SequenceGroup::Ptr& request : m_requests) {
            if (request->handle_dropped())
                request->push_empty_outputs();
        }
    }

    void _free_non_running_requests() {
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

    inline void
    compile_model(std::shared_ptr<ov::Model> model,
                  const SchedulerConfig& scheduler_config,
                  const ov::AnyMap& plugin_config,
                  const std::string& device,
                  ov::Core& core) {
        DeviceConfig device_config(core, scheduler_config, device);
        set_type_and_shape_to_kv_cache(model, device_config);

        ov::InferRequest infer_request = core.compile_model(model, device_config.get_device(), plugin_config).create_infer_request();

        // setup KV caches
        m_cache_manager = std::make_shared<CacheManager>(device_config);
        for (size_t decoder_layer_id = 0; decoder_layer_id < device_config.get_num_layers(); ++decoder_layer_id) {
            infer_request.set_input_tensor(2 + decoder_layer_id * 2, m_cache_manager->get_key_cache(decoder_layer_id));
            infer_request.set_input_tensor(2 + decoder_layer_id * 2 + 1, m_cache_manager->get_value_cache(decoder_layer_id));
        }
        SchedulerConfig updated_config = scheduler_config;
        // update KV number in scheduler config
        if (scheduler_config.num_kv_blocks != device_config.get_num_kv_blocks()) {
            updated_config.num_kv_blocks = device_config.get_num_kv_blocks();
        }
        m_scheduler = std::make_shared<Scheduler>(updated_config);
        // and finally create model runner
        m_model_runner = std::make_shared<ModelRunner>(infer_request, updated_config);
        m_sampler = std::make_shared<Sampler>();
        m_sampler->set_seed(m_generation_config.rng_seed);
        m_sampler->set_seed(0);

        // read default generation config
    }

    inline void pull_awaiting_requests() {
        if (m_requests.empty()) {
            // Pull awaiting requests
            if (!m_awaiting_requests.empty()) {
                std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
                m_requests.insert(m_requests.end(), m_awaiting_requests.begin(), m_awaiting_requests.end());
                m_awaiting_requests.clear();
            }
        }
    }

public:
    Impl(const std::string& models_path, const Tokenizer& tokenizer, const SchedulerConfig& scheduler_config, const std::string& device, const ov::AnyMap& plugin_config) :
        m_tokenizer{tokenizer} {
        ov::Core core;
        // The model can be compiled for GPU as well
        std::shared_ptr<ov::Model> model = read_model_and_apply_paged_attention(models_path, core);
        compile_model(model, scheduler_config, plugin_config, device, core);
    }

    Impl(const std::string& models_path, const SchedulerConfig& scheduler_config, const std::string& device, const ov::AnyMap& llm_plugin_config, const ov::AnyMap& tokenizer_plugin_config)
        : Impl{models_path, Tokenizer(models_path, tokenizer_plugin_config), scheduler_config, device, llm_plugin_config} {}
    
    Impl(ov::Core& core, std::shared_ptr<ov::Model> model, const Tokenizer& tokenizer, const SchedulerConfig& scheduler_config, const std::string& device, const ov::AnyMap& plugin_config, bool is_validation_mode = false) :
        m_is_validation_mode_enabled(is_validation_mode),
        m_tokenizer{tokenizer} {
        compile_model(model, scheduler_config, plugin_config, device, core);
    }

    ov::genai::GenerationConfig get_config() const {
        return m_generation_config;
    }

    PipelineMetrics get_metrics() const {
        return m_pipeline_metrics;
    }

    ov::genai::Tokenizer get_tokenizer() {
        return m_tokenizer;
    }

    GenerationHandle add_request(uint64_t request_id, const ov::Tensor& input_ids, ov::genai::GenerationConfig sampling_params) {
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
    }

    GenerationHandle add_request(uint64_t request_id, const std::string& prompt, ov::genai::GenerationConfig sampling_params) {
        static ManualTimer timer("tokenize");
        timer.start();
        ov::Tensor input_ids = m_tokenizer.encode(prompt).input_ids;
        timer.end();
        return add_request(request_id, input_ids, sampling_params);
    }

    void step() {
        static ManualTimer step_timer("step()");
        step_timer.start();

        pull_awaiting_requests();

        m_pipeline_metrics.requests = m_requests.size();
        Scheduler::Output scheduler_output;
        {
            static ManualTimer timer("scheduling");
            timer.start();
            // todo: iefode: to move to other place?
            m_scheduler->clean_empty_blocks(m_requests);
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

            // collect detailed statistic
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

    bool has_non_finished_requests() {
        std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
        return !m_awaiting_requests.empty() || !m_requests.empty();
    }

    std::vector<EncodedGenerationResult> generate(const std::vector<ov::Tensor>& input_ids, const std::vector<GenerationConfig>& sampling_params, const StreamerVariant& streamer) {
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
                OPENVINO_ASSERT(1 == token.begin()->second.generated_token_ids.size());
                continue_generation = !streamer_ptr->put(token.begin()->second.generated_token_ids.at(0));
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
                result.m_generation_ids.push_back(std::move(generation_output.generated_token_ids));
                result.m_scores.push_back(generation_output.score);
            }
            result.m_status = generation->get_status();
            results.push_back(std::move(result));
        }

        OPENVINO_ASSERT(results.size() == input_ids.size());
        return results;
    }

    std::vector<GenerationResult> generate(const std::vector<std::string>& prompts, std::vector<ov::genai::GenerationConfig> sampling_params, const StreamerVariant& streamer) {
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

    void start_chat(const std::string& system_message) {
        if (!system_message.empty()) {
            m_history.push_back({{"role", "system"}, {"content", system_message}});
        }
        m_is_chat_conversation = true;
    };

    void finish_chat() {
        m_is_chat_conversation = false;
        m_history.clear();
    };

    void finish_all_requests() {
        while (!m_requests.empty()) {
            const auto& request = *m_requests.rbegin();
            for (const auto& sequence : request->get_sequences()) {
                m_scheduler->free_sequence(sequence->get_id());
            }
            m_sampler->clear_beam_search_info(request->get_request_id());
            m_requests.pop_back();
        }
    }

    std::vector<ContinuousBatchingPipeline::GeneratedSequence> get_generated_sequences() {
        pull_awaiting_requests();
        std::vector<ContinuousBatchingPipeline::GeneratedSequence> result;
        for (const auto& request : m_requests) {
            const auto request_id = request->get_request_id();
            for (const auto& sequence : request->get_sequences()) {
                auto generated_ids = sequence->get_generated_ids();
                auto log_probs = sequence->get_log_probs();
                result.emplace_back(request_id, sequence->get_grouped_id(), generated_ids, log_probs);
            }
        }
        return result;

    }

    ContinuousBatchingPipeline::UpdateSeqResult
    update_generated_sequence(const ContinuousBatchingPipeline::GeneratedSequence& candidate_sequence) {
        pull_awaiting_requests();
        bool is_empty_generated_tokens = false;
        for (auto& request : m_requests) {
            if (candidate_sequence.request_id == request->get_request_id()) {
                bool is_seq_exists = false;
                // todo: iefode: multiseq
                size_t to_remove_tokens = 0, to_insert_tokens = 0;
                for (auto& sequence : request->get_sequences()) {
                    if (candidate_sequence.sequence_id == sequence->get_grouped_id()) {
                        is_seq_exists = true;
                        auto present_ids = sequence->get_generated_ids();
                        const auto& candidate_ids = candidate_sequence.token_ids;

                        // remove extra tokens from sequence
                        {
                            auto token_idx = std::min(present_ids.size(), candidate_ids.size());
                            if (token_idx) {
                                while (token_idx-- > 0) {
                                    if (present_ids[token_idx] == candidate_ids[token_idx]) {
                                        break;
                                    }
                                }
                                to_remove_tokens = present_ids.size() - (token_idx + 1);
                                if (to_remove_tokens > 0) {
                                    const auto gen_ids_before = sequence->get_generated_ids();
                                    sequence->remove_last_n_tokens(to_remove_tokens);
                                    present_ids = sequence->get_generated_ids();
                                    const size_t gen_len_before = gen_ids_before.size(),
                                                 gen_len_after = present_ids.size();
                                    if (gen_len_after == 0) {
                                        is_empty_generated_tokens = true;
                                    }
                                    OPENVINO_ASSERT(gen_len_after < gen_len_before);
                                    for (size_t i = gen_len_after; i < gen_len_before; ++i) {
                                        m_sampler->update_logit_processor(request->get_request_id(), gen_ids_before[i]);
                                    }
                                }
                            }
                        }
                        // insert new tokens to sequence
                        {
                            OPENVINO_ASSERT(candidate_ids.size() >= present_ids.size());
                            const auto& candidate_log_probs = candidate_sequence.log_probs;
                            const size_t start_id = std::min(present_ids.size(), candidate_ids.size()),
                                         stop_id = std::max(present_ids.size(), candidate_ids.size());
                            to_insert_tokens = stop_id - start_id;
                            for (size_t i = start_id; i < stop_id; ++i) {
                                sequence->append_token(candidate_ids[i],  i < candidate_log_probs.size() ? candidate_log_probs[i] : 0.f);
                            }
                        }
                    }
                    break;
                }
                if (!is_seq_exists) {
                    Sequence::Ptr new_sequence(new Sequence(candidate_sequence.sequence_id));
                    const auto& generated_tokens = candidate_sequence.token_ids;
                    const auto& generated_log_probs = candidate_sequence.log_probs;
                    for (size_t i = 0; i < generated_tokens.size(); ++i) {
                        new_sequence->append_token(generated_tokens[i], generated_log_probs[i]);
                    }
                    request->add_sequence(new_sequence);
                }
                if (!is_empty_generated_tokens) {
                    // in case of non-prompt we need to take prev tokens + token to validate
                    if (request->get_num_processed_tokens())
                        ++to_insert_tokens;
                    if (to_remove_tokens > 0) {
                        request->decrease_processed_tokens(to_remove_tokens);
                    }
                    // to validate tokens/extend kv-cache before generation
                    request->set_validation_len(to_insert_tokens);
                } else if (to_remove_tokens > 0) {
                    request->update_processed_tokens_num(request->get_prompt_len());
                }
                return ContinuousBatchingPipeline::UpdateSeqResult(to_insert_tokens, to_remove_tokens);
            }
        }
    }
};

ContinuousBatchingPipeline::ContinuousBatchingPipeline( const std::string& models_path,
                                                        const SchedulerConfig& scheduler_config,
                                                        const std::string& device,
                                                        const ov::AnyMap& llm_plugin_config,
                                                        const ov::AnyMap& tokenizer_plugin_config) {
    m_impl = std::make_shared<Impl>(models_path, scheduler_config, device, llm_plugin_config, tokenizer_plugin_config);
}

ContinuousBatchingPipeline::ContinuousBatchingPipeline(
    const std::string& model_path,
    const Tokenizer& tokenizer,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& plugin_config
) : m_impl{std::make_shared<Impl>(model_path, tokenizer, scheduler_config, device, plugin_config)} {}

ContinuousBatchingPipeline::ContinuousBatchingPipeline(
    ov::Core& core,
    const std::shared_ptr<ov::Model>& model,
    const ov::genai::Tokenizer& tokenizer,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& plugin_config,
    bool is_enable_validation_mode
) : m_impl{std::make_shared<Impl>(core, model, tokenizer, scheduler_config, device, plugin_config, is_enable_validation_mode)} {}

ov::genai::Tokenizer ContinuousBatchingPipeline::get_tokenizer() {
    return m_impl->get_tokenizer();
}

ov::genai::GenerationConfig ContinuousBatchingPipeline::get_config() const{
    return m_impl->get_config();
}

PipelineMetrics ContinuousBatchingPipeline::get_metrics() const{
    return m_impl->get_metrics();
}

GenerationHandle ContinuousBatchingPipeline::add_request(uint64_t request_id, const std::string& prompt, const ov::genai::GenerationConfig& sampling_params) {
    return m_impl->add_request(request_id, prompt, sampling_params);
}

GenerationHandle ContinuousBatchingPipeline::add_request(uint64_t request_id, const ov::Tensor& input_ids, const ov::genai::GenerationConfig& sampling_params) {
    return m_impl->add_request(request_id, input_ids, sampling_params);
}

void ContinuousBatchingPipeline::step() {
    m_impl->step();
}

bool ContinuousBatchingPipeline::has_non_finished_requests() {
    return m_impl->has_non_finished_requests();
}

std::vector<EncodedGenerationResult> ContinuousBatchingPipeline::generate(const std::vector<ov::Tensor>& input_ids, const std::vector<ov::genai::GenerationConfig>& sampling_params, const StreamerVariant& streamer) {
    return m_impl->generate(input_ids, sampling_params, streamer);
}

std::vector<GenerationResult> ContinuousBatchingPipeline::generate(const std::vector<std::string>& prompts, const std::vector<ov::genai::GenerationConfig>& sampling_params, const StreamerVariant& streamer) {
    return m_impl->generate(prompts, sampling_params, streamer);
}

void ContinuousBatchingPipeline::start_chat(const std::string& system_message) {
    m_impl->start_chat(system_message);
};

void ContinuousBatchingPipeline::finish_chat() {
    m_impl->finish_chat();
};
void ContinuousBatchingPipeline::finish_all_requests() {
    m_impl->finish_all_requests();
}

std::vector<ContinuousBatchingPipeline::GeneratedSequence> 
ContinuousBatchingPipeline::get_generated_sequences() {
    return m_impl->get_generated_sequences();
}

ContinuousBatchingPipeline::UpdateSeqResult
ContinuousBatchingPipeline::update_generated_sequence(const ContinuousBatchingPipeline::GeneratedSequence& new_sequence) {
    return m_impl->update_generated_sequence(new_sequence);
}
