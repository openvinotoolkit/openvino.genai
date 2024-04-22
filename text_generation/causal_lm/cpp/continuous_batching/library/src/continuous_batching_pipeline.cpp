// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching_pipeline.hpp"
#include "cache_manager.hpp"
#include "sampler.hpp"
#include "model_runner.hpp"
#include "scheduler.hpp"
#include "timer.hpp"
#include "model_config.hpp"
#include "model_config.hpp"
#include "tokenizer.hpp"
#include "paged_attention.hpp"

#include "debug_utils.hpp"

namespace {

GenerationResult from_sequence_group(std::shared_ptr<Tokenizer> tokenizer, SequenceGroup::CPtr sequence_group) {
    GenerationResult result;
    result.m_request_id = sequence_group->get_request_id();

    OPENVINO_ASSERT(sequence_group->num_finished_seqs() == sequence_group->num_total_seqs() &&
                    sequence_group->has_finished());
    for (size_t sequence_id = 0; sequence_id < sequence_group->num_finished_seqs(); ++sequence_id) {
        Sequence::CPtr sequence = (*sequence_group)[sequence_id];

        // TODO: they are not correct in case of beam search at least
        // we need to pass beam score instead of cumulative log probs (e.g. normalized by length)
        result.m_scores.push_back(sequence->get_cumulative_log_probs());

        std::string output_text = tokenizer->decode(sequence->get_generated_ids());
        result.m_generation_ids.push_back(output_text);
    }

    return result;
}

} // namespace

void apply_paged_attention_transformations(std::shared_ptr<ov::Model> model,
                                           const ModelConfig& model_config, const DeviceConfig& device_config);

class ContinuousBatchingPipeline::Impl {
    std::shared_ptr<Tokenizer> m_tokenizer;
    std::shared_ptr<Scheduler> m_scheduler;
    std::shared_ptr<CacheManager> m_cache_manager;
    std::shared_ptr<ModelRunner> m_model_runner;
    std::shared_ptr<Sampler> m_sampler;

    GenerationConfig m_generation_config;

    // current requests to process
    std::vector<SequenceGroup::Ptr> m_requests;

    void _free_finished_requests() {
        auto new_end = std::remove_if(m_requests.begin(), m_requests.end(), [] (SequenceGroup::CPtr seq_group) -> bool {
            return seq_group->has_finished();
        });
        m_requests.erase(new_end, m_requests.end());
    }

public:
    Impl(const std::string& models_path, const SchedulerConfig& scheduler_config) {
        ov::Core core;
        core.add_extension<PagedAttention>();
        m_tokenizer = std::make_shared<Tokenizer>(models_path);

        // The model can be compiled for GPU as well
        std::shared_ptr<ov::Model> model = core.read_model(models_path + "/openvino_model.xml");
        ModelConfig model_config(model);

        const std::string device = "CPU"; 
        DeviceConfig device_config(core, scheduler_config, model_config, device);

        apply_paged_attention_transformations(model, model_config, device_config);
        ov::InferRequest infer_request = core.compile_model(model, device_config.get_device()).create_infer_request();

        // setup KV caches
        m_cache_manager = std::make_shared<CacheManager>(model_config, device_config);
        for (size_t decoder_layer_id = 0; decoder_layer_id < model_config.get_num_layers(); ++decoder_layer_id) {
            infer_request.set_input_tensor(2 + decoder_layer_id * 2, m_cache_manager->get_key_cache(decoder_layer_id));
            infer_request.set_input_tensor(2 + decoder_layer_id * 2 + 1, m_cache_manager->get_value_cache(decoder_layer_id));
        }

        m_scheduler = std::make_shared<Scheduler>(scheduler_config);
        // and finally create model runner
        m_model_runner = std::make_shared<ModelRunner>(infer_request, scheduler_config);
        m_sampler = std::make_shared<Sampler>();

        // read default generation config
    }

    GenerationConfig get_config() const {
        return m_generation_config;
    }

    std::shared_ptr<Tokenizer> get_tokenizer() {
        return m_tokenizer;
    }

    void add_request(uint64_t request_id, std::string prompt, GenerationConfig sampling_params) {
        SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(request_id, m_tokenizer->encode(prompt),
                                                                            sampling_params, m_scheduler->get_config().block_size);
        m_requests.push_back(sequence_group);
    }

    std::vector<GenerationResult> step() {
        static ScopedTimer step_timer("step()");
        step_timer.start();

        Scheduler::Output scheduler_output;
        {
            static ScopedTimer timer("scheduling");
            timer.start();
            scheduler_output = m_scheduler->schedule(m_requests);
            m_cache_manager->copy_blocks(scheduler_output.m_block_copy_map);
            timer.end();
        }

        ov::Tensor logits;
        {
            static ScopedTimer timer("forward");
            timer.start();
            logits = m_model_runner->forward(m_requests, scheduler_output);
            timer.end();
        }

        SamplerOutput sampler_output;
        {
            static ScopedTimer timer("sample");
            timer.start();
            sampler_output = m_sampler->sample(m_requests, logits);
            timer.end();
        }

        // process sampler_output (e.g. fork or drop sequences from BlockScheduler)
        {
            static ScopedTimer timer("fork / free sequence");
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

        // perform post-processing of current step

        std::vector<GenerationResult> currently_finished_requests;
        {
            static ScopedTimer timer("create finished results");
            timer.start();

            for (size_t i = 0; i < scheduler_output.m_scheduled_sequence_groups_ids.size(); ++i) {
                uint64_t seq_group_id = scheduler_output.m_scheduled_sequence_groups_ids[i];
                SequenceGroup::CPtr sequence_group = m_requests[seq_group_id];
                if (sequence_group->has_finished()) {
                   currently_finished_requests.push_back(from_sequence_group(m_tokenizer, sequence_group));
                }
            }

            _free_finished_requests();

            timer.end();
        }

        step_timer.end();
        return currently_finished_requests;
    }

    bool has_running_requests() const {
        return !m_requests.empty();
    }

    std::vector<GenerationResult> generate(const std::vector<std::string> prompts, std::vector<GenerationConfig> sampling_params) {
        OPENVINO_ASSERT(!has_running_requests(), "Generate cannot be called while ContinuousBatchingPipeline is already in running state. Use ContinuousBatchingPipeline::add_request");
        OPENVINO_ASSERT(prompts.size() == sampling_params.size());

        for (size_t request_id = 0; request_id < prompts.size(); ++request_id) {
            add_request(request_id, prompts[request_id], sampling_params[request_id]);
        }

        std::vector<GenerationResult> results;
        results.reserve(m_requests.size());

        while (has_running_requests()) {
            std::vector<GenerationResult> partial_results = step();
            results.insert(results.end(), partial_results.begin(), partial_results.end());
        }

        // sort results according to request_id to return results in order of initial prompts
        std::sort(results.begin(), results.end(), [] (const GenerationResult& r1, const GenerationResult& r2) -> bool {
            return r1.m_request_id < r2.m_request_id;
        });

        OPENVINO_ASSERT(results.size() == prompts.size());
        return results;
    }
};

ContinuousBatchingPipeline::ContinuousBatchingPipeline(const std::string& models_path,
                     const SchedulerConfig& scheduler_config) {
    m_impl = std::make_shared<Impl>(models_path, scheduler_config);
}

std::shared_ptr<Tokenizer> ContinuousBatchingPipeline::get_tokenizer() {
    return m_impl->get_tokenizer();
}

void ContinuousBatchingPipeline::add_request(uint64_t request_id, std::string prompt, GenerationConfig sampling_params) {
    return m_impl->add_request(request_id, prompt, sampling_params);
}

std::vector<GenerationResult> ContinuousBatchingPipeline::step() {
     return m_impl->step();
}

bool ContinuousBatchingPipeline::has_running_requests() const {
    return m_impl->has_running_requests();
}

std::vector<GenerationResult> ContinuousBatchingPipeline::generate(const std::vector<std::string>& prompts, std::vector<GenerationConfig> sampling_params) {
    return m_impl->generate(prompts, sampling_params);
}
