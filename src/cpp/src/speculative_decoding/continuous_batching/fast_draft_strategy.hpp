// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "continuous_batching/pipeline_impl.hpp"
#include "openvino/genai/speculative_decoding/perf_metrics.hpp"
#include "speculative_decoding/continuous_batching/pipeline_impl.hpp"
#include "speculative_decoding/speculative_decoding_metrics.hpp"
#include "utils.hpp"

namespace ov::genai {
struct GenerateStrategy {
    std::function<void(size_t,
                       const ov::Tensor& in_ids,
                       GenerationConfig& main_cfg,
                       GenerationConfig& draft_cfg,
                       ov::Tensor& main_in,
                       ov::Tensor& draft_in)> prepare_request;
    std::function<void(const std::shared_ptr<ThreadedStreamerWrapper>&,
                       const std::vector<ov::Tensor>&,
                       const std::vector<GenerationConfig>&)> check_streaming;
    std::function<TimePoint()> start_timer;
    std::function<uint64_t(TimePoint)> stop_timer;
};

template<class Impl>
std::vector<EncodedGenerationResult> generate_common(
        Impl* self,
        const std::vector<ov::Tensor>& input_ids,
        const std::vector<GenerationConfig>& sampling_params,
        const StreamerVariant& streamer,
        std::optional<std::vector<ov::Tensor>> token_type_ids,
        GenerateStrategy& strategy) {

    OPENVINO_ASSERT(!token_type_ids.has_value());
    self->perf_metrics() = ov::genai::SDPerModelsPerfMetrics();
    self->draft_pipeline()->raw_perf_metrics.m_inference_durations = {{ MicroSeconds(0.0f) }};

    OPENVINO_ASSERT(!self->has_non_finished_requests(),
                    "Generate cannot be called while ContinuousBatchingPipeline is already running");
    OPENVINO_ASSERT(input_ids.size() == sampling_params.size());

    auto t_start = strategy.start_timer();

    for (size_t i = 1; i < sampling_params.size(); ++i) {
        OPENVINO_ASSERT(sampling_params[i - 1].adapters == sampling_params[i].adapters,
                        "LoRA adapters must be same for all requests");
    }
    self->main_pipeline()->set_adapters(sampling_params[0].adapters);
    self->draft_pipeline()->set_adapters(sampling_params[0].adapters);

    auto streamer_ptr = std::make_shared<ThreadedStreamerWrapper>(streamer, self->tokenizer());

    strategy.check_streaming(streamer_ptr, input_ids, sampling_params);

    std::vector<GenerationHandle> main_generations;
    {
        std::lock_guard<std::mutex> lock(self->draft_generations_mutex());
        for (size_t rid = 0; rid < input_ids.size(); ++rid) {
            GenerationConfig main_cfg = sampling_params[rid];
            GenerationConfig draft_cfg = main_cfg;
            ov::Tensor main_in, draft_in;
            strategy.prepare_request(rid, input_ids[rid],
                                    main_cfg, draft_cfg,
                                    main_in, draft_in);
            main_generations.push_back(self->main_pipeline()->add_request(rid, main_in, main_cfg));
            self->draft_generations().insert({rid,
                self->draft_pipeline()->add_request(rid, draft_in, draft_cfg)});
        }
    }

    auto all_requests = self->get_awaiting_requests();
    GenerationHandle& generation = main_generations.at(0);

    streamer_ptr->start();
    while (self->has_non_finished_requests()) {
        try {
            self->step();
        } catch (...) {
            self->drop_requests();
            streamer_ptr->end();
            std::rethrow_exception(std::current_exception());
        }
        self->stream_tokens(streamer_ptr, generation);
    }
    streamer_ptr->end();

    OPENVINO_ASSERT(self->is_requests_empty(), "Internal error: current request is supposed to be dropped within step() function as completed");

    self->perf_metrics().draft_model_metrics.raw_metrics = self->draft_pipeline()->raw_perf_metrics;
    uint64_t generate_duration_us = strategy.stop_timer(t_start);

    std::vector<EncodedGenerationResult> results;
    results.reserve(all_requests.size());

    for (size_t rid = 0; rid < all_requests.size(); ++rid) {
        const auto& request = all_requests[rid];
        auto cfg = request->get_sampling_parameters();
        const auto& seqs = request->get_finished_sequences();
        size_t num_out = std::min(cfg.num_return_sequences, seqs.size());

        EncodedGenerationResult result;
        result.m_request_id = rid;
        result.m_generation_ids.resize(num_out);
        result.m_scores.resize(num_out);
        result.m_status = main_generations[rid]->get_status();

        for (size_t i = 0; i < num_out; ++i) {
            const auto& seq = seqs[i];
            float score = cfg.is_beam_search() ?
                        seq->get_beam_search_score(cfg) :
                        seq->get_cumulative_log_prob();
            const auto& gen_ids = seq->get_generated_ids();
            if (cfg.echo) {
                result.m_generation_ids[i] = request->get_prompt_ids();
            }
            std::copy(gen_ids.begin(), gen_ids.end(),
                    std::back_inserter(result.m_generation_ids[i]));
            result.m_scores[i] = score;
        }

        self->perf_metrics().raw_metrics.generate_durations.clear();
        self->perf_metrics().raw_metrics.generate_durations.emplace_back(generate_duration_us);
        self->perf_metrics().num_input_tokens = request->get_prompt_len();
        self->perf_metrics().evaluate_statistics(t_start);

        result.perf_metrics = self->perf_metrics();
        result.extended_perf_metrics = std::make_shared<SDPerModelsPerfMetrics>(self->perf_metrics());
        results.push_back(std::move(result));
    }

    OPENVINO_ASSERT(results.size() == input_ids.size());
    return results;
}

class ContinuousBatchingPipeline::SpeculativeDecodingImpl : public ContinuousBatchingPipeline::IContinuousBatchingPipeline {
protected:
    std::shared_ptr<ContinuousBatchingForSpeculativeDecodingImpl> m_main_pipeline, m_draft_pipeline;
    // Metrics
    SpeculativeDecodingMetrics m_sd_metrics;
    ov::genai::SDPerModelsPerfMetrics m_perf_metrics;

    // Mutex protecting access to m_draft_generations, so add_request and step methods can be called from different threads
    std::mutex m_draft_generations_mutex;
    std::map<uint64_t, GenerationHandle> m_draft_generations;

    void drop_requests();
    bool is_requests_empty();
    std::vector<SequenceGroup::Ptr> get_awaiting_requests();
    std::pair<ov::genai::SchedulerConfig, ov::genai::SchedulerConfig> init_speculative_models(const ov::genai::ModelDesc& main_model_desc, const ov::genai::ModelDesc& draft_model_desc);
    std::map<uint64_t, GenerationHandle>& draft_generations() { return m_draft_generations; }
public:
    template<class Impl>
    friend std::vector<EncodedGenerationResult> generate_common(
            Impl* self,
            const std::vector<ov::Tensor>& input_ids,
            const std::vector<GenerationConfig>& sampling_params,
            const StreamerVariant& streamer,
            std::optional<std::vector<ov::Tensor>> token_type_ids,
            GenerateStrategy& strategy);

    SpeculativeDecodingImpl() = default;
    SpeculativeDecodingImpl(const ov::genai::ModelDesc& main_model_desc, const ov::genai::ModelDesc& draft_model_desc);

    GenerationHandle add_request(uint64_t request_id,
                                 const ov::Tensor& input_ids,
                                 const ov::genai::GenerationConfig& sampling_params,
                                 std::optional<ov::Tensor> token_type_ids = std::nullopt) override;
    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 const ov::genai::GenerationConfig& sampling_params) override;

    bool has_non_finished_requests() override;

    void step() override;

    std::vector<EncodedGenerationResult>
    generate(const std::vector<ov::Tensor>& input_ids,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer,
             const std::optional<std::vector<ov::Tensor>>& token_type_ids = std::nullopt,
             const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids = std::nullopt) override;

    SpeculativeDecodingMetrics get_speculative_decoding_metrics();
    SDPerModelsPerfMetrics& perf_metrics() { return m_perf_metrics; }
    SDPerModelsPerfMetrics const& perf_metrics() const { return m_perf_metrics; }
    std::shared_ptr<ContinuousBatchingForSpeculativeDecodingImpl>& draft_pipeline() { return m_draft_pipeline; }
    std::shared_ptr<ContinuousBatchingForSpeculativeDecodingImpl>& main_pipeline() { return m_main_pipeline; }

    Tokenizer& tokenizer() { return m_tokenizer; }
    const Tokenizer& tokenizer() const { return m_tokenizer; }

    std::mutex& draft_generations_mutex() { return m_draft_generations_mutex; }
};

}  // namespace ov::genai
