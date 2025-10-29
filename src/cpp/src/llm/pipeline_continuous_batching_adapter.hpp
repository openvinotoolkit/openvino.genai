// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "llm/pipeline_base.hpp"

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include <memory>

namespace ov::genai {

Tokenizer dont_construct() {
    OPENVINO_THROW("Continuous Batching backend can't be constructed"
        "from ireq because the model must be transformed");
}

template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

class ContinuousBatchingAdapter final : public LLMPipelineImplBase {
    std::unique_ptr<ContinuousBatchingPipeline> m_impl;
public:
    ContinuousBatchingAdapter(
        const ov::InferRequest& request,
        const Tokenizer& tokenizer,
        OptionalGenerationConfig generation_config
        ): LLMPipelineImplBase{dont_construct(), GenerationConfig{}},
        m_impl{std::make_unique<ContinuousBatchingPipeline>(std::filesystem::path{}, SchedulerConfig{}, std::string{})} { }
        
    ContinuousBatchingAdapter(
        const std::filesystem::path& models_path,
        const Tokenizer& tokenizer,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        const ov::AnyMap& plugin_config
        ): LLMPipelineImplBase{tokenizer, GenerationConfig()} {
        auto mutable_plugin_config = plugin_config;
        mutable_plugin_config["sampler_num_threads"] = 1;
        m_impl = std::make_unique<ContinuousBatchingPipeline>(models_path, tokenizer, scheduler_config, device, mutable_plugin_config);
        m_generation_config = m_impl->get_config();
        }

    ContinuousBatchingAdapter(
        const std::string& model_str,
        const ov::Tensor& weights_tensor,
        const Tokenizer& tokenizer,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        const ov::AnyMap& plugin_config,
        const ov::genai::GenerationConfig& generation_config
    ): LLMPipelineImplBase{tokenizer, GenerationConfig()} {
        auto mutable_plugin_config = plugin_config;
        mutable_plugin_config["sampler_num_threads"] = 1;
        m_impl = std::make_unique<ContinuousBatchingPipeline>(model_str, weights_tensor, tokenizer, scheduler_config, device, mutable_plugin_config, generation_config);
    }

    ContinuousBatchingAdapter(
        const std::filesystem::path& models_path,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        const ov::AnyMap& plugin_config
    ): LLMPipelineImplBase{Tokenizer(models_path, plugin_config), GenerationConfig()} {
        auto mutable_plugin_config = plugin_config;
        mutable_plugin_config["sampler_num_threads"] = 1;
        m_impl = std::make_unique<ContinuousBatchingPipeline>(models_path, m_tokenizer, scheduler_config, device, mutable_plugin_config);
        m_generation_config = m_impl->get_config();
    }

    DecodedResults generate(
        StringInputs inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) override {
        // Get the currrent timestamp in order to evaluate total generate duration.
        auto start_time =  std::chrono::steady_clock::now();
        
        std::vector<std::string> prompts = std::visit(overloaded{
            [](const std::string& prompt) {
                return std::vector{prompt};
            },
            [](std::vector<std::string>& prompts) {
                return prompts;
            }
        }, inputs);
        const GenerationConfig& config = generation_config.value_or(m_generation_config);
        // -1 == config.eos_token_id and config.validate() are handled in m_impl.
        std::vector<GenerationResult> generated = m_impl->generate(prompts,
            std::vector<GenerationConfig>{prompts.size(), config},
            streamer
        );
        std::vector<std::string> plain_replies;
        std::vector<float> plain_scores;
        for (GenerationResult& res : generated) {
            OPENVINO_ASSERT(res.m_status == GenerationStatus::FINISHED || res.m_status == GenerationStatus::STOP || res.m_status == GenerationStatus::CANCEL, "Got unfinished GenerationStatus");
            std::move(res.m_generation_ids.begin(), res.m_generation_ids.end(), std::back_inserter(plain_replies));
            std::move(res.m_scores.begin(), res.m_scores.end(), std::back_inserter(plain_scores));
        }

        PerfMetrics perf_metrics;
        // For GenerationResults, all perf_metrics are the same except tokenization and detokenization durations.
        // Since we return here only one perf_metrics, we should accumulate all tokenization and detokenization times.
        OPENVINO_ASSERT(!generated.empty());
        perf_metrics = generated[0].perf_metrics;
        perf_metrics.load_time = m_load_time_ms;

        // Tokenizations and detokenization times are dispersed across GenerationResult vector.
        // Need to collect them into a single perf_metric for DecodedResult.
        auto& raw_metrics = perf_metrics.raw_metrics;
        for (size_t i = 1; i < generated.size(); ++i){
            auto tok_durations = generated[i].perf_metrics.raw_metrics.tokenization_durations;
            auto detok_durations = generated[i].perf_metrics.raw_metrics.detokenization_durations;
            for (size_t j = 0; j < tok_durations.size(); ++j) {
                raw_metrics.tokenization_durations.emplace_back(tok_durations[j]);
            }
            for (size_t j = 0; j < detok_durations.size(); ++j) {
                raw_metrics.detokenization_durations.emplace_back(detok_durations[j]);
            }
        }

        raw_metrics.generate_durations.clear();
        raw_metrics.generate_durations.emplace_back(PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start_time));
        // Need to reevaluate statistics with the updated start_time which includes tokenization/detokenization durations.
        perf_metrics.m_evaluated = false;
        perf_metrics.evaluate_statistics(start_time);

        return {std::move(plain_replies), std::move(plain_scores), std::move(perf_metrics), generated[0].extended_perf_metrics};
    }

    DecodedResults generate(
        const ChatHistory& history,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) override {
        auto start_time =  std::chrono::steady_clock::now();
        const GenerationConfig& config = generation_config.value_or(m_generation_config);
        std::vector<ChatHistory> histories = std::vector{history};
        std::vector<GenerationResult> generated = m_impl->generate(histories,
            std::vector<GenerationConfig>{histories.size(), config},
            streamer
        );
        // TODO Consider moving to method and reuse
        std::vector<std::string> plain_replies;
        std::vector<float> plain_scores;
        for (GenerationResult& res : generated) {
            OPENVINO_ASSERT(res.m_status == GenerationStatus::FINISHED || res.m_status == GenerationStatus::STOP || res.m_status == GenerationStatus::CANCEL, "Got unfinished GenerationStatus");
            std::move(res.m_generation_ids.begin(), res.m_generation_ids.end(), std::back_inserter(plain_replies));
            std::move(res.m_scores.begin(), res.m_scores.end(), std::back_inserter(plain_scores));
        }

        PerfMetrics perf_metrics;
        // For GenerationResults, all perf_metrics are the same except tokenization and detokenization durations.
        // Since we return here only one perf_metrics, we should accumulate all tokenization and detokenization times.
        OPENVINO_ASSERT(!generated.empty());
        perf_metrics = generated[0].perf_metrics;
        perf_metrics.load_time = m_load_time_ms;

        // Tokenizations and detokenization times are dispersed across GenerationResult vector.
        // Need to collect them into a single perf_metric for DecodedResult.
        auto& raw_metrics = perf_metrics.raw_metrics;
        for (size_t i = 1; i < generated.size(); ++i){
            auto tok_durations = generated[i].perf_metrics.raw_metrics.tokenization_durations;
            auto detok_durations = generated[i].perf_metrics.raw_metrics.detokenization_durations;
            for (size_t j = 0; j < tok_durations.size(); ++j) {
                raw_metrics.tokenization_durations.emplace_back(tok_durations[j]);
            }
            for (size_t j = 0; j < detok_durations.size(); ++j) {
                raw_metrics.detokenization_durations.emplace_back(detok_durations[j]);
            }
        }

        raw_metrics.generate_durations.clear();
        raw_metrics.generate_durations.emplace_back(PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start_time));
        // Need to reevaluate statistics with the updated start_time which includes tokenization/detokenization durations.
        perf_metrics.m_evaluated = false;
        perf_metrics.evaluate_statistics(start_time);

        return {std::move(plain_replies), std::move(plain_scores), std::move(perf_metrics), generated[0].extended_perf_metrics};
    }

    EncodedResults generate(
        const EncodedInputs& inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) override {
        // Get the currrent timestamp in order to evaluate total generate duration.
        auto start_time =  std::chrono::steady_clock::now();

        std::vector<ov::Tensor> input_ids = std::visit(overloaded{
            [](const ov::Tensor& inp) {
                size_t batch_size = inp.get_shape().at(0);
                if (1 == batch_size) {
                    return std::vector{inp};
                }
                std::vector<ov::Tensor> input_ids;
                input_ids.reserve(batch_size);
                size_t max_len = inp.get_shape().at(1);
                const int64_t* const source = inp.data<const int64_t>();
                for (size_t batch_id = 0; batch_id < batch_size; ++batch_id) {
                    input_ids.emplace_back(ov::element::i64, ov::Shape(1, max_len));
                    int64_t* destination = input_ids.back().data<int64_t>();
                    std::copy_n(source + batch_id * max_len, max_len, destination);
                }
                return input_ids;
            },
            [](const TokenizedInputs& inp) {
                size_t batch_size = inp.input_ids.get_shape().at(0);
                std::vector<ov::Tensor> input_ids;
                input_ids.reserve(batch_size);
                size_t max_len = inp.input_ids.get_shape().at(1);
                OPENVINO_SUPPRESS_DEPRECATED_START
                const int64_t* const source = inp.input_ids.data<const int64_t>();
                const int64_t* const attention_mask = inp.attention_mask.data<const int64_t>();
                OPENVINO_SUPPRESS_DEPRECATED_END
                for (size_t batch_id = 0; batch_id < batch_size; ++batch_id) {
                    input_ids.emplace_back(ov::element::i64, ov::Shape(1, max_len));
                    int64_t* destination = input_ids.back().data<int64_t>();
                    size_t copy_count = 0;
                    for (size_t idx = 0; idx < max_len; ++idx) {
                        if (1 == attention_mask[batch_id * max_len + idx]) {
                            destination[copy_count++] = source[batch_id * max_len + idx];
                        }
                    }
                    input_ids.back().set_shape({1, copy_count});
                }
                return input_ids;
            }
        }, inputs);

        const GenerationConfig& config = generation_config.value_or(m_generation_config);
        // -1 == config.eos_token_id and config.validate() are handled in m_impl.
        std::vector<EncodedGenerationResult> generated = m_impl->generate(input_ids, 
            std::vector<GenerationConfig>{input_ids.size(), config}, 
            streamer
        );
               
        std::vector<std::vector<int64_t>> plain_tokens;
        std::vector<float> plain_scores;
        for (EncodedGenerationResult& res : generated) {
            OPENVINO_ASSERT(res.m_status == GenerationStatus::FINISHED || res.m_status == GenerationStatus::STOP || res.m_status == GenerationStatus::CANCEL, "Got unfinished GenerationStatus");
            std::move(res.m_generation_ids.begin(), res.m_generation_ids.end(), std::back_inserter(plain_tokens));
            std::move(res.m_scores.begin(), res.m_scores.end(), std::back_inserter(plain_scores));
        }
        
        PerfMetrics perf_metrics;
        // For EncodedGenerationResults, all perf_metrics are the same.
        OPENVINO_ASSERT(!generated.empty());
        perf_metrics = generated[0].perf_metrics;
        perf_metrics.load_time = m_load_time_ms;

        auto& raw_counters = perf_metrics.raw_metrics;
        raw_counters.generate_durations.clear();
        raw_counters.generate_durations.emplace_back(PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start_time));
        // Reevaluate statistics with the new start_time which includes time for preparing inputs
        perf_metrics.m_evaluated = false;
        perf_metrics.evaluate_statistics(start_time);

        return {std::move(plain_tokens), std::move(plain_scores), std::move(perf_metrics), generated[0].extended_perf_metrics};
    }

    void start_chat(const std::string& system_message) override {
        m_impl->start_chat(system_message);
    }

    void finish_chat() override {
        m_impl->finish_chat();
    }
};

} // namespace ov::genai
