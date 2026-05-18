// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visual_language/pipeline_base.hpp"
#include "visual_language/chat_history_state.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"

using namespace ov::genai;

class ov::genai::VLMPipeline::VLMContinuousBatchingAdapter : public ov::genai::VLMPipeline::VLMPipelineBase {
public:
    ContinuousBatchingPipeline m_impl;

    VLMContinuousBatchingAdapter(
        const std::filesystem::path& models_dir,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        const ov::AnyMap& properties
    ): m_impl{
        models_dir, 
        scheduler_config, 
        device, 
        properties} {
        set_attention_backend(PA_BACKEND);
    }

    VLMContinuousBatchingAdapter(
        const std::shared_ptr<ov::Model>& language_model,
        const std::filesystem::path& models_dir,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        const ov::AnyMap& properties
    ): m_impl{
        language_model,
        models_dir,
        scheduler_config,
        device,
        properties} {
        set_attention_backend(PA_BACKEND);
    }

    VLMContinuousBatchingAdapter(
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        const ov::AnyMap& properties,
        const ov::genai::GenerationConfig& generation_config
    ): m_impl{
        models_map,
        tokenizer,
        scheduler_config,
        device,
        config_dir_path,
        properties,
        generation_config} {
        set_attention_backend(PA_BACKEND);
    }

    VLMContinuousBatchingAdapter(
        const std::shared_ptr<ov::Model>& language_model,
        const ModelsMap& models_map,
        const Tokenizer& tokenizer,
        const std::filesystem::path& config_dir_path,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        const ov::AnyMap& properties,
        const ov::genai::GenerationConfig& generation_config
    ): m_impl{
        language_model,
        models_map,
        tokenizer,
        scheduler_config,
        device,
        config_dir_path,
        properties,
        generation_config} {
        set_attention_backend(PA_BACKEND);
    }

    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        return generate(prompt, images, {}, std::move(generation_config), streamer);
    }

    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        return generate(prompt, images, videos, {}, std::move(generation_config), streamer);
    }

    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const std::vector<VideoMetadata>& videos_metadata,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        auto start_time = std::chrono::steady_clock::now();
        std::vector<ov::genai::GenerationConfig> generation_configs = {std::move(generation_config)};
        const auto decoded_results = m_impl.generate(
            {prompt},
            ov::genai::images_batches({images}),
            ov::genai::videos_batches({videos}),
            ov::genai::videos_metadata_batches({videos_metadata}),
            ov::genai::generation_config_batches(generation_configs),
            ov::genai::streamer(streamer)
        )[0];
        auto stop_time = std::chrono::steady_clock::now();
        return finalize_decoded_results(decoded_results, start_time, stop_time);
    }

    VLMDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        return generate(history, images, {}, std::move(generation_config), streamer);
    }

    VLMDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        return generate(history, images, videos, {}, std::move(generation_config), streamer);
    }

    VLMDecodedResults generate(
        const ChatHistory& history,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& videos,
        const std::vector<VideoMetadata>& videos_metadata,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        auto start_time = std::chrono::steady_clock::now();
        // Ensure chat history internal state is initialized for original history
        ChatHistoryInternalState::get_or_create(history);
        std::vector<ov::genai::GenerationConfig> generation_configs = {std::move(generation_config)};
        const auto decoded_results = m_impl.generate(
            {history},
            ov::genai::images_batches({images}),
            ov::genai::videos_batches({videos}),
            ov::genai::videos_metadata_batches({videos_metadata}),
            ov::genai::generation_config_batches(generation_configs),
            ov::genai::streamer(streamer)
        )[0];
        auto stop_time = std::chrono::steady_clock::now();
        return finalize_decoded_results(decoded_results, start_time, stop_time);
    }

    virtual void start_chat(const std::string& system_message) override { m_impl.start_chat(system_message); };

    virtual void finish_chat() override { m_impl.finish_chat(); };

    virtual Tokenizer get_tokenizer() const override { return m_impl.get_tokenizer(); };

    virtual void set_chat_template(const std::string& new_template) override { OPENVINO_THROW("Chat mode is not supported."); };

    virtual GenerationConfig get_generation_config() const override { return m_impl.get_config(); };

    virtual void set_generation_config(const GenerationConfig& new_config)  override { m_impl.set_config(new_config); };

private:
    VLMDecodedResults finalize_decoded_results(
        const VLMDecodedResults& decoded_results,
        std::chrono::steady_clock::time_point start_time,
        std::chrono::steady_clock::time_point stop_time
    ) {
        VLMDecodedResults final_decoded_results;
        final_decoded_results.perf_metrics = decoded_results.perf_metrics;
        final_decoded_results.perf_metrics.load_time = get_load_time();
        final_decoded_results.perf_metrics.raw_metrics.generate_durations.clear();
        final_decoded_results.perf_metrics.raw_metrics.generate_durations.emplace_back(
            PerfMetrics::get_microsec(stop_time - start_time)
        );
        final_decoded_results.perf_metrics.m_evaluated = false;
        final_decoded_results.perf_metrics.evaluate_statistics(start_time);
        final_decoded_results.texts = decoded_results.texts;
        final_decoded_results.scores = decoded_results.scores;
        final_decoded_results.finish_reasons = decoded_results.finish_reasons;
        return final_decoded_results;
    }
};
