// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visual_language/pipeline_base.hpp"
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
        properties} { }

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
    }

    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        return generate(prompt, images, {}, generation_config, streamer);
    }

    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& images,
        const std::vector<ov::Tensor>& video,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        auto start_time = std::chrono::steady_clock::now();
        auto result = m_impl.generate({prompt}, {images}, {video}, {generation_config}, streamer)[0];
        auto stop_time = std::chrono::steady_clock::now();
        
        VLMDecodedResults decoded;
        decoded.perf_metrics = result.perf_metrics;
        decoded.perf_metrics.load_time = get_load_time();

        decoded.perf_metrics.raw_metrics.generate_durations.clear();
        decoded.perf_metrics.raw_metrics.generate_durations.emplace_back(PerfMetrics::get_microsec(stop_time - start_time));
        decoded.perf_metrics.m_evaluated = false;
        decoded.perf_metrics.evaluate_statistics(start_time);
        
        for (size_t idx = 0; idx < result.texts.size(); ++idx) {
            decoded.texts.push_back(result.texts.at(idx));
            decoded.scores.push_back(result.scores.at(idx));
        }
        return decoded;
    }

    virtual void start_chat(const std::string& system_message) override { m_impl.start_chat(system_message); };

    virtual void finish_chat() override { m_impl.finish_chat(); };

    virtual Tokenizer get_tokenizer() const override { return m_impl.get_tokenizer(); };

    virtual void set_chat_template(const std::string& new_template) override { OPENVINO_THROW("Chat mode is not supported."); };

    virtual GenerationConfig get_generation_config() const override { return m_impl.get_config(); };

    virtual void set_generation_config(const GenerationConfig& new_config)  override { m_impl.set_config(new_config); };
};
