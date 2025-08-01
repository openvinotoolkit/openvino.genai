// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visual_language/pipeline_base.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"

using namespace ov::genai;

class ov::genai::VLMPipeline::VLMContinuousBatchingAdapter : public ov::genai::VLMPipeline::VLMPipelineBase {
public:
    ContinuousBatchingPipeline m_impl;
    
    // CDPruner configuration storage
    ov::AnyMap m_cdpruner_config;

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
        // Initialize CDPruner configuration with default values
        m_cdpruner_config = {
            {"num_visual_tokens", static_cast<size_t>(64)},
            {"relevance_weight", 0.5f},
            {"enable_pruning", true}
        };
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
        // Initialize CDPruner configuration with default values
        m_cdpruner_config = {
            {"num_visual_tokens", static_cast<size_t>(64)},
            {"relevance_weight", 0.5f},
            {"enable_pruning", true}
        };
    }

    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& rgbs,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        auto start_time = std::chrono::steady_clock::now();
        
        // Set CDPruner configuration in the ContinuousBatchingPipeline
        // Add text prompt to the configuration for CDPruner
        ov::AnyMap vision_config = m_cdpruner_config;
        vision_config["text_prompt"] = prompt;
        
        // Pass CDPruner configuration to the underlying pipeline
        m_impl.set_visual_token_pruning_config(vision_config);
        
        auto result = m_impl.generate({prompt}, {rgbs}, {generation_config}, streamer)[0];
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

    virtual void set_visual_token_pruning_config(
        size_t num_visual_tokens,
        float relevance_weight,
        bool enable_pruning
    ) override {
        // Validate input parameters
        OPENVINO_ASSERT(num_visual_tokens > 0 && num_visual_tokens <= 1024,
            "num_visual_tokens must be between 1 and 1024, got: ", num_visual_tokens);
        OPENVINO_ASSERT(relevance_weight >= 0.0f && relevance_weight <= 1.0f,
            "relevance_weight must be between 0.0 and 1.0, got: ", relevance_weight);

        // Update configuration
        m_cdpruner_config["num_visual_tokens"] = num_visual_tokens;
        m_cdpruner_config["relevance_weight"] = relevance_weight;
        m_cdpruner_config["enable_pruning"] = enable_pruning;
    }

    virtual ov::AnyMap get_visual_token_pruning_config() const override {
        return m_cdpruner_config;
    }

    virtual void set_visual_token_pruning_enabled(bool enable) override {
        m_cdpruner_config["enable_pruning"] = enable;
    }

    virtual bool is_visual_token_pruning_enabled() const override {
        auto it = m_cdpruner_config.find("enable_pruning");
        if (it != m_cdpruner_config.end()) {
            try {
                return it->second.as<bool>();
            } catch (const std::exception&) {
                return true; // default value
            }
        }
        return true; // default value
    }
};
