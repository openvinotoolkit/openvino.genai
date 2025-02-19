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
        "./", 
        scheduler_config, 
        device, 
        properties} {
        OPENVINO_THROW("Not implemented.");
    }

    VLMDecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& rgbs,
        GenerationConfig generation_config,
        const StreamerVariant& streamer
    ) override {
        auto result = m_impl.generate({prompt}, {rgbs}, {generation_config}, streamer)[0];
        VLMDecodedResults decoded;
        for (size_t idx = 0; idx < result.m_generation_ids.size(); ++idx) {
            decoded.texts.push_back(result.m_generation_ids.at(idx));
            decoded.scores.push_back(result.m_scores.at(idx));
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
