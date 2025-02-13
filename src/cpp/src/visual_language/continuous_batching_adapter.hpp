// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <random>

#include "openvino/genai/visual_language/perf_metrics.hpp"
#include "openvino/genai/tokenizer.hpp"

#include "visual_language/vlm_config.hpp"
#include "visual_language/inputs_embedder.hpp"
#include "visual_language/embedding_model.hpp"
#include "visual_language/pipeline_base.hpp"

#include "sampler.hpp"
#include "text_callback_streamer.hpp"
#include "utils.hpp"
#include "lm_encoding.hpp"

class ov::genai::VLMPipeline::VLMContinuousBatchingAdapter : public ov::genai::VLMPipeline::VLMPipelineBase {
public:
    ContinuousBatchingPipeline m_impl;

    VLMContinuousBatchingAdapter(
        const std::filesystem::path& models_dir,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        const ov::AnyMap& properties
    ): m_impl{
        models_dir / "openvino_language_model.xml", 
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

        auto decode_start_time = std::chrono::steady_clock::now();
        VLMDecodedResults decoded;
        for (size_t idx = 0; idx < result.m_generation_ids.size(); ++idx) {
            decoded.texts.push_back(result.m_generation_ids.at(idx));
            decoded.scores.push_back(result.m_scores.at(idx));
        }
        return decoded;
    }



    virtual void start_chat(const std::string& system_message) override {};

    virtual void finish_chat() override {};

    virtual Tokenizer get_tokenizer() const override { OPENVINO_THROW("Not implemented.");};

    virtual void set_chat_template(const std::string& new_template)  override {};

    virtual GenerationConfig get_generation_config() const  override {OPENVINO_THROW("Not implemented.");};

    virtual void set_generation_config(const GenerationConfig& new_config)  override {};
};
