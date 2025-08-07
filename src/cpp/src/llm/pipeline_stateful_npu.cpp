
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pipeline_stateful_npu.hpp"
#include "speculative_decoding/speculative_decoding_npu.hpp"
#include "llm/pipeline_stateful.hpp"
#include "llm/pipeline_static.hpp"
#include "utils.hpp"

#include <fstream>

#include "openvino/runtime/core.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/genai/text_streamer.hpp"

namespace {
    ov::genai::ModelDesc
    extract_draft_model_from_config(ov::AnyMap& config) {
        ov::genai::ModelDesc draft_model;
        if (config.find(ov::genai::utils::DRAFT_MODEL_ARG_NAME) != config.end()) {
            draft_model = config.at(ov::genai::utils::DRAFT_MODEL_ARG_NAME).as<ov::genai::ModelDesc>();
            config.erase(ov::genai::utils::DRAFT_MODEL_ARG_NAME);
        }
        return draft_model;
}
} // anonymous namespace

namespace ov::genai {

// NB: No constructor for creation of pipeline from infer request, as pipeline from infer request
//     for NPU is handled inside of ov::genai::StatefulLLMPipeline class iself.
StatefulLLMPipelineNPU::StatefulLLMPipelineNPU(
    const std::filesystem::path& models_path,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap& properties)
    : StatefulLLMPipelineNPU(
        utils::read_model(models_path, properties),
        tokenizer,
        device,
        properties,
        utils::from_config_json_if_exists(models_path)
    ) {}

StatefulLLMPipelineNPU::StatefulLLMPipelineNPU(
    const std::filesystem::path& models_path,
    const std::string& device,
    const ov::AnyMap& plugin_config)
    : StatefulLLMPipelineNPU{models_path, Tokenizer(models_path, plugin_config), device, plugin_config} {}

StatefulLLMPipelineNPU::StatefulLLMPipelineNPU(
    const std::shared_ptr<ov::Model>& model,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap& properties,
    const ov::genai::GenerationConfig& generation_config)
    : LLMPipelineImplBase(tokenizer, generation_config) {
    auto properties_without_draft_model = properties;
    auto draft_model_descr = extract_draft_model_from_config(properties_without_draft_model);
     if (draft_model_descr.model != nullptr) {
        auto main_model_descr = ov::genai::ModelDesc(model, tokenizer, device, properties_without_draft_model, {}, generation_config);
        m_pimpl = std::make_unique<SpeculativeLLMPipelineNPU>(main_model_descr, draft_model_descr);
    } else if (properties_without_draft_model.count("STATIC_PIPELINE")) {
        m_pimpl = static_llm::LLMPipelineFactory::create(model, tokenizer,
            properties_without_draft_model, generation_config);
    } else {
        m_pimpl = std::make_unique<StatefulLLMPipeline>(model, tokenizer, "NPU",
            properties_without_draft_model, generation_config);
    }
}

DecodedResults StatefulLLMPipelineNPU::generate(
    StringInputs inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer) {
        return m_pimpl->generate(inputs, generation_config, streamer);
}

EncodedResults StatefulLLMPipelineNPU::generate(
    const EncodedInputs& inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer) {
        return m_pimpl->generate(inputs, generation_config, streamer);
}

void StatefulLLMPipelineNPU::start_chat(const std::string& system_message) {
    m_pimpl->start_chat(system_message);
}

// FIXME: Do we need it?
// void StatefulLLMPipelineNPU::reset_kv_state() {
//     m_pimpl->reset_kv_state();
// }

void StatefulLLMPipelineNPU::finish_chat() {
    m_pimpl->finish_chat();
}

} // namespace ov::genai
