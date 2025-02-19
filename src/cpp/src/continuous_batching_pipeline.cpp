// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <mutex>
#include <memory>
#include <openvino/runtime/properties.hpp>

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "continuous_batching_impl.hpp"
#include "speculative_decoding/speculative_decoding_impl.hpp"
#include "prompt_lookup/prompt_lookup_impl.hpp"
#include "timer.hpp"
#include "utils.hpp"
#include "debug_utils.hpp"
#include "visual_language/inputs_embedder.hpp"

using namespace ov::genai;

inline ov::genai::ModelDesc
extract_draft_model_from_config(ov::AnyMap& config) {
    ov::genai::ModelDesc draft_model;
    if (config.find(utils::DRAFT_MODEL_ARG_NAME) != config.end()) {
        draft_model = config.at(utils::DRAFT_MODEL_ARG_NAME).as<ov::genai::ModelDesc>();
        config.erase(utils::DRAFT_MODEL_ARG_NAME);
    }
    return draft_model;
}

inline bool
extract_prompt_lookup_from_config(ov::AnyMap& config) {
    bool res = false;
    if (config.find(ov::genai::prompt_lookup.name()) != config.end()) {
        res = config.at(ov::genai::prompt_lookup.name()).as<bool>();
        config.erase(ov::genai::prompt_lookup.name());
    }
    return res;
}

inline float get_load_time(std::chrono::steady_clock::time_point start_time) {
    auto stop_time = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
}

inline bool ends_with(std::string str, std::string postfix) {
    return str.rfind(postfix) == str.size() - postfix.size();
}

std::string get_directory(const std::string& s) {
    // Linux-style separator
    auto pos = s.find_last_of('/');
    if (pos != std::string::npos) {
        return s.substr(0, pos ? pos : 1);
    }
    // Windows-style separator
    pos = s.find_last_of('\\');
    if (pos != std::string::npos) {
        return s.substr(0, pos);
    } else if (s.empty()) {
        return {};
    } else {
        return {'.'};
    }
}

ContinuousBatchingPipeline::ContinuousBatchingPipeline( const std::filesystem::path& models_path,
                                                        const SchedulerConfig& scheduler_config,
                                                        const std::string& device,
                                                        const ov::AnyMap& properties,
                                                        const ov::AnyMap& tokenizer_properties) {
    auto start_time = std::chrono::steady_clock::now();

    auto properties_without_draft_model = properties;
    auto draft_model_desr = extract_draft_model_from_config(properties_without_draft_model);
    auto is_prompt_lookup_enabled = extract_prompt_lookup_from_config(properties_without_draft_model);

    std::filesystem::path model_path = models_path;
    std::filesystem::path directory = models_path;
    if (std::filesystem::exists(model_path / "openvino_model.xml")) {
        model_path = model_path / "openvino_model.xml";
    }
    else if (std::filesystem::exists(model_path / "openvino_language_model.xml")) {
        model_path = model_path / "openvino_language_model.xml";
    }
    else {
        OPENVINO_THROW("Could not find a model in the directory.");
    }

    auto model = utils::singleton_core().read_model(model_path, {}, properties);
    auto tokenizer = ov::genai::Tokenizer(directory, tokenizer_properties);
    auto generation_config = utils::from_config_json_if_exists(directory);

    if (is_prompt_lookup_enabled) {
        OPENVINO_ASSERT(draft_model_desr.model == nullptr, "Speculative decoding and prompt lookup decoding are mutually exclusive");
        m_impl = std::make_shared<PromptLookupImpl>(model, tokenizer, scheduler_config, device, properties_without_draft_model, generation_config);
    } else if (draft_model_desr.model != nullptr) {
        auto main_model_descr = ov::genai::ModelDesc(model, tokenizer, device, properties_without_draft_model, scheduler_config, generation_config);
        m_impl = std::make_shared<SpeculativeDecodingImpl>(main_model_descr, draft_model_desr);
    } else if (std::filesystem::exists(directory / "openvino_text_embeddings_model.xml") ) {
        auto vlm_config = ov::genai::VLMConfig{ utils::from_config_json_if_exists<VLMConfig>(directory, "config.json")};
        auto inputs_embedder = std::make_shared<InputsEmbedder>(vlm_config, directory, device, properties);
        m_impl = std::make_shared<ContinuousBatchingImpl>(model, inputs_embedder, tokenizer, scheduler_config, device, properties, generation_config);
    }
    else {
        m_impl = std::make_shared<ContinuousBatchingImpl>(model, tokenizer, scheduler_config, device, properties, generation_config);
    }

    m_impl->m_load_time_ms = get_load_time(start_time);
}

ContinuousBatchingPipeline::ContinuousBatchingPipeline(
    const std::filesystem::path& models_path,
    const Tokenizer& tokenizer,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& properties) {
    auto start_time = std::chrono::steady_clock::now();

    auto properties_without_draft_model = properties;
    auto draft_model_desr = extract_draft_model_from_config(properties_without_draft_model);
    auto is_prompt_lookup_enabled = extract_prompt_lookup_from_config(properties_without_draft_model);
    std::filesystem::path model_path = models_path;
    std::filesystem::path directory = models_path;
    if (std::filesystem::exists(model_path / "openvino_model.xml")) {
        model_path = model_path / "openvino_model.xml";
    }
    else if (std::filesystem::exists(model_path / "openvino_language_model.xml")) {
        model_path = model_path / "openvino_language_model.xml";
    }
    else {
        OPENVINO_THROW("Could not find a model in the directory.");
    }
    auto model = utils::singleton_core().read_model(model_path, {}, properties_without_draft_model);
    auto generation_config = utils::from_config_json_if_exists(directory);

    if (is_prompt_lookup_enabled) {
        OPENVINO_ASSERT(draft_model_desr.model == nullptr, "Speculative decoding and prompt lookup decoding are mutually exclusive");
        m_impl = std::make_shared<PromptLookupImpl>(model, tokenizer, scheduler_config, device, properties_without_draft_model, generation_config);
    } else if (draft_model_desr.model != nullptr) {
        auto main_model_descr = ov::genai::ModelDesc(model, tokenizer, device, properties_without_draft_model, scheduler_config, generation_config);
        m_impl = std::make_shared<SpeculativeDecodingImpl>(main_model_descr, draft_model_desr);
    } else if (std::filesystem::exists(directory / "openvino_text_embeddings_model.xml") ) {
        auto vlm_config = ov::genai::VLMConfig{ utils::from_config_json_if_exists<VLMConfig>(directory, "config.json")};
        auto inputs_embedder = std::make_shared<InputsEmbedder>(vlm_config, directory, device, properties);
        m_impl = std::make_shared<ContinuousBatchingImpl>(model, inputs_embedder, tokenizer, scheduler_config, device, properties, generation_config);
    } else {
        m_impl = std::make_shared<ContinuousBatchingImpl>(model, tokenizer, scheduler_config, device, properties, generation_config);
    }

    m_impl->m_load_time_ms = get_load_time(start_time);
}

ContinuousBatchingPipeline::ContinuousBatchingPipeline(
    const std::string& model_str,
    const ov::Tensor& weights_tensor,
    const Tokenizer& tokenizer,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& properties,
    const ov::genai::GenerationConfig& generation_config) {
    auto start_time = std::chrono::steady_clock::now();

    auto properties_without_draft_model = properties;
    auto draft_model_desr = extract_draft_model_from_config(properties_without_draft_model);
    auto is_prompt_lookup_enabled = extract_prompt_lookup_from_config(properties_without_draft_model);
    auto model = utils::singleton_core().read_model(model_str, weights_tensor);
    auto directory = std::filesystem::path(get_directory(model_str));

    if (is_prompt_lookup_enabled) {
        OPENVINO_ASSERT(draft_model_desr.model == nullptr, "Speculative decoding and prompt lookup decoding are mutually exclusive");
        m_impl = std::make_shared<PromptLookupImpl>(model, tokenizer, scheduler_config, device, properties_without_draft_model, generation_config);
    } else if (draft_model_desr.model != nullptr) {
        auto main_model_descr = ov::genai::ModelDesc(model, tokenizer, device, properties_without_draft_model, scheduler_config, generation_config);
        m_impl = std::make_shared<SpeculativeDecodingImpl>(main_model_descr, draft_model_desr);
    } else if (std::filesystem::exists(directory / "openvino_text_embeddings_model.xml")) {
        auto vlm_config = ov::genai::VLMConfig{ utils::from_config_json_if_exists<VLMConfig>(directory, "config.json")};
        auto inputs_embedder = std::make_shared<InputsEmbedder>(vlm_config, directory, device, properties);
        m_impl = std::make_shared<ContinuousBatchingImpl>(model, inputs_embedder, tokenizer, scheduler_config, device, properties, generation_config);
    } else {
        m_impl = std::make_shared<ContinuousBatchingImpl>(model, tokenizer, scheduler_config, device, properties, generation_config);
    }

    m_impl->m_load_time_ms = get_load_time(start_time);
}

ov::genai::Tokenizer ContinuousBatchingPipeline::get_tokenizer() const{
    return m_impl->get_tokenizer();
}

ov::genai::GenerationConfig ContinuousBatchingPipeline::get_config() const{
    return m_impl->get_config();
}

void ContinuousBatchingPipeline::set_config(const ov::genai::GenerationConfig& config) {
    m_impl->set_config(config);
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

GenerationHandle ContinuousBatchingPipeline::add_request(uint64_t request_id, const std::string& prompt, const std::vector<ov::Tensor>& rgbs, const ov::genai::GenerationConfig& sampling_params) {
    return m_impl->add_request(request_id, prompt, rgbs, sampling_params);
}

void ContinuousBatchingPipeline::step() {
    m_impl->step();
}

bool ContinuousBatchingPipeline::has_non_finished_requests() {
    return m_impl->has_non_finished_requests();
}

std::vector<EncodedGenerationResult> ContinuousBatchingPipeline::generate(const std::vector<ov::Tensor>& input_ids, const std::vector<ov::genai::GenerationConfig>& sampling_params, const StreamerVariant& streamer) {
    auto encoded_results = m_impl->generate(input_ids, sampling_params, streamer);

    for (auto& encoded_result : encoded_results) {
        encoded_result.perf_metrics.load_time = m_impl->m_load_time_ms;
    }

    return encoded_results;
}

std::vector<GenerationResult> ContinuousBatchingPipeline::generate(const std::vector<std::string>& prompts, const std::vector<ov::genai::GenerationConfig>& sampling_params, const StreamerVariant& streamer) {
    auto decoded_results = m_impl->generate(prompts, sampling_params, streamer);

    for (auto& decoded_result : decoded_results) {
        decoded_result.perf_metrics.load_time = m_impl->m_load_time_ms;
    }

    return decoded_results;
}

std::vector<GenerationResult> ContinuousBatchingPipeline::generate(
             const std::vector<std::string>& prompts,
             const std::vector<std::vector<ov::Tensor>>& rgbs,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer) {
    return m_impl->generate(prompts, rgbs, sampling_params, streamer);;
}


void ContinuousBatchingPipeline::start_chat(const std::string& system_message) {
    m_impl->start_chat(system_message);
};

void ContinuousBatchingPipeline::finish_chat() {
    m_impl->finish_chat();
};
