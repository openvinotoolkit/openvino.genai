// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_llm_inference.hpp"

namespace ov {
namespace genai {
namespace module {

void LLMInferenceModule::print_static_config() {
    std::cout << R"(
global_context:
  model_type: "qwen2_5_vl"
pipeline_modules:
  llm_inference:
    type: "LLMInferenceModule"
    description: "LLM module for Continuous Batch pipeline"
    device: "CPU"
    inputs:
      - name: "embeds"              # [Optional] embedding feature.
        type: "OVTensor"
        source: "ParentModuleName.OutputPortName"
      - name: "position_ids"        # [Optional]
        type: "OVTensor"
        source: "ParentModuleName.OutputPortName"
      - name: "embeds_list"         # [Optional]
        type: "VecOVTensor"
        source: "ParentModuleName.OutputPortName"
      - name: "position_ids_list"   # [Optional]
        type: "VecOVTensor"
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "generated_text"      # Correspoding input: embeds
        type: "String"
      - name: "generated_texts"     # Correspoding input: embeds_list
        type: "VecString"
    params:
      model_path: "model_path"
      max_new_tokens: "256"
      do_sample: "false"
      top_p: "1.0"
      top_k: "50"
      temperature: "1.0"
      repetition_penalty: "1.0"
    )" << std::endl;
}

LLMInferenceModule::LLMInferenceModule(const IBaseModuleDesc::PTR& desc) : IBaseModule(desc) {
    if (!initialize()) {
        GENAI_ERR("Failed to initialize LLMInferenceModule");
    }
}

bool LLMInferenceModule::load_generation_config(const std::string& config_path) {
    try {
        std::ifstream f(config_path);
        if (!f.is_open()) {
        	GENAI_ERR("Failed to open generation config file: " + config_path);
            return false;
        }
        m_generation_config = ov::genai::GenerationConfig(config_path);
        return true;
    } catch (const std::exception& e) {
    	GENAI_ERR(std::string("Error loading generation config: ") + e.what());
        return false;
    }
}

bool LLMInferenceModule::initialize() {
    const auto& params = module_desc->params;

    auto it_models_path = params.find("model_path");
    if (it_models_path == params.end()) {
    	GENAI_ERR("LLMInferenceModule[" + module_desc->name + "]: 'models_path' not found in params");
        return false;
    }
    std::filesystem::path models_path = it_models_path->second;

    // Get device: Default CPU.
    std::string device = module_desc->device.empty() ? "CPU" : module_desc->device;

    // Force to use PA backend
    ov::AnyMap cfg{};
    cfg["ATTENTION_BACKEND"] = "PA";

    load_generation_config(it_models_path->second + "generation_config.json");
    // Override with parameters from module config
    auto apply_param = [&](const std::string& key, auto& target, auto converter) {
        auto it = params.find(key);
        if (it != params.end() && !it->second.empty()) {
            try {
                target = converter(it->second);
            } catch (...) {
                GENAI_ERR("Failed to parse parameter: " + key);
            }
        }
    };

    apply_param("max_new_tokens", m_generation_config.max_new_tokens,
                [](const std::string& s) { return std::stoull(s); });
    apply_param("do_sample", m_generation_config.do_sample,
                [](const std::string& s) { return s == "true" || s == "1"; });
    apply_param("top_p", m_generation_config.top_p,
                [](const std::string& s) { return std::stof(s); });
    apply_param("top_k", m_generation_config.top_k,
                [](const std::string& s) { return std::stoull(s); });
    apply_param("temperature", m_generation_config.temperature,
                [](const std::string& s) { return std::stof(s); });
    apply_param("repetition_penalty", m_generation_config.repetition_penalty,
                [](const std::string& s) { return std::stof(s); });
    apply_param("apply_chat_template", m_generation_config.apply_chat_template,
                [](const std::string& s) { return s == "true" || s == "1"; });

    try {
		auto [properties, attention_backend] = utils::extract_attention_backend(cfg);
		auto [plugin_properties, scheduler_config] = utils::extract_scheduler_config(properties, utils::get_latency_oriented_scheduler_config());
		m_cb_pipeline = std::make_unique<ov::genai::VLMPipeline::VLMContinuousBatchingAdapter>(models_path, scheduler_config, device, plugin_properties);
    	return true;
    } catch (const std::exception& e) {
    	GENAI_ERR("LLMInferenceModule[" + module_desc->name + "]: Failed to create pipeline: " + e.what());
        return false;
    }
}

void LLMInferenceModule::run() {
	GENAI_INFO("Running module: " + module_desc->name);

    prepare_inputs();

    bool is_batch = false;
    std::vector<ov::Tensor> embeds_list;
    std::vector<ov::Tensor> position_ids_list;
    if (this->inputs.find("embeds") != this->inputs.end()) {
        embeds_list.push_back(inputs["embeds"].data.as<ov::Tensor>());
        position_ids_list.push_back(inputs["position_ids"].data.as<ov::Tensor>());
    } else if (this->inputs.find("embeds_list") != this->inputs.end()) {
        embeds_list = inputs["embeds_list"].data.as<std::vector<ov::Tensor>>();
        position_ids_list = inputs["position_ids_list"].data.as<std::vector<ov::Tensor>>();
        is_batch = true;
    } else {
        GENAI_ERR("TextEmbeddingModule[" + module_desc->name + "]: 'embeds or embeds_list' input not found")
    }

    std::vector<std::pair<ov::Tensor, std::optional<int64_t>>> input_position_ids_list;
    for (auto& pids : position_ids_list) {
        input_position_ids_list.push_back({pids, std::optional<int64_t>(std::nullopt)});
    }

    auto results_vec = m_cb_pipeline->generate(embeds_list, std::vector<GenerationConfig>{m_generation_config}, std::monostate(), std::nullopt, input_position_ids_list);
    std::string generated_text = "";
    if (results_vec.size()) {
        auto& results = results_vec[0];
        // Decode the generated token IDs to text
        if (!results.m_generation_ids.empty() && !results.m_generation_ids[0].empty()) {
            generated_text = m_cb_pipeline->get_tokenizer().decode(results.m_generation_ids[0]);
    	    GENAI_INFO("LLM output: " + generated_text);
    	}

        if (is_batch) {
            this->outputs["generated_texts"].data = std::vector<std::string>{generated_text};
        } else {
            this->outputs["generated_text"].data = generated_text;
        }
    }

    GENAI_INFO("LLMInferenceModule[" + module_desc->name + "] generation completed.");
}

}  // namespace module
}  // namespace genai
}  // namespace ov
