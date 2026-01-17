// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "md_llm_inference.hpp"

#include "module_genai/module_factory.hpp"
#include <fstream>

namespace ov {
namespace genai {

extern std::shared_ptr<ov::Model> g_llm_model;
extern std::shared_ptr<ov::Model> g_model_vision_embeddings_merger;
namespace module {

GENAI_REGISTER_MODULE_SAME(LLMInferenceModule);

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
      - name: "rope_delta"          # [Optional]
        type: "Int"
        source: "ParentModuleName.OutputPortName"
      - name: "embeds_list"         # [Optional]
        type: "VecOVTensor"
        source: "ParentModuleName.OutputPortName"
      - name: "position_ids_list"   # [Optional]
        type: "VecOVTensor"
        source: "ParentModuleName.OutputPortName"
      - name: "rope_delta_list"     # [Optional]
        type: "VecInt"
        source: "ParentModuleName.OutputPortName"
    outputs:
      - name: "generated_text"      # Correspoding input: embeds
        type: "String"
      - name: "generated_texts"     # Correspoding input: embeds_list
        type: "VecString"
    params:
      model_path: "model_path"              # Optional, if 'model_path' is not provided, model will be loaded from 'models_map', refer: ModulePipeline constructor.
                                            # 'ov_model_embed', 'ov_model' should be provided in models_map in this case.
      model_cfg_path: "model_config.json"   # Optional, if model_path is not provided, model_cfg_path is required, else it will be ignored.
      max_new_tokens: "256"
      do_sample: "false"
      top_p: "1.0"
      top_k: "50"
      temperature: "1.0"
      repetition_penalty: "1.0"
    )" << std::endl;
}

LLMInferenceModule::LLMInferenceModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : IBaseModule(desc, pipeline_desc) {
    if (!initialize()) {
        GENAI_ERR("Failed to initialize LLMInferenceModule");
    }
}

LLMInferenceModule::~LLMInferenceModule() {}

bool LLMInferenceModule::load_generation_config(const std::filesystem::path& config_path) {
    try {
        std::ifstream f(config_path.string());
        if (!f.is_open()) {
            GENAI_ERR("Failed to open generation config file: " + config_path.string());
            return false;
        }
        m_generation_config = ov::genai::GenerationConfig(config_path.string());
        return true;
    } catch (const std::exception& e) {
        GENAI_ERR(std::string("Error loading generation config: ") + e.what());
        return false;
    }
}

bool LLMInferenceModule::initialize() {
    const auto& params = module_desc->params;

    bool has_param_model_path = false;
    std::filesystem::path models_path = get_optional_param("model_path");
    if (models_path.empty()) {
        m_ov_model_embed = get_ov_model_from_cfg_models_map("ov_model_embed", true);
        models_path = get_param("model_cfg_path");

        // Pass model to global variables, tmp solution for vLLM pipeline
        g_llm_model = m_ov_model;
        g_model_vision_embeddings_merger = m_ov_model_embed;
    }

    // Force to use PA backend
    ov::AnyMap cfg{};
    cfg["ATTENTION_BACKEND"] = "PA";

    load_generation_config(models_path / "generation_config.json");
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
		m_cb_pipeline = std::make_unique<ov::genai::VLMPipeline::VLMContinuousBatchingAdapter>(models_path, scheduler_config, module_desc->device, plugin_properties);
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
    std::vector<int> rope_delta_list;
    if (this->inputs.find("embeds") != this->inputs.end()) {
        embeds_list.push_back(inputs["embeds"].data.as<ov::Tensor>());
        position_ids_list.push_back(inputs["position_ids"].data.as<ov::Tensor>());
        rope_delta_list.push_back(inputs["rope_delta"].data.as<int>());
    } else if (this->inputs.find("embeds_list") != this->inputs.end()) {
        embeds_list = inputs["embeds_list"].data.as<std::vector<ov::Tensor>>();
        position_ids_list = inputs["position_ids_list"].data.as<std::vector<ov::Tensor>>();
        rope_delta_list = inputs["rope_delta_list"].data.as<std::vector<int>>();
        is_batch = true;
    } else {
        GENAI_ERR("LLMInferenceModule[" + module_desc->name + "]: 'embeds or embeds_list' input not found")
    }

    std::vector<std::pair<ov::Tensor, std::optional<int64_t>>> input_position_ids_list;
    input_position_ids_list.reserve(position_ids_list.size());
    for (size_t i = 0; i < position_ids_list.size(); ++i) {
        input_position_ids_list.push_back({position_ids_list[i], std::optional<int64_t>(rope_delta_list[i])});
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
