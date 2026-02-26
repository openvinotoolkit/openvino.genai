// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module_genai/module_base.hpp"

#include <filesystem>
#include "logger.hpp"

namespace ov {
namespace genai {
namespace module {

namespace fs = std::filesystem;

IBaseModule::IBaseModule(const IBaseModuleDesc::PTR& desc, const PipelineDesc::PTR& pipeline_desc)
    : module_desc(desc),
      pipeline_desc(pipeline_desc) {
    for (auto& input : desc->inputs) {
        this->inputs[input.name] = InputModule();
        this->inputs[input.name].parent_port_name = input.source_module_out_name;
    }
    for (auto& output : desc->outputs) {
        this->outputs[output.name] = OutputModule();
    }

    // Initilize first ov::Model with key "ov_model"
    init_ov_model();
}

void IBaseModule::prepare_inputs() {
    for (auto& input : this->inputs) {
        const auto& parent_port_name = input.second.parent_port_name;
        if (input.second.module_ptr.lock() != nullptr) {
            input.second.data = input.second.module_ptr.lock()->outputs[parent_port_name].data;
        }
    }
}

const std::string& IBaseModule::get_module_name() const {
    return module_desc->name;
}

bool IBaseModule::exists_input(const std::string& input_name) {
    return inputs.find(input_name) != inputs.end();
}

ov::Any& IBaseModule::get_input(const std::string& input_name) {
    OPENVINO_ASSERT(exists_input(input_name),
                    "Module[" + module_desc->name + "]: input '" + input_name + "' not found in inputs");
    return inputs[input_name].data;
}

std::string IBaseModule::get_param(const std::string& param_item) {
    const auto& params = module_desc->params;
    auto it_models_path = params.find(param_item);
    OPENVINO_ASSERT(it_models_path != params.end(),
                    "Module[" + module_desc->name + "]: '" + param_item + "' not found in params");
    return it_models_path->second;
}

std::string IBaseModule::get_optional_param(const std::string& param_item) {
    const auto& params = module_desc->params;
    auto it_models_path = params.find(param_item);
    if (it_models_path == params.end()) {
        return std::string();
    }
    return it_models_path->second;
}

void IBaseModule::init_ov_model() {
    if (m_ov_model == nullptr) {
        m_ov_model = get_ov_model_from_cfg_models_map("ov_model", false);
    }
}

std::shared_ptr<ov::Model> IBaseModule::get_ov_model_from_cfg_models_map(const std::string& param_name, bool required) {
    const auto& models_map = pipeline_desc->getConfigModelsMap();
    auto it = models_map.find(module_desc->name);
    if (it == models_map.end()) {
        OPENVINO_ASSERT(!required, "Module[" + module_desc->name + "] is not found in models_map");
        return nullptr;
    }

    auto param_it = it->second.find(param_name);
    if (param_it == it->second.end()) {
        OPENVINO_ASSERT(
            !required,
            "Module[" + module_desc->name + "]: ov::Model with name '" + param_name + "' not found in models_map");
    }

    return param_it->second;
}

std::string IBaseModuleDesc::get_full_path(const std::string& fn) {
    // Check if fn is absolute path or file exists
    if (fs::exists(fn) || fs::path(fn).is_absolute()) {
        return fn;
    }

    fs::path joined_path = config_root_path / fn;
    if (fs::exists(joined_path) || fs::path(joined_path).is_absolute()) {
        return joined_path.string();
    }
    OPENVINO_ASSERT(false, "File path is invalid: " + fn);
}

void IBaseModule::check_dynamic_load_weights() {
    const auto& params = module_desc->params;
    auto it_path = params.find("dynamic_load_weights");
    if (it_path != params.end()) {
        std::string val = module_desc->params["dynamic_load_weights"];
        if (val == "true" || val == "True" || val == "TRUE" || val == "1") {
            m_dynamic_load_weights = true;
            GENAI_INFO("Module[" + module_desc->name + "]: m_dynamic_load_weights = true");
            check_cache_dir();
            if (m_cache_dir.empty()) {
                OPENVINO_THROW("Module[" + module_desc->name +
                               "]: 'cache_dir' must be set when 'dynamic_load_weights' is true");
            }
        }
    }
}

void IBaseModule::check_cache_dir() {
    if (m_cache_dir.empty()) {
        const auto& params = module_desc->params;
        auto it_path = params.find("cache_dir");
        if (it_path != params.end()) {
            m_cache_dir = module_desc->params["cache_dir"];
            GENAI_INFO("Module[" + module_desc->name + "]: m_cache_dir = " + m_cache_dir);
        }
    } else {
        GENAI_INFO("Module[" + module_desc->name + "]: m_cache_dir = " + m_cache_dir);
    }
}

void IBaseModule::check_splitted_model() {
    auto splitted_model = get_optional_param("splitted_model");
    if (splitted_model.empty()) {
        return;
    }

    if (splitted_model == "true" || splitted_model == "True" || splitted_model == "TRUE" || splitted_model == "1") {
        m_splitted_model = true;
        GENAI_INFO("Module[" + module_desc->name + "]: m_splitted_model = true");
    }
}

bool IBaseModule::check_bool_param(const std::string& param_name, const bool& default_value) {
    auto p = get_optional_param(param_name);
    if (p.empty()) {
        return default_value;
    }

    if (p == "true" || p == "True" || p == "TRUE" || p == "1") {
        GENAI_INFO("Module[" + module_desc->name + "]: " + param_name + " = true");
        return true;
    } else if (p == "false" || p == "False" || p == "FALSE" || p == "0") {
        GENAI_INFO("Module[" + module_desc->name + "]: " + param_name + " = false");
        return false;
    }
    GENAI_ERR("Module[" + module_desc->name + "]: Invalid bool param value for '" + param_name + "': " + p +
              ", use default value: " + (default_value ? "true" : "false"));
    return default_value;
}

// PipelineDesc implementation
PipelineDesc::PipelineDesc() : m_resource_cache(std::make_unique<PipelineResourceCache>()) {}

PipelineDesc::~PipelineDesc() = default;

PipelineResourceCache& PipelineDesc::get_resource_cache() {
    return *m_resource_cache;
}

}  // namespace module
}  // namespace genai
}  // namespace ov
