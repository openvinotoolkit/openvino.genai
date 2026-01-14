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

}  // namespace module
}  // namespace genai
}  // namespace ov
