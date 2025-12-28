// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pipeline_impl.hpp"

#include "module.hpp"
#include "utils/yaml_utils.hpp"
#include "modules/md_io.hpp"

namespace ov {
namespace genai {

namespace module {

// config_yaml_path: yaml file.
ModulePipelineImpl::ModulePipelineImpl(const std::filesystem::path& config_yaml_path) {
    auto pipeline_desc = utils::load_config(config_yaml_path);

    // Construct pipeline
    construct_pipeline(pipeline_desc, m_modules);

    // Sort pipeline
    m_modules = sort_pipeline(m_modules);
}

ModulePipelineImpl::ModulePipelineImpl(const std::string& config_yaml_content) {
    auto pipeline_desc = utils::load_config_from_string(config_yaml_content);

    // Construct pipeline
    construct_pipeline(pipeline_desc, m_modules);

    // Sort pipeline
    m_modules = sort_pipeline(m_modules);
}

ModulePipelineImpl::~ModulePipelineImpl() {}

// input all parameters in config.yaml
// "prompt": string
// "image": image ov::Tensor or std::vector<ov::Tensor>
// "video": video ov::Tensor
void ModulePipelineImpl::generate(ov::AnyMap& inputs, StreamerVariant streamer) {
    for (auto& module : m_modules) {
        if (module->is_input_module) {
            std::dynamic_pointer_cast<ParameterModule>(module)->run(inputs);
        } else if (module->is_output_module) {
            std::dynamic_pointer_cast<ResultModule>(module)->run(this->outputs);
        } else {
            module->run();
        }
    }
}

ov::Any ModulePipelineImpl::get_output(const std::string& output_name) {
    return outputs[output_name];
}

void ModulePipelineImpl::start_chat(const std::string& system_message) {}

void ModulePipelineImpl::finish_chat() {}

}  // namespace module
}  // namespace genai
}  // namespace ov
