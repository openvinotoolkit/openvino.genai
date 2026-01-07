// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/module_genai/pipeline.hpp"

#include <optional>

#include "module.hpp"
#include "modules/md_io.hpp"
#include "pipeline_impl.hpp"
#include "utils/yaml_utils.hpp"

namespace ov {
namespace genai {

namespace module {

// config_yaml_path: yaml file.
ModulePipeline::ModulePipeline(const std::filesystem::path& config_yaml_path) {
    auto pipeline_desc = utils::load_config(config_yaml_path);
    ModulePipelineImpl* pImpl = new ModulePipelineImpl(pipeline_desc->main_pipeline_desc, pipeline_desc);
    OPENVINO_ASSERT(pImpl != NULL, "Create ModulePipelineImpl return null.");
    m_pipeline_impl = (ModulePipelineImpl*)pImpl;
}

ModulePipeline::ModulePipeline(const std::string& config_yaml_content) {
    auto pipeline_desc = utils::load_config_from_string(config_yaml_content);
    ModulePipelineImpl* pImpl = new ModulePipelineImpl(pipeline_desc->main_pipeline_desc, pipeline_desc);
    OPENVINO_ASSERT(pImpl != NULL, "Create ModulePipelineImpl return null.");
    m_pipeline_impl = (ModulePipelineImpl*)pImpl;
}

ModulePipeline::~ModulePipeline() {
    auto* pImpl = (ModulePipelineImpl*)m_pipeline_impl;
    delete pImpl;
    m_pipeline_impl = nullptr;
}

// input all parameters in config.yaml
// "prompt": string
// "image": image ov::Tensor or std::vector<ov::Tensor>
// "video": video ov::Tensor
void ModulePipeline::generate(ov::AnyMap& inputs, StreamerVariant streamer) {
    auto* pImpl = (ModulePipelineImpl*)m_pipeline_impl;
    pImpl->generate(inputs, streamer);
}

ov::Any ModulePipeline::get_output(const std::string& output_name) {
    auto* pImpl = (ModulePipelineImpl*)m_pipeline_impl;
    return pImpl->get_output(output_name);
}

void ModulePipeline::start_chat(const std::string& system_message) {
    auto* pImpl = (ModulePipelineImpl*)m_pipeline_impl;
    return pImpl->start_chat(system_message);
}

void ModulePipeline::finish_chat() {
    auto* pImpl = (ModulePipelineImpl*)m_pipeline_impl;
    return pImpl->finish_chat();
}

}  // namespace module
}  // namespace genai
}  // namespace ov
