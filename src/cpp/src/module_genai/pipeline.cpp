// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <optional>
#include <sstream>

#include "openvino/genai/module_genai/pipeline.hpp"

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

// execute generate asynchronously
void ModulePipeline::generate_async(ov::AnyMap& inputs, StreamerVariant streamer) {
    auto* pImpl = (ModulePipelineImpl*)m_pipeline_impl;
    pImpl->generate_async(inputs, streamer);
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

ModulePipeline::ValidationResult ModulePipeline::validate_config(const std::filesystem::path& config_yaml_path) {
    std::ifstream file(config_yaml_path);
    if (!file.is_open()) {
        return {false, {"Failed to open config file: " + config_yaml_path.string()}, {}};
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return validate_config_string(buffer.str());
}

ModulePipeline::ValidationResult ModulePipeline::validate_config_string(const std::string& config_yaml_content) {
    ValidationResult result;
    result.valid = true;

    try {
        YAML::Node config = YAML::Load(config_yaml_content);

        // Check for global_context (optional, just a warning)
        if (!config["global_context"]) {
            result.warnings.push_back("Missing 'global_context' section (optional)");
        }

        // Check if pipeline_modules exists
        if (!config["pipeline_modules"]) {
            result.errors.push_back("Missing 'pipeline_modules' section");
            result.valid = false;
            return result;
        }

        YAML::Node pipeline_modules = config["pipeline_modules"];

        // Check for ParameterModule
        bool has_parameter_module = false;
        bool has_result_module = false;

        for (const auto& module : pipeline_modules) {
            std::string module_name = module.first.as<std::string>();
            YAML::Node module_config = module.second;

            if (module_config["type"]) {
                std::string type = module_config["type"].as<std::string>();
                if (type == "ParameterModule") {
                    has_parameter_module = true;
                } else if (type == "ResultModule") {
                    has_result_module = true;
                }
            }
        }

        if (!has_parameter_module) {
            result.errors.push_back("Missing ParameterModule configuration");
            result.valid = false;
        }

        if (!has_result_module) {
            result.errors.push_back("Missing ResultModule configuration");
            result.valid = false;
        }

    } catch (const YAML::Exception& e) {
        result.errors.push_back(std::string("YAML parsing error: ") + e.what());
        result.valid = false;
    } catch (const std::exception& e) {
        result.errors.push_back(std::string("Validation error: ") + e.what());
        result.valid = false;
    }

    return result;
}

}  // namespace module
}  // namespace genai
}  // namespace ov
