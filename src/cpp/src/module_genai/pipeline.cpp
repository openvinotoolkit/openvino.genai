// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <optional>
#include <sstream>
#include "logger.hpp"

#include "openvino/genai/module_genai/pipeline.hpp"

#include "comfyui/comfyui.hpp"
#include "module.hpp"
#include "modules/md_io.hpp"
#include "pipeline_impl.hpp"
#include "utils/yaml_utils.hpp"

namespace ov {
namespace genai {

namespace module {

// config_yaml_path: yaml file.
ModulePipeline::ModulePipeline(const std::filesystem::path& config_yaml_path, ConfigModelsMap models_map) {
    auto pipeline_desc = utils::load_config(config_yaml_path.string());
    pipeline_desc->setConfigModelsMap(models_map);

    ModulePipelineImpl* pImpl = new ModulePipelineImpl(pipeline_desc->main_pipeline_desc, pipeline_desc);
    OPENVINO_ASSERT(pImpl != NULL, "Create ModulePipelineImpl return null.");
    m_pipeline_impl = (ModulePipelineImpl*)pImpl;
}

ModulePipeline::ModulePipeline(const std::string& config_yaml_content, ConfigModelsMap models_map) {
    auto pipeline_desc = utils::load_config_from_string(config_yaml_content);
    pipeline_desc->setConfigModelsMap(models_map);

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

    // Check for empty content
    if (config_yaml_content.empty()) {
        result.errors.push_back("Empty YAML content provided");
        result.valid = false;
        return result;
    }

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

namespace {

// Helper function to convert parsed JSON to YAML
std::string convert_parsed_json_to_yaml(
    comfyui::ComfyUIJsonParser& parser,
    ov::AnyMap& pipeline_inputs) {

    GENAI_INFO("JSON parsed successfully");

    const auto& nodes = parser.get_nodes();
    comfyui::log_parsed_nodes(nodes);

    // Get the API JSON (parser handles workflow->API conversion internally)
    const auto& api_json = parser.get_api_json();

    // Validate the prompt before conversion
    auto validation_result = parser.validate_prompt();
    if (!validation_result.success) {
        parser.get_validation_errors_string(validation_result);
        return "";
    }
    GENAI_INFO("Prompt validation passed");

    // Convert to YAML using ComfyUIToGenAIConverter with pipeline_inputs extraction
    // This will also extract pipeline parameters from the JSON
    auto options = comfyui::create_conversion_options(pipeline_inputs);
    comfyui::ComfyUIToGenAIConverter converter;
    std::string yaml_content = converter.convert_to_yaml(api_json, pipeline_inputs, options);

    // Validate the generated YAML before returning
    auto yaml_validation_result = ModulePipeline::validate_config_string(yaml_content);
    if (!yaml_validation_result.valid) {
        GENAI_ERR("Generated YAML validation failed:");
        for (const auto& err : yaml_validation_result.errors) {
            GENAI_ERR("  - %s", err.c_str());
        }
        return "";
    }

    return yaml_content;
}

}  // anonymous namespace

std::string ModulePipeline::comfyui_json_to_yaml(
    const std::filesystem::path& comfyui_json_path,
    ov::AnyMap& pipeline_inputs) {

    try {
        // Parse and validate JSON using ComfyUIJsonParser
        comfyui::ComfyUIJsonParser parser;

        if (!parser.load_json_file(comfyui_json_path)) {
            GENAI_DEBUG("Failed to parse JSON file: %s", comfyui_json_path.string().c_str());
            return "";
        }

        return convert_parsed_json_to_yaml(parser, pipeline_inputs);

    } catch (const std::exception& e) {
        GENAI_ERR("ComfyUI JSON to YAML conversion failed: %s", e.what());
        return "";
    }
}

std::string ModulePipeline::comfyui_json_string_to_yaml(
    const std::string& comfyui_json_content,
    ov::AnyMap& pipeline_inputs) {

    try {
        // Parse and validate JSON using ComfyUIJsonParser
        comfyui::ComfyUIJsonParser parser;

        if (!parser.parse_json_string(comfyui_json_content)) {
            GENAI_DEBUG("Failed to parse JSON content");
            return "";
        }

        return convert_parsed_json_to_yaml(parser, pipeline_inputs);

    } catch (const std::exception& e) {
        GENAI_ERR("ComfyUI JSON to YAML conversion failed: %s", e.what());
        return "";
    }
}

}  // namespace module
}  // namespace genai
}  // namespace ov
