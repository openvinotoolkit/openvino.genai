// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module_genai/utils/yaml_utils.hpp"
#include "module_genai/utils/data_type_converter.hpp"
#include "logger.hpp"

#include <yaml-cpp/yaml.h>
#include <filesystem>

#include "module_genai/modules/md_img_preprocess.hpp"
#include "module_genai/modules/md_io.hpp"
#include "module_genai/modules/md_text_encoder.hpp"

namespace ov {
namespace genai {

namespace module {
namespace utils {

std::pair<std::string, std::string> parse_source(const std::string& source) {
    size_t dot_pos = source.find('.');
    OPENVINO_ASSERT(dot_pos != std::string::npos, "Source string doesn't contain '.'");

    std::string part1 = source.substr(0, dot_pos);
    std::string part2 = source.substr(dot_pos + 1);
    return {part1, part2};
}

InputPort parse_input_port(const YAML::Node& node, bool is_input) {
    InputPort port;
    if (node["name"]) {
        port.name = node["name"].as<std::string>();
    }
    if (node["type"]) {
        port.dt_type = DataTypeConverter::fromString(node["type"].as<std::string>());
    }
    if (is_input && node["source"]) {
        std::string source = node["source"].as<std::string>();
        auto source_md = parse_source(source);
        port.source_module_name = source_md.first;
        port.source_module_out_name = source_md.second;
    }
    return port;
}

OutputPort parse_output_port(const YAML::Node& node, bool is_input) {
    OutputPort port;
    if (node["name"]) {
        port.name = node["name"].as<std::string>();
    }
    if (node["type"]) {

        port.dt_type = DataTypeConverter::fromString(node["type"].as<std::string>());
    }
    return port;
}

IBaseModuleDesc::PTR parse_module(const YAML::Node& node) {
    IBaseModuleDesc::PTR desc = IBaseModuleDesc::create();

    ModuleType module_type = ModuleType::Unknown;
    if (node["type"]) {
        std::string md_type = node["type"].as<std::string>();
        module_type = ModuleTypeConverter::fromString(md_type);
        OPENVINO_ASSERT(module_type != ModuleType::Unknown, "Unknown ModuleType string: " + md_type);
        desc->type = module_type;
    }

    // Parse common contribute.
    if (node["device"])
        desc->device = node["device"].as<std::string>();
    if (node["description"])
        desc->description = node["description"].as<std::string>();

    if (node["inputs"] && node["inputs"].IsSequence()) {
        for (const auto& input_node : node["inputs"]) {
            desc->inputs.push_back(parse_input_port(input_node, true));
        }
    }

    if (node["outputs"] && node["outputs"].IsSequence()) {
        for (const auto& output_node : node["outputs"]) {
            desc->outputs.push_back(parse_output_port(output_node, false));
        }
    }

    if (node["params"] && node["params"].IsMap()) {
        for (YAML::const_iterator it = node["params"].begin(); it != node["params"].end(); ++it) {
            desc->params[it->first.as<std::string>()] =
                it->second.IsScalar() ? it->second.as<std::string>() : "[Complex Value]";
        }
    }
    
    return desc;
}

PipelineModuleDesc parse_pipeline_config_internal(const YAML::Node& config, const std::string& root_path = ".") {
    PipelineModuleDesc pipeline_desc;
    const YAML::Node& global = config["global_context"];
    std::string model_type;
    if (global) {
        std::string device = global["default_device"] ? global["default_device"].as<std::string>() : "N/A";
        bool shared_mem = global["enable_shared_memory"] ? global["enable_shared_memory"].as<bool>() : false;
        model_type = global["model_type"] ? global["model_type"].as<std::string>() : "N/A";

        GENAI_INFO("  Default Device: " + device);
        GENAI_INFO("  Enable Shared Memory: " + std::string(shared_mem ? "True" : "False"));
        GENAI_INFO("  Model Type: " + model_type);
    }

    const YAML::Node& modules_node = config["pipeline_modules"];
    if (modules_node && modules_node.IsMap()) {
        for (YAML::const_iterator it = modules_node.begin(); it != modules_node.end(); ++it) {
            std::string module_name = it->first.as<std::string>();
            const YAML::Node& module_config = it->second;

            auto module_desc = parse_module(module_config);
            module_desc->name = module_name;
            module_desc->model_type = model_type;
            module_desc->config_root_path = root_path;
            pipeline_desc[module_name] = module_desc;

            GENAI_INFO((std::stringstream() << module_desc).str());
        }
    } else {
        GENAI_ERR("'pipeline_modules' key not found or is not a map.");
    }
    return pipeline_desc;
}

PipelineModuleDesc load_config(const std::string& cfg_path) {
    try {
        YAML::Node config = YAML::LoadFile(cfg_path);
        yaml_cfg_auto_padding(config);

        std::string root_path = std::filesystem::path(cfg_path).has_parent_path()
                                    ? std::filesystem::path(cfg_path).parent_path().string()
                                    : std::filesystem::current_path().string();
        return parse_pipeline_config_internal(config, root_path);

    } catch (const YAML::BadFile& e) {
        GENAI_ERR("Could not find or open 'config.yaml'. Please make sure the file exists.");
    } catch (const YAML::Exception& e) {
        GENAI_ERR(std::string("Error parsing YAML: ") + e.what());
    }
    return {};
}

PipelineModuleDesc load_config_from_string(const std::string& content) {
    try {
        YAML::Node config = YAML::Load(content);
        yaml_cfg_auto_padding(config);
        return parse_pipeline_config_internal(config);
    } catch (const YAML::Exception& e) {
        GENAI_ERR(std::string("Error parsing YAML: ") + e.what());
    }
    return {};
}

std::ostream& operator<<(std::ostream& os, const IBaseModuleDesc::PTR& desc) {
    // 1. Output the ModuleType
    os << "-- Module Name :" << desc->name << "\n";

    // 2. Output Inputs and Outputs count
    os << "    Inputs (" << desc->inputs.size() << "):\n";
    for (const auto& input : desc->inputs) {
        // Use std::quoted for safety if values might contain spaces/special chars
        os << "      - name: " << input.name << "\n";
        os << "      - type: " << DataTypeConverter::toString(input.dt_type) << "\n";
        os << "      - source: " << input.source_module_name << "." << input.source_module_out_name << "\n";
    }
    os << "    Onputs (" << desc->outputs.size() << "):\n";
    for (const auto& output : desc->outputs) {
        // Use std::quoted for safety if values might contain spaces/special chars
        os << "      - name: " << output.name << "\n";
        os << "      - type: " << DataTypeConverter::toString(output.dt_type) << "\n";
    }

    return os;
}

}  // namespace utils
}  // namespace module
}  // namespace genai
}  // namespace ov