// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module_genai/utils/yaml_utils.hpp"

#include <yaml-cpp/yaml.h>

#include <filesystem>

#include "logger.hpp"
#include "module_genai/modules/md_vision_preprocess/md_img_preprocess.hpp"
#include "module_genai/modules/md_io.hpp"
#include "module_genai/modules/md_text_encoder.hpp"
#include "module_genai/utils/com_utils.hpp"

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

    if (node["thread_mode"])
        desc->thread_mode = ThreadModeConverter::fromString(node["thread_mode"].as<std::string>());

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

void parse_global_context(const YAML::Node& global_context, PipelineDesc::PTR& pipeline_desc) {
    const YAML::Node& global = global_context;

    OPENVINO_ASSERT(global, "'global_context' key not found.");

    // bool shared_mem = global["enable_shared_memory"] ? global["enable_shared_memory"].as<bool>() : false;
    // GENAI_INFO("  enable_shared_memory: " + std::string(shared_mem ? "True" : "False"));

    OPENVINO_ASSERT(global["model_type"], "'model_type' key not found in 'global_context'.");
    pipeline_desc->model_type = global["model_type"].as<std::string>();
    GENAI_INFO("model_type: " + pipeline_desc->model_type);
}

void parse_main_pipeline_config_internal(const YAML::Node& pipeline_modules, PipelineDesc::PTR& pipeline_desc, const std::string& root_path = ".") {
    GENAI_INFO("Parsing main pipeline modules...");
    const YAML::Node& modules_node = pipeline_modules;
    if (modules_node && modules_node.IsMap()) {
        for (YAML::const_iterator it = modules_node.begin(); it != modules_node.end(); ++it) {
            std::string module_name = it->first.as<std::string>();
            const YAML::Node& module_config = it->second;

            auto module_desc = parse_module(module_config);
            module_desc->name = module_name;
            module_desc->model_type = pipeline_desc->model_type;
            module_desc->config_root_path = root_path;
            pipeline_desc->main_pipeline_desc[module_name] = module_desc;

            GENAI_INFO(module_desc_to_string(module_desc));
        }
    } else {
        GENAI_ERR("'pipeline_modules' key not found or is not a map.");
    }
}

void parse_sub_modules_pipeline_config_internal(const YAML::Node& sub_modules, PipelineDesc::PTR& pipeline_desc, const std::string& root_path = ".") {
    GENAI_INFO("Parsing sub pipeline modules...");
    if (!sub_modules) {
        return;
    }

    if (sub_modules && sub_modules.IsSequence()) {
        for (auto sub_module_entry : sub_modules) {
            OPENVINO_ASSERT(sub_module_entry["name"], "Each sub_module entry must have a 'name' key.");
            std::string sub_pipeline_name = sub_module_entry["name"].as<std::string>();

            PipelineModulesDesc sub_pipeline_desc;
            for (YAML::const_iterator it = sub_module_entry.begin(); it != sub_module_entry.end(); ++it) {
                std::string key = it->first.as<std::string>();
                if (key == "name") {
                    continue;                    
                }

                const YAML::Node& module_config = it->second;
                auto module_desc = parse_module(module_config);
                module_desc->name = key;
                module_desc->model_type = pipeline_desc->model_type;
                module_desc->config_root_path = root_path;
                sub_pipeline_desc[key] = module_desc;

                GENAI_INFO(module_desc_to_string(module_desc));
            }
            pipeline_desc->sub_pipeline_descs.emplace_back(sub_pipeline_name, sub_pipeline_desc);
        }
    }
}

PipelineDesc::PTR load_config(const std::filesystem::path& cfg_path) {
    try {
        YAML::Node config = YAML::LoadFile(cfg_path.string());
        yaml_cfg_auto_padding(config);

        std::string root_path = std::filesystem::path(cfg_path).has_parent_path()
                                    ? std::filesystem::path(cfg_path).parent_path().string()
                                    : std::filesystem::current_path().string();

        PipelineDesc::PTR pipeline_desc = PipelineDesc::create();
        parse_global_context(config["global_context"], pipeline_desc);
        parse_main_pipeline_config_internal(config["pipeline_modules"], pipeline_desc, root_path);
        parse_sub_modules_pipeline_config_internal(config["sub_modules"], pipeline_desc, root_path);
        return pipeline_desc;

    } catch (const YAML::BadFile& e) {
        GENAI_ERR("Could not find or open 'config.yaml'. Please make sure the file exists.");
    } catch (const YAML::Exception& e) {
        GENAI_ERR(std::string("Error parsing YAML: ") + e.what());
    }
    return {};
}

PipelineDesc::PTR load_config_from_string(const std::string& content) {
    // Check content is not a file path - YAML content should contain newlines
    // A simple file path would not contain newlines
    bool looks_like_yaml = content.find('\n') != std::string::npos ||
                           content.find("global_context") != std::string::npos ||
                           content.find("pipeline_modules") != std::string::npos;
    OPENVINO_ASSERT(looks_like_yaml,
                    "The provided content seems to be a file path. Please use 'std::filesystem::path' to pass file path "
                    "instead. content: " +
                        content);

    try {
        PipelineDesc::PTR pipeline_desc = PipelineDesc::create();
        YAML::Node config = YAML::Load(content);

        yaml_cfg_auto_padding(config);

        if (check_env_variable("DUMP_YAML")) {
            save_yaml_to_file(config, "dumped_config.yaml");
        }

        parse_global_context(config["global_context"], pipeline_desc);
        parse_main_pipeline_config_internal(config["pipeline_modules"], pipeline_desc);
        parse_sub_modules_pipeline_config_internal(config["sub_modules"], pipeline_desc);
        return pipeline_desc;
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

const std::string module_desc_to_string(const IBaseModuleDesc::PTR& desc) {
    std::stringstream ss;
    ss << desc;
    return ss.str();
}

}  // namespace utils
}  // namespace module
}  // namespace genai
}  // namespace ov