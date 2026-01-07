// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module_genai/utils/yaml_utils.hpp"
#include "module_genai/utils/data_type_converter.hpp"
#include "logger.hpp"

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>

namespace ov {
namespace genai {

namespace module {
namespace utils {

inline void yaml_cfg_auto_padding_io_module(YAML::Node& modules) {
    std::map<std::string, std::string> extracted_params;
    std::map<std::string, std::string> extracted_results;

    // When only one module is specified, the ParameterModule and ResultModule will be automatically inferred and added.
    std::string auto_padding_param_name = "pipeline_params";
    std::string auto_padding_result_name = "pipeline_results";

    // only one node recursive
    std::string test_module_name;
    for (auto it = modules.begin(); it != modules.end(); ++it) {
        test_module_name = it->first.as<std::string>();
        if (test_module_name == "name") {
            // skip name key
            continue;
        }

        // get inputs
        YAML::Node inputs = it->second["inputs"];
        if (inputs && inputs.IsSequence()) {
            for (const auto& input : inputs) {
                std::string input_name = input["name"].as<std::string>("");
                std::string source = input["source"].as<std::string>(std::string());
                if (source.empty()) {
                    std::string type = input["type"].as<std::string>("");
                    extracted_params[input_name] = type;
                    int index = 0;
                    for (const auto &item: it->second["inputs"]) {
                        if (item["name"].as<std::string>() == input_name) {
                            break;
                        }
                        index++;
                    }
                    it->second["inputs"][index]["source"] = auto_padding_param_name + "." + input_name;
                    continue;
                } else if (source.find(auto_padding_param_name + ".") == 0) {
                    std::string param_name = source.substr(auto_padding_param_name.size() + 1);
                    std::string type = input["type"].as<std::string>("");
                    extracted_params[param_name] = type;
                } else {
                    OPENVINO_ASSERT(false, "Error: Input[" + input_name + "] source format error. ");
                }
            }
        }

        // get outputs
        YAML::Node outputs = it->second["outputs"];
        if (outputs && outputs.IsSequence()) {
            for (const auto& output : outputs) {
                std::string name = output["name"].as<std::string>("");
                std::string type = output["type"].as<std::string>("");
                extracted_results[name] = type;
            }
        }
    }

    // pipeline_params
    YAML::Node params_node;
    params_node["type"] = "ParameterModule";
    YAML::Node outputs_seq;
    for (const auto& param : extracted_params) {
        YAML::Node item;
        item["name"] = param.first;
        item["type"] = param.second;
        outputs_seq.push_back(item);
    }
    if (outputs_seq.size() > 0) {
        params_node["outputs"] = outputs_seq;
    }
    modules[auto_padding_param_name] = params_node;

    // pipeline_results
    YAML::Node results_node;
    results_node["type"] = "ResultModule";
    YAML::Node inputs_seq;
    for (const auto& result : extracted_results) {
        YAML::Node item;
        item["name"] = result.first;
        item["type"] = result.second;
        item["source"] = test_module_name + "." + result.first;
        inputs_seq.push_back(item);
    }
    if (inputs_seq.size() > 0) {
        results_node["inputs"] = inputs_seq;
    }
    modules[auto_padding_result_name] = results_node;
}

void yaml_cfg_auto_padding(YAML::Node& config_node) {
    OPENVINO_ASSERT(config_node["pipeline_modules"], "Test yaml config miss 'pipeline_modules'.");

    YAML::Node modules = config_node["pipeline_modules"];
    if (modules.size() == 1) {
        // only one module, auto padding ParameterModule and ResultModule
        yaml_cfg_auto_padding_io_module(modules);
    }

    // dump yaml to file for debug
    // save_yaml_to_file(config_node, "debug_auto_padding.yaml");

    YAML::Node sub_modules = config_node["sub_modules"];
    if (sub_modules && sub_modules.IsSequence()) {
        for (auto sub_module_entry : sub_modules) {
            OPENVINO_ASSERT(sub_module_entry["name"], "Each sub_module entry must have a 'name' key.");
            std::string sub_pipeline_name = sub_module_entry["name"].as<std::string>();

            if (sub_module_entry.size() == 2) {
                 // only one module except for "name", auto padding ParameterModule and ResultModule
                for (YAML::const_iterator it = sub_module_entry.begin(); it != sub_module_entry.end(); ++it) {
                    std::string key = it->first.as<std::string>();
                    if (key != "name") {
                        yaml_cfg_auto_padding_io_module(sub_module_entry);
                        break;
                    }
                }
            }
        }
    }

    // save_yaml_to_file(config_node, "debug_auto_padding_2.yaml");
}

void save_yaml_to_file(const YAML::Node& node, const std::filesystem::path& file_path) {
    std::ofstream fout(file_path, std::ios::out | std::ios::binary);
    fout << YAML::Dump(node);
    fout.close();
}

}  // namespace utils
}  // namespace module
}  // namespace genai
}  // namespace ov