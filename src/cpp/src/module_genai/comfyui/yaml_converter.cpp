// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file yaml_converter.cpp
 * @brief Implementation of ComfyUIToGenAIConverter class
 *
 * Conversion Strategy:
 *   This converter uses a node-centric approach where each ComfyUI node is
 *   processed based on its class_type. Each node handler extracts relevant
 *   parameters and marks the node as processed. The conversion continues
 *   until all nodes are processed.
 */

#include "comfyui.hpp"
#include "yaml_module_generators.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <functional>
#include <set>
#include <yaml-cpp/yaml.h>
#include "logger.hpp"

namespace ov {
namespace genai {
namespace module {
namespace comfyui {

// ============================================================================
// Utility Functions
// ============================================================================

void log_validation_errors(const PromptValidationResult& result) {
    if (result.success) {
        return;
    }

    GENAI_DEBUG("VALIDATION FAILED");
    GENAI_DEBUG("Main Error:");
    GENAI_DEBUG("  Type: %s", result.error.type.c_str());
    GENAI_DEBUG("  Message: %s", result.error.message.c_str());
    GENAI_DEBUG("  Details: %s", result.error.details.c_str());

    if (!result.output_errors.empty()) {
        GENAI_DEBUG("Output Node Errors:");
        for (const auto& [output_id, errors] : result.output_errors) {
            GENAI_DEBUG("  Output Node %s:", output_id.c_str());
            for (const auto& err : errors) {
                GENAI_DEBUG("    - [%s] %s", err.type.c_str(), err.message.c_str());
                GENAI_DEBUG("      Details: %s", err.details.c_str());
            }
        }
    }

    if (!result.node_errors.empty()) {
        GENAI_DEBUG("Node-Specific Errors:");
        for (const auto& [node_id, node_error_json] : result.node_errors) {
            GENAI_DEBUG("  Node %s (%s):", node_id.c_str(),
                      node_error_json["class_type"].template get<std::string>().c_str());

            for (const auto& err : node_error_json["errors"]) {
                GENAI_DEBUG("    - [%s] %s", err["type"].template get<std::string>().c_str(),
                          err["message"].template get<std::string>().c_str());
                GENAI_DEBUG("      Details: %s", err["details"].template get<std::string>().c_str());
            }

            std::string affected_outputs;
            for (const auto& output : node_error_json["dependent_outputs"]) {
                affected_outputs += output.template get<std::string>() + " ";
            }
            GENAI_DEBUG("      Affects outputs: %s", affected_outputs.c_str());
        }
    }

    GENAI_DEBUG("Please fix the errors above before executing the workflow.");
}

ConversionOptions create_conversion_options(const ov::AnyMap& pipeline_inputs) {
    std::string model_path = "./models/";
    std::string device = "CPU";
    int tile_size = 0;

    auto it_model_path = pipeline_inputs.find("model_path");
    if (it_model_path != pipeline_inputs.end()) {
        try {
            model_path = it_model_path->second.as<std::string>();
        } catch (const ov::Exception&) {
            GENAI_WARN("model_path has invalid type, using default");
        }
    }

    auto it_device = pipeline_inputs.find("device");
    if (it_device != pipeline_inputs.end()) {
        try {
            device = it_device->second.as<std::string>();
        } catch (const ov::Exception&) {
            GENAI_WARN("device has invalid type, using default");
        }
    }

    auto it_tile_size = pipeline_inputs.find("tile_size");
    if (it_tile_size != pipeline_inputs.end()) {
        try {
            tile_size = it_tile_size->second.as<int>();
        } catch (const ov::Exception&) {
            GENAI_WARN("tile_size has invalid type, using default");
        }
    }

    ConversionOptions options;
    options.model_path = model_path;
    options.device = device;
    options.tile_size = tile_size;
    return options;
}

void log_parsed_nodes(const std::map<std::string, Node>& nodes) {
    GENAI_INFO("Found %zu nodes", nodes.size());
    for (const auto& [node_id, node] : nodes) {
        GENAI_DEBUG("  - Node %s: %s", node_id.c_str(), node.class_type.c_str());
    }
}

// ============================================================================
// Public Methods
// ============================================================================

std::string ComfyUIToGenAIConverter::get_yaml_for_pipeline(
    const std::string& json_file_path,
    const ConversionOptions& options) {

    try {
        std::ifstream file(json_file_path);
        if (!file.is_open()) {
            GENAI_ERR("Failed to open JSON file: %s", json_file_path.c_str());
            return "";
        }

        json api_json = json::parse(file);
        file.close();

        return convert_to_yaml(api_json, options);

    } catch (const std::exception& e) {
        GENAI_ERR("get_yaml_for_pipeline failed: %s", e.what());
        return "";
    }
}

bool ComfyUIToGenAIConverter::convert_file(
    const std::string& api_json_path,
    const std::string& output_yaml_path,
    const ConversionOptions& options) {

    try {
        std::ifstream file(api_json_path);
        if (!file.is_open()) {
            GENAI_ERR("Failed to open API JSON file: %s", api_json_path.c_str());
            return false;
        }

        json api_json = json::parse(file);
        file.close();

        std::string yaml_content = convert_to_yaml(api_json, options);

        std::ofstream out_file(output_yaml_path);
        if (!out_file.is_open()) {
            GENAI_ERR("Failed to create output YAML file: %s", output_yaml_path.c_str());
            return false;
        }

        out_file << yaml_content;
        out_file.close();

        GENAI_INFO("[OK] Converted %s -> %s", api_json_path.c_str(), output_yaml_path.c_str());
        return true;

    } catch (const std::exception& e) {
        GENAI_ERR("Conversion error: %s", e.what());
        return false;
    }
}

std::string ComfyUIToGenAIConverter::convert_to_yaml(
    const json& api_json,
    const ConversionOptions& options) {

    PipelineParams params;
    return generate_yaml(api_json, params, options);
}

std::string ComfyUIToGenAIConverter::convert_to_yaml(
    const json& api_json,
    ov::AnyMap& pipeline_inputs,
    const ConversionOptions& options) {

    PipelineParams params;
    std::string yaml = generate_yaml(api_json, params, options);

    // Extract pipeline inputs directly from stored JSON nodes
    // CLIPTextEncode -> prompt and negative_prompt
    if (auto* nodes = params.get_nodes("CLIPTextEncode")) {
        for (const auto& node : *nodes) {
            if (!node.inputs.contains("text")) continue;

            if (node.title.find("Negative") != std::string::npos) {
                // First negative prompt found
                if (pipeline_inputs.find("negative_prompt") == pipeline_inputs.end()) {
                    pipeline_inputs["negative_prompt"] = node.inputs["text"].template get<std::string>();
                }
            } else {
                // First non-negative prompt found
                if (pipeline_inputs.find("prompt") == pipeline_inputs.end()) {
                    pipeline_inputs["prompt"] = node.inputs["text"].template get<std::string>();
                }
            }
        }
    }
    // KSampler -> cfg, steps, seed
    pipeline_inputs["guidance_scale"] = params.get_value<float>("KSampler", "cfg", 3.5f);
    pipeline_inputs["num_inference_steps"] = params.get_value<int>("KSampler", "steps", 4);
    pipeline_inputs["seed"] = params.get_value<int64_t>("KSampler", "seed", 0);

    // EmptySD3LatentImage -> width, height
    pipeline_inputs["width"] = params.get_value<int>("EmptySD3LatentImage", "width", 1024);
    pipeline_inputs["height"] = params.get_value<int>("EmptySD3LatentImage", "height", 1024);
    pipeline_inputs["batch_size"] = params.get_value<int>("EmptySD3LatentImage", "batch_size", 1);

    // FluxGuidance -> max_sequence_length
    pipeline_inputs["max_sequence_length"] = params.get_value<int>("FluxGuidance", "max_sequence_length", 512);

    // VAEDecodeSwitcher -> tile_size
    pipeline_inputs["tile_size"] = params.get_value<int>("VAEDecodeSwitcher", "tile_size", 0);

    return yaml;
}

std::string ComfyUIToGenAIConverter::generate_yaml(
    const json& api_json,
    PipelineParams& params,
    const ConversionOptions& options) {

    // Step 1: Store all nodes by type with their node_id_str for later access
    GENAI_INFO("Storing nodes by type...");
    for (const auto& [node_id, node_data] : api_json.items()) {
        std::string class_type = node_data["class_type"].get<std::string>();

        // Build NodeInfo
        NodeInfo node_info;
        node_info.class_type = class_type;
        node_info.inputs = node_data["inputs"];
        if (node_data.contains("_meta") && node_data["_meta"].contains("title")) {
            node_info.title = node_data["_meta"]["title"].get<std::string>();
        }

        // Generate node_id_str: format is "class_type_node_id" (e.g., "CLIPTextEncode_6")
        // Use "_" instead of "." to avoid YAML path separator issues
        // node_id is extracted from api_json (already added by workflow_json.cpp or api_json.cpp)
        std::string node_id_val = node_data.contains("node_id") ? node_data["node_id"].get<std::string>() : node_id;
        node_info.node_id_str = class_type + "_" + node_id_val;

        // Store node for later access
        params.store_node(class_type, node_info);
        GENAI_DEBUG("[NODE] Stored %s (id=%s, node_id_str=%s)", class_type.c_str(), node_id.c_str(), node_info.node_id_str.c_str());
    }

    // Step 2: Generate YAML directly from stored nodes
    GENAI_INFO("Generating YAML...");

    YAML::Node root;

    // Global context - determine model_type from UNETLoader's unet_name
    std::string model_type = "unknown";
    std::string unet_name = params.get_value<std::string>("UNETLoader", "unet_name", "");
    if (unet_name.find("z_image_turbo") != std::string::npos) {
        model_type = "zimage";
    }
    root["global_context"]["model_type"] = model_type;

    // Pipeline modules
    YAML::Node pipeline_modules = root["pipeline_modules"];

    // 1. ParameterModule - pipeline_params (always generated)
    {
        GENAI_DEBUG("[YAML] Adding ParameterModule (pipeline_params)");
        YAML::Node module = pipeline_modules["pipeline_params"];
        YAML::Node outputs;
        outputs.push_back(create_output_node("prompt", "String"));
        outputs.push_back(create_output_node("negative_prompt", "String"));
        outputs.push_back(create_output_node("guidance_scale", "Float"));
        outputs.push_back(create_output_node("max_sequence_length", "Int"));
        outputs.push_back(create_output_node("num_inference_steps", "Int"));
        outputs.push_back(create_output_node("width", "Int"));
        outputs.push_back(create_output_node("height", "Int"));
        outputs.push_back(create_output_node("batch_size", "Int"));
        outputs.push_back(create_output_node("seed", "Int"));
        module["outputs"] = outputs;
        module["type"] = "ParameterModule";
    }

    // 2. Initialize and get generator registry
    auto& generator_registry = YamlModuleGeneratorRegistry::instance();
    generator_registry.initialize_defaults();

    // 3. Iterate over api_json.items() and call each node's generator
    for (const auto& [node_id, node_data] : api_json.items()) {
        std::string class_type = node_data["class_type"].get<std::string>();

        // Get generator for this node type
        auto* generator = generator_registry.get_generator(class_type);
        if (!generator) {
            GENAI_DEBUG("[YAML] No generator for node type: %s (node_id=%s)", class_type.c_str(), node_id.c_str());
            continue;
        }

        // Get the stored NodeInfo for this node
        auto* nodes = params.get_nodes(class_type);
        if (!nodes) continue;

        // Find the specific node by matching node_id
        std::string node_id_val = node_data.contains("node_id") ? node_data["node_id"].get<std::string>() : node_id;
        std::string expected_node_id_str = class_type + "_" + node_id_val;

        for (const auto& node_info : *nodes) {
            if (node_info.node_id_str == expected_node_id_str) {
                // Create context and call generator
                YamlGeneratorContext ctx(pipeline_modules, root, params, options, node_info);
                generator->generate(ctx);
                break;
            }
        }
    }

    // ResultModule (always generated last)
    YamlModuleGeneratorRegistry::generate_result_module(pipeline_modules, root, params, options);

    GENAI_INFO("[YAML] YAML generation complete.");

    // Use YAML::Emitter to generate YAML string
    YAML::Emitter emitter;
    emitter << root;

    // Check if emitter succeeded
    if (!emitter.good()) {
        GENAI_ERR("YAML emitter failed: %s", emitter.GetLastError().c_str());
        return "";
    }

    // Dump YAML to file for debugging
    std::string yaml_content = emitter.c_str();

    // Check for empty content
    if (yaml_content.empty()) {
        GENAI_ERR("Generated YAML content is empty");
        return "";
    }

    // Dump YAML to file for debugging if DUMP_YAML environment variable is set
    if (std::getenv("DUMP_YAML")) {
        std::ofstream ofs("comfyui_generated_pipeline.yaml");
        if (ofs.is_open()) {
            ofs << yaml_content;
            GENAI_DEBUG("YAML dumped to: comfyui_generated_pipeline.yaml");
        }
    }

    return yaml_content;
}

}  // namespace comfyui
}  // namespace module
}  // namespace genai
}  // namespace ov
