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
#include <queue>
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
    int use_tiling = -1;  // -1 = auto

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

    auto it_use_tiling = pipeline_inputs.find("use_tiling");
    if (it_use_tiling != pipeline_inputs.end()) {
        try {
            use_tiling = it_use_tiling->second.as<bool>() ? 1 : 0;
        } catch (const ov::Exception&) {
            try {
                use_tiling = it_use_tiling->second.as<int>();
            } catch (const ov::Exception&) {
                GENAI_WARN("use_tiling has invalid type, using auto");
            }
        }
    }

    ConversionOptions options;
    options.model_path = model_path;
    options.device = device;
    options.tile_size = tile_size;
    options.use_tiling = use_tiling;
    return options;
}

void log_parsed_nodes(const std::map<std::string, Node>& nodes) {
    GENAI_INFO("Found %zu nodes", nodes.size());
    for (const auto& [node_id, node] : nodes) {
        GENAI_DEBUG("  - Node %s: %s", node_id.c_str(), node.class_type.c_str());
    }
}

/**
 * @brief Topological sort of nodes in API JSON
 *
 * Similar to ComfyUI's TopologicalSort in graph.py, this function sorts nodes
 * by their dependencies using Kahn's algorithm. Nodes with no dependencies
 * are processed first, followed by nodes that depend on them.
 *
 * @param api_json The API JSON containing node definitions
 * @return Vector of node IDs in topological order (dependencies first)
 */
std::vector<std::string> topological_sort_nodes(const json& api_json) {
    // blockCount: number of nodes this node is directly blocked by (dependencies)
    // blocking: which nodes are blocked by this node (dependents)
    std::map<std::string, int> block_count;
    std::map<std::string, std::set<std::string>> blocking;

    // Initialize all nodes
    for (const auto& [node_id, node_data] : api_json.items()) {
        block_count[node_id] = 0;
        blocking[node_id] = {};
    }

    // Build dependency graph by analyzing inputs
    // In ComfyUI API JSON, a link is represented as [from_node_id, from_socket_index]
    for (const auto& [node_id, node_data] : api_json.items()) {
        if (!node_data.contains("inputs")) continue;

        for (const auto& [input_name, input_value] : node_data["inputs"].items()) {
            // Check if input is a link (array with 2 elements: [from_node_id, from_socket])
            if (input_value.is_array() && input_value.size() == 2) {
                std::string from_node_id = input_value[0].get<std::string>();

                // Add dependency: from_node_id -> node_id
                // node_id depends on from_node_id, so from_node_id blocks node_id
                if (blocking.find(from_node_id) != blocking.end()) {
                    if (blocking[from_node_id].find(node_id) == blocking[from_node_id].end()) {
                        blocking[from_node_id].insert(node_id);
                        block_count[node_id]++;
                        GENAI_DEBUG("[TOPO] %s[%s] -> %s (block_count=%d)",
                                   from_node_id.c_str(), input_name.c_str(), node_id.c_str(), block_count[node_id]);
                    }
                }
            }
        }
    }

    // Topological sort using Kahn's algorithm
    std::vector<std::string> sorted_node_ids;
    std::queue<std::string> ready_queue;

    // Find all nodes with no dependencies (block_count == 0)
    for (const auto& [node_id, count] : block_count) {
        if (count == 0) {
            ready_queue.push(node_id);
            GENAI_DEBUG("[TOPO] Initial ready node: %s", node_id.c_str());
        }
    }

    // Process nodes in topological order
    while (!ready_queue.empty()) {
        std::string node_id = ready_queue.front();
        ready_queue.pop();
        sorted_node_ids.push_back(node_id);

        // Unblock dependent nodes
        for (const auto& blocked_node_id : blocking[node_id]) {
            block_count[blocked_node_id]--;
            GENAI_DEBUG("[TOPO] Unblocking %s: block_count=%d", blocked_node_id.c_str(), block_count[blocked_node_id]);
            if (block_count[blocked_node_id] == 0) {
                ready_queue.push(blocked_node_id);
            }
        }
    }

    // Check for cycles
    if (sorted_node_ids.size() != api_json.size()) {
        GENAI_ERR("[TOPO] Dependency cycle detected! Sorted %zu nodes out of %zu",
                 sorted_node_ids.size(), api_json.size());
    }

    // Log sorted order
    GENAI_INFO("[TOPO] Topological order (%zu nodes):", sorted_node_ids.size());
    for (size_t i = 0; i < sorted_node_ids.size(); ++i) {
        const auto& nid = sorted_node_ids[i];
        std::string class_type = api_json[nid]["class_type"].get<std::string>();
        GENAI_DEBUG("[TOPO]   %zu: %s (%s)", i, nid.c_str(), class_type.c_str());
    }

    return sorted_node_ids;
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

    // Latent image/video nodes -> width, height, batch_size, num_frames
    // Note: Only one latent node type should be present per workflow
    if (params.get_node("EmptySD3LatentImage")) {
        pipeline_inputs["width"] = params.get_value<int>("EmptySD3LatentImage", "width", 1024);
        pipeline_inputs["height"] = params.get_value<int>("EmptySD3LatentImage", "height", 1024);
        pipeline_inputs["batch_size"] = params.get_value<int>("EmptySD3LatentImage", "batch_size", 1);
    } else if (params.get_node("EmptyHunyuanLatentVideo")) {
        pipeline_inputs["width"] = params.get_value<int>("EmptyHunyuanLatentVideo", "width", 1024);
        pipeline_inputs["height"] = params.get_value<int>("EmptyHunyuanLatentVideo", "height", 1024);
        pipeline_inputs["batch_size"] = params.get_value<int>("EmptyHunyuanLatentVideo", "batch_size", 1);
        pipeline_inputs["num_frames"] = params.get_value<int>("EmptyHunyuanLatentVideo", "length", 16);
    }

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
    if (unet_name.find("wan2.1") != std::string::npos) {
        model_type = "wan2.1";
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
        if (model_type == "wan2.1") {
            outputs.push_back(create_output_node("num_frames", "Int"));
        }
        outputs.push_back(create_output_node("batch_size", "Int"));
        outputs.push_back(create_output_node("seed", "Int"));
        module["outputs"] = outputs;
        module["type"] = "ParameterModule";
    }

    // 2. Initialize and get generator registry
    auto& generator_registry = YamlModuleGeneratorRegistry::instance();
    generator_registry.initialize_defaults();

    // 3. Topological sort of nodes (similar to ComfyUI's TopologicalSort in graph.py)
    std::vector<std::string> sorted_node_ids = topological_sort_nodes(api_json);

    // 4. Iterate over sorted nodes and call each node's generator
    for (const auto& node_id : sorted_node_ids) {
        const auto& node_data = api_json[node_id];
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
                YamlGeneratorContext ctx(pipeline_modules, root, params, options, node_info, model_type);
                generator->generate(ctx);
                GENAI_DEBUG("[YAML] Generated module for %s (%s)", node_id.c_str(), class_type.c_str());
                break;
            }
        }
    }

    // ResultModule (always generated last)
    YamlModuleGeneratorRegistry::generate_result_module(pipeline_modules, root, params, options, model_type);

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
