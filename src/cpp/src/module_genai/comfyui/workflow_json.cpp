// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file workflow_to_api_converter_new.cpp
 * @brief Implementation of WorkflowToApiConverter class
 */

#include "comfyui.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include "logger.hpp"

namespace ov {
namespace genai {
namespace module {
namespace comfyui {

// ============================================================================
// Helper Functions
// ============================================================================

// Pretty names for node titles when title is not specified in workflow JSON
static const std::unordered_map<std::string, std::string>& get_pretty_names() {
    static const std::unordered_map<std::string, std::string> pretty_names = {
        {"SaveImage", "Save Image"},
        {"UNETLoader", "Load Diffusion Model"},
        {"VAELoader", "Load VAE"},
        {"CLIPLoader", "Load CLIP"},
        {"VAEDecodeSwitcher", "VAE Decode Switch"},
        {"StringConcatenate", "Concatenate"},
        {"VAEDecode", "VAE Decode"},
        {"CLIPTextEncode", "CLIP Text Encode (Prompt)"},
        {"EmptyHunyuanLatentVideo", "Empty HunyuanVideo 1.0 Latent"},
    };
    return pretty_names;
}

bool WorkflowToApiConverter::is_uuid(const std::string& str) {
    // Simple check: UUID contains hyphens and is 36 characters
    return str.find('-') != std::string::npos && str.length() == 36;
}

std::vector<std::string> WorkflowToApiConverter::get_widget_input_names(
    const std::string& node_type, size_t widget_count) {

    // Common node types and their widget parameters
    static const std::unordered_map<std::string, std::vector<std::string>> node_widgets = {
        {"SaveImage", {"filename_prefix"}},
        {"SaveAnimatedWEBP", {"filename_prefix", "fps", "lossless", "quality", "method"}},
        {"SaveWEBM", {"filename_prefix", "codec", "fps", "crf"}},
        {"KSampler", {"seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"}},
        {"CLIPTextEncode", {"text"}},
        {"CheckpointLoaderSimple", {"ckpt_name"}},
        {"EmptyLatentImage", {"width", "height", "batch_size"}},
        {"EmptySD3LatentImage", {"width", "height", "batch_size"}},
        {"EmptyHunyuanLatentVideo", {"width", "height", "length", "batch_size"}},
        {"VAEDecode", {}},
        {"VAELoader", {"vae_name"}},
        {"UNETLoader", {"unet_name", "weight_dtype"}},
        {"CLIPLoader", {"clip_name", "type", "device"}},
        {"ModelSamplingFlux", {"max_shift", "base_shift", "width", "height"}},
        {"ModelSamplingAuraFlow", {"shift"}},
        {"ModelSamplingSD3", {"shift"}},
        {"VAEDecodeSwitcher", {"select_decoder", "tile_size", "overlap", "temporal_size", "temporal_overlap"}},
        {"PrimitiveStringMultiline", {"value"}},
        {"StringConcatenate", {"string_a", "string_b", "delimiter"}},
        {"ConditioningZeroOut", {}},
    };

    auto it = node_widgets.find(node_type);
    if (it != node_widgets.end()) {
        return it->second;
    }

    // Default: generate generic names
    std::vector<std::string> names;
    for (size_t i = 0; i < widget_count; ++i) {
        names.push_back("param_" + std::to_string(i));
    }
    return names;
}

// ============================================================================
// Subgraph Expansion
// ============================================================================

void WorkflowToApiConverter::expand_subgraph(
    const nlohmann::json& parent_node,
    const nlohmann::json& subgraph,
    std::vector<std::pair<std::string, json>>& node_list,
    const std::unordered_map<int, nlohmann::json>& link_map,
    const std::unordered_map<int, int>& bypass_map) {

    if (!parent_node.contains("id") || !subgraph.contains("nodes")) {
        return;
    }

    int parent_id = parent_node["id"];

    // Build internal link map for subgraph
    std::unordered_map<int, nlohmann::json> subgraph_link_map;
    if (subgraph.contains("links") && subgraph["links"].is_array()) {
        for (const auto& link_obj : subgraph["links"]) {
            if (link_obj.is_object() && link_obj.contains("id")) {
                int link_id = link_obj["id"];
                subgraph_link_map[link_id] = link_obj;
            }
        }
    }

    // Get parent node's widgets_values
    std::vector<nlohmann::json> parent_widgets;
    if (parent_node.contains("widgets_values") && parent_node["widgets_values"].is_array()) {
        for (const auto& widget : parent_node["widgets_values"]) {
            parent_widgets.push_back(widget);
        }
    }

    // Build map from subgraph input link IDs to parent widgets or external connections
    std::unordered_map<int, nlohmann::json> input_widget_map;
    std::unordered_map<int, std::pair<int, int>> input_connection_map;

    if (subgraph.contains("inputs") && subgraph["inputs"].is_array()) {
        const auto& inputs = subgraph["inputs"];
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (!inputs[i].contains("linkIds") || !inputs[i]["linkIds"].is_array()) continue;
            if (!inputs[i].contains("name")) continue;

            std::string input_name = inputs[i]["name"].get<std::string>();

            for (const auto& link_id : inputs[i]["linkIds"]) {
                if (!link_id.is_number_integer()) continue;
                int link_id_val = link_id.get<int>();

                // Check if parent node has a corresponding input connection
                bool found_connection = false;
                if (parent_node.contains("inputs") && parent_node["inputs"].is_array()) {
                    for (const auto& parent_input : parent_node["inputs"]) {
                        if (!parent_input.contains("name") || !parent_input.contains("link")) continue;
                        if (parent_input["link"].is_null()) continue;

                        std::string parent_input_name = parent_input["name"].get<std::string>();
                        if (parent_input_name == input_name) {
                            int parent_link_id = parent_input["link"];
                            if (link_map.find(parent_link_id) != link_map.end()) {
                                const auto& parent_link = link_map.at(parent_link_id);
                                if (parent_link.is_array() && parent_link.size() >= 3) {
                                    int from_node = parent_link[1];
                                    int from_slot = parent_link[2];
                                    input_connection_map[link_id_val] = {from_node, from_slot};
                                    found_connection = true;
                                    break;
                                }
                            }
                        }
                    }
                }

                // If no connection found, check for widget value
                if (!found_connection && i < parent_widgets.size()) {
                    input_widget_map[link_id_val] = parent_widgets[i];
                }
            }
        }
    }

    // Process each node in subgraph
    if (!subgraph["nodes"].is_array()) {
        return;
    }

    for (const auto& sub_node : subgraph["nodes"]) {
        if (!sub_node.contains("id") || !sub_node.contains("type")) {
            continue;
        }

        // Skip bypassed nodes
        if (sub_node.contains("mode") && sub_node["mode"].is_number() && sub_node["mode"] == 4) {
            continue;
        }

        int sub_node_id = sub_node["id"];
        std::string sub_node_type = sub_node["type"];
        std::string composite_id = std::to_string(parent_id) + ":" + std::to_string(sub_node_id);

        // Create API node
        json api_node = json::object();
        json inputs = json::object();

        // Process widgets_values
        std::vector<std::pair<std::string, nlohmann::json>> widget_inputs;
        if (sub_node.contains("widgets_values") && sub_node["widgets_values"].is_array()) {
            const auto& widgets = sub_node["widgets_values"];

            // Special handling for KSampler
            if (sub_node_type == "KSampler" && widgets.size() >= 7) {
                std::vector<std::string> param_names = {"seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"};
                std::vector<size_t> widget_indices = {0, 2, 3, 4, 5, 6};

                for (size_t i = 0; i < param_names.size() && i < widget_indices.size(); ++i) {
                    size_t widget_idx = widget_indices[i];
                    if (widget_idx < widgets.size()) {
                        widget_inputs.push_back({param_names[i], widgets[widget_idx]});
                    }
                }
            } else {
                auto input_names = get_widget_input_names(sub_node_type, widgets.size());
                for (size_t i = 0; i < widgets.size() && i < input_names.size(); ++i) {
                    widget_inputs.push_back({input_names[i], widgets[i]});
                }
            }
        }

        // Add widget inputs
        for (const auto& [name, value] : widget_inputs) {
            inputs[name] = value;
        }

        // Process connections
        if (sub_node.contains("inputs") && sub_node["inputs"].is_array()) {
            for (const auto& input : sub_node["inputs"]) {
                if (!input.contains("name")) continue;

                std::string input_name = input["name"];

                if (input.contains("link") && !input["link"].is_null()) {
                    int link_id = input["link"];

                    // Check if this is an input from parent node's external connection
                    if (input_connection_map.find(link_id) != input_connection_map.end()) {
                        auto [from_node, from_slot] = input_connection_map[link_id];
                        inputs[input_name] = json::array();
                        inputs[input_name].push_back(std::to_string(from_node));
                        inputs[input_name].push_back(from_slot);
                        continue;
                    }

                    // Check if this is an input from parent node's widget
                    if (input_widget_map.find(link_id) != input_widget_map.end()) {
                        inputs[input_name] = input_widget_map[link_id];
                        continue;
                    }

                    // Otherwise, resolve from subgraph links
                    if (subgraph_link_map.find(link_id) != subgraph_link_map.end()) {
                        const auto& link = subgraph_link_map[link_id];
                        if (link.is_object() && link.contains("origin_id") && link.contains("origin_slot")) {
                            int from_node = link["origin_id"];
                            int from_slot = link["origin_slot"];

                            std::string from_composite_id = std::to_string(parent_id) + ":" + std::to_string(from_node);

                            if (!inputs.contains(input_name)) {
                                inputs[input_name] = json::array();
                                inputs[input_name].push_back(from_composite_id);
                                inputs[input_name].push_back(from_slot);
                            }
                        }
                    }
                }
            }
        }

        // Get title with pretty name mapping
        std::string title;
        if (sub_node.contains("title")) {
            title = sub_node["title"];
        } else {
            const auto& pretty_names = get_pretty_names();
            auto it = pretty_names.find(sub_node_type);
            if (it != pretty_names.end()) {
                title = it->second;
            } else {
                title = sub_node_type;
            }
        }

        // Build api_node in correct order
        api_node["inputs"] = inputs;
        api_node["class_type"] = sub_node_type;
        api_node["_meta"] = json::object();
        api_node["_meta"]["title"] = title;

        GENAI_DEBUG("[WORKFLOW] Subgraph node: node_id=%s, class_type=%s, title=%s",
                    composite_id.c_str(), sub_node_type.c_str(), title.c_str());

        node_list.push_back({composite_id, api_node});
    }
}

// ============================================================================
// Main Conversion Functions
// ============================================================================

json WorkflowToApiConverter::convert(const nlohmann::json& workflow_json) {
    auto node_list = convert_to_node_map(workflow_json);

    // Manually build JSON string to preserve order
    std::ostringstream oss;
    oss << "{\n";
    bool first = true;
    for (const auto& pair : node_list) {
        if (!first) {
            oss << ",\n";
        }
        first = false;

        oss << "  \"" << pair.first << "\": ";
        std::string node_str = pair.second.dump(2);

        std::istringstream node_stream(node_str);
        std::string line;
        bool first_line = true;
        while (std::getline(node_stream, line)) {
            if (!first_line) {
                oss << "\n  ";
            }
            first_line = false;
            oss << line;
        }
    }
    oss << "\n}\n";

    return json::parse(oss.str());
}

std::vector<std::pair<std::string, json>> WorkflowToApiConverter::convert_to_node_map(
    const nlohmann::json& workflow_json) {

    std::vector<std::pair<std::string, json>> node_list;

    try {
        if (!workflow_json.contains("nodes") || !workflow_json["nodes"].is_array()) {
            GENAI_ERR("Invalid workflow JSON: missing 'nodes' array");
            return std::vector<std::pair<std::string, json>>();
        }

        // Parse definitions.subgraphs if present
        std::unordered_map<std::string, nlohmann::json> subgraph_map;
        if (workflow_json.contains("definitions") && workflow_json["definitions"].is_object()) {
            const auto& definitions = workflow_json["definitions"];
            if (definitions.contains("subgraphs") && definitions["subgraphs"].is_array()) {
                for (const auto& subgraph : definitions["subgraphs"]) {
                    if (subgraph.contains("id") && subgraph["id"].is_string()) {
                        std::string subgraph_id = subgraph["id"];
                        subgraph_map[subgraph_id] = subgraph;
                    }
                }
            }
        }

        // Build link map: link_id -> [from_node, from_slot, to_node, to_slot, type]
        std::unordered_map<int, nlohmann::json> link_map;
        if (workflow_json.contains("links") && workflow_json["links"].is_array()) {
            for (const auto& link_obj : workflow_json["links"]) {
                const auto* link_ptr = &link_obj;
                if (link_obj.is_object() && link_obj.contains("value")) {
                    link_ptr = &link_obj["value"];
                }

                const auto& link = *link_ptr;
                if (link.is_array() && link.size() >= 5) {
                    int link_id = link[0];
                    link_map[link_id] = link;
                }
            }
        }

        // Identify bypassed nodes (mode == 4)
        std::unordered_map<int, int> bypass_map;
        for (const auto& node : workflow_json["nodes"]) {
            if (node.contains("mode") && node["mode"].is_number() && node["mode"] == 4) {
                if (node.contains("id") && node.contains("inputs") && node["inputs"].is_array()) {
                    int bypassed_node_id = node["id"];
                    for (const auto& input : node["inputs"]) {
                        if (input.contains("link") && !input["link"].is_null()) {
                            int link_id = input["link"];
                            if (link_map.find(link_id) != link_map.end()) {
                                const auto& link = link_map[link_id];
                                if (link.is_array() && link.size() >= 2) {
                                    int from_node = link[1];
                                    bypass_map[bypassed_node_id] = from_node;
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Build subgraph output mapping
        std::unordered_map<int, std::unordered_map<int, std::string>> subgraph_output_map;
        for (const auto& node : workflow_json["nodes"]) {
            if (!node.contains("id") || !node.contains("type")) continue;
            std::string node_type = node["type"];
            if (is_uuid(node_type) && subgraph_map.find(node_type) != subgraph_map.end()) {
                int parent_id = node["id"];
                const auto& subgraph = subgraph_map[node_type];

                if (subgraph.contains("outputs") && subgraph["outputs"].is_array()) {
                    for (const auto& output : subgraph["outputs"]) {
                        if (!output.contains("linkIds") || !output["linkIds"].is_array()) continue;

                        for (const auto& link_id : output["linkIds"]) {
                            if (!link_id.is_number_integer()) continue;
                            int output_link_id = link_id.get<int>();

                            if (subgraph.contains("links") && subgraph["links"].is_array()) {
                                for (const auto& link_obj : subgraph["links"]) {
                                    if (link_obj.is_object() && link_obj.contains("id") && link_obj["id"] == output_link_id) {
                                        if (link_obj.contains("origin_id") && link_obj.contains("origin_slot")) {
                                            int origin_node = link_obj["origin_id"];
                                            int origin_slot = link_obj["origin_slot"];

                                            if (!subgraph_output_map[parent_id].count(origin_slot)) {
                                                std::string composite_id = std::to_string(parent_id) + ":" + std::to_string(origin_node);
                                                subgraph_output_map[parent_id][origin_slot] = composite_id;
                                            }
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Process each node
        for (const auto& node : workflow_json["nodes"]) {
            if (!node.contains("id") || !node.contains("type")) {
                continue;
            }

            // Skip nodes with mode 4 (bypass/muted)
            if (node.contains("mode") && node["mode"].is_number() && node["mode"] == 4) {
                continue;
            }

            // Skip non-executable nodes
            std::string node_type = node["type"];
            if (node_type == "Note" || node_type == "Reroute" || node_type == "MarkdownNote") {
                continue;
            }

            int node_id = node["id"];
            std::string node_id_str = std::to_string(node_id);

            // Check if this node is a subgraph reference
            if (is_uuid(node_type) && subgraph_map.find(node_type) != subgraph_map.end()) {
                expand_subgraph(node, subgraph_map[node_type], node_list, link_map, bypass_map);
                continue;
            }

            // Create API node
            json api_node = json::object();
            json inputs = json::object();

            // Get title
            std::string title;
            if (node.contains("title")) {
                title = node["title"];
            } else {
                const auto& pretty_names = get_pretty_names();
                auto it = pretty_names.find(node_type);
                if (it != pretty_names.end()) {
                    title = it->second;
                } else {
                    title = node_type;
                }
            }

            // Process widgets_values
            std::vector<std::pair<std::string, nlohmann::json>> widget_inputs;
            nlohmann::json delimiter_value;
            bool has_delimiter = false;

            if (node.contains("widgets_values") && node["widgets_values"].is_array()) {
                const auto& widgets = node["widgets_values"];

                if (node_type == "KSampler" && widgets.size() >= 7) {
                    std::vector<std::string> param_names = {"seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"};
                    std::vector<size_t> widget_indices = {0, 2, 3, 4, 5, 6};

                    for (size_t i = 0; i < param_names.size() && i < widget_indices.size(); ++i) {
                        size_t widget_idx = widget_indices[i];
                        if (widget_idx < widgets.size()) {
                            widget_inputs.push_back({param_names[i], widgets[widget_idx]});
                        }
                    }
                } else if (node_type == "StringConcatenate" && widgets.size() >= 3) {
                    widget_inputs.push_back({"string_a", widgets[0]});
                    delimiter_value = widgets[2];
                    has_delimiter = true;
                } else {
                    auto input_names = get_widget_input_names(node_type, widgets.size());
                    for (size_t i = 0; i < widgets.size() && i < input_names.size(); ++i) {
                        widget_inputs.push_back({input_names[i], widgets[i]});
                    }
                }
            }

            // Add widget inputs first
            for (const auto& [name, value] : widget_inputs) {
                inputs[name] = value;
            }

            // Process connections
            if (node.contains("inputs") && node["inputs"].is_array()) {
                for (const auto& input : node["inputs"]) {
                    if (!input.contains("name")) continue;

                    std::string input_name = input["name"];

                    if (input.contains("link") && !input["link"].is_null()) {
                        int link_id = input["link"];

                        if (link_map.find(link_id) != link_map.end()) {
                            const auto& link = link_map[link_id];
                            if (!link.is_array() || link.size() < 3) {
                                GENAI_WARN("Warning: Invalid link format for link_id %d", link_id);
                                continue;
                            }

                            int from_node = link[1];
                            int from_slot = link[2];

                            // Check if from_node is a subgraph parent node
                            if (subgraph_output_map.find(from_node) != subgraph_output_map.end()) {
                                const auto& slot_map = subgraph_output_map[from_node];
                                if (slot_map.find(from_slot) != slot_map.end()) {
                                    std::string composite_from = slot_map.at(from_slot);
                                    inputs[input_name] = json::array();
                                    inputs[input_name].push_back(composite_from);
                                    inputs[input_name].push_back(from_slot);
                                    continue;
                                }
                            }

                            // Resolve bypassed nodes
                            std::unordered_set<int> visited;
                            while (bypass_map.find(from_node) != bypass_map.end()) {
                                if (visited.find(from_node) != visited.end()) {
                                    GENAI_WARN("Warning: Cycle detected in bypass chain for node %d", from_node);
                                    break;
                                }
                                visited.insert(from_node);
                                from_node = bypass_map[from_node];
                            }

                            if (!inputs.contains(input_name)) {
                                inputs[input_name] = json::array();
                                inputs[input_name].push_back(std::to_string(from_node));
                                inputs[input_name].push_back(from_slot);
                            }
                        }
                    }
                }
            }

            // For StringConcatenate, add delimiter after all connections
            if (has_delimiter) {
                inputs["delimiter"] = delimiter_value;
            }

            // Build api_node in correct order
            api_node["inputs"] = inputs;
            api_node["class_type"] = node_type;
            api_node["_meta"] = json::object();
            api_node["_meta"]["title"] = title;

            GENAI_DEBUG("[WORKFLOW] Regular node: node_id=%s, class_type=%s, title=%s",
                        node_id_str.c_str(), node_type.c_str(), title.c_str());

            node_list.push_back({node_id_str, api_node});
        }

    } catch (const std::exception& e) {
        GENAI_ERR("Exception in convert_to_node_map: %s", e.what());
        throw;
    }

    // Separate and sort nodes
    std::vector<std::pair<std::string, json>> regular_nodes;
    std::vector<std::pair<std::string, json>> subgraph_nodes;

    for (const auto& pair : node_list) {
        if (pair.first.find(':') == std::string::npos) {
            regular_nodes.push_back(pair);
        } else {
            subgraph_nodes.push_back(pair);
        }
    }

    // Sort regular nodes by numeric ID
    std::sort(regular_nodes.begin(), regular_nodes.end(),
        [](const std::pair<std::string, json>& a, const std::pair<std::string, json>& b) {
            return std::stoi(a.first) < std::stoi(b.first);
        });

    // Combine: regular nodes first, then subgraph nodes
    std::vector<std::pair<std::string, json>> result;
    result.insert(result.end(), regular_nodes.begin(), regular_nodes.end());
    result.insert(result.end(), subgraph_nodes.begin(), subgraph_nodes.end());

    return result;
}

bool WorkflowToApiConverter::convert_file(const std::string& workflow_path, const std::string& api_path) {
    std::ifstream input_file(workflow_path);
    if (!input_file.is_open()) {
        GENAI_ERR("Failed to open workflow file: %s", workflow_path.c_str());
        return false;
    }

    nlohmann::json workflow_json;
    try {
        input_file >> workflow_json;
    } catch (const std::exception& e) {
        GENAI_ERR("Failed to parse workflow JSON: %s", e.what());
        return false;
    }
    input_file.close();

    auto node_list = convert_to_node_map(workflow_json);

    std::ofstream output_file(api_path);
    if (!output_file.is_open()) {
        GENAI_ERR("Failed to create API file: %s", api_path.c_str());
        return false;
    }

    // Write JSON to preserve insertion order
    output_file << "{\n";
    bool first = true;
    for (const auto& pair : node_list) {
        if (!first) {
            output_file << ",\n";
        }
        first = false;

        output_file << "  \"" << pair.first << "\": ";
        std::string node_str = pair.second.dump(2);

        std::istringstream node_stream(node_str);
        std::string line;
        bool first_line = true;
        while (std::getline(node_stream, line)) {
            if (!first_line) {
                output_file << "\n  ";
            }
            first_line = false;
            output_file << line;
        }
    }
    output_file << "\n}\n";
    output_file.close();

    return true;
}

}  // namespace comfyui
}  // namespace module
}  // namespace genai
}  // namespace ov
