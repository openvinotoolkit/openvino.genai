// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file comfyui_json_parser.cpp
 * @brief Implementation of ComfyUIJsonParser class
 */

#include "comfyui.hpp"
#include "node_class_registry.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include "logger.hpp"

namespace ov {
namespace genai {
namespace module {
namespace comfyui {

/**
 * @brief Constructs a ComfyUIJsonParser and initializes default node class mappings.
 *
 * The parser supports both ComfyUI API JSON format and Workflow JSON format.
 * Default node classes (KSampler, CLIPTextEncode, SaveImage, etc.) are registered
 * during construction to enable validation of common node types.
 */
ComfyUIJsonParser::ComfyUIJsonParser() : json_format_(JsonFormat::UNKNOWN) {
    // Load node class mappings from the centralized registry
    auto& registry = NodeClassRegistry::instance();
    registry.initialize_defaults();
    node_class_mappings_ = registry.get_mappings();
}

ComfyUIJsonParser::~ComfyUIJsonParser() = default;

// ============================================================================
// JSON Loading and Parsing
// ============================================================================

bool ComfyUIJsonParser::is_workflow_json(const json& j) const {
    return j.is_object() && j.contains("nodes") && j["nodes"].is_array();
}

bool ComfyUIJsonParser::load_json_file(const std::filesystem::path& file_path) {
    try {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            GENAI_ERR("Failed to open file: %s", file_path.string().c_str());
            return false;
        }

        json source_json = json::parse(file);
        bool result = process_json(source_json);

        // Debug: Save converted API JSON to file if it was workflow format
        if (result && json_format_ == JsonFormat::WORKFLOW) {
            std::filesystem::path debug_filepath = file_path.parent_path() / (file_path.stem().string() + "_converted_api.json");
            std::ofstream debug_file(debug_filepath);
            if (debug_file.is_open()) {
                debug_file << prompt_.dump(2);
                debug_file.close();
                std::cout << "Saved converted API JSON to " << debug_filepath.string() << std::endl;
                GENAI_DEBUG("Saved converted API JSON to %s", debug_filepath.string().c_str());
            }
        }

        return result;
    } catch (const json::exception& e) {
        GENAI_DEBUG("JSON parsing error: %s", e.what());
        return false;
    }
}

bool ComfyUIJsonParser::parse_json_string(const std::string& json_str) {
    try {
        if (json_str.empty()) {
            GENAI_DEBUG("Empty JSON string provided");
            return false;
        }

        json source_json = json::parse(json_str);
        return process_json(source_json);
    } catch (const json::exception& e) {
        GENAI_DEBUG("JSON parsing error: %s", e.what());
        return false;
    }
}

bool ComfyUIJsonParser::process_json(const json& source_json) {
    // Check if it's workflow JSON and convert if needed
    if (is_workflow_json(source_json)) {
        GENAI_INFO("Detected workflow JSON format, converting to API format...");
        json_format_ = JsonFormat::WORKFLOW;
        prompt_ = WorkflowToApiConverter::convert(source_json);
        GENAI_INFO("Workflow converted to API format");
    } else {
        GENAI_INFO("Detected API JSON format");
        json_format_ = JsonFormat::API;
        prompt_ = source_json;
    }

    // Parse nodes from API JSON
    nodes_.clear();
    for (auto& [key, value] : prompt_.items()) {
        if (!value.contains("class_type")) {
            GENAI_WARN("Node %s missing class_type, skipping", key.c_str());
            continue;
        }
        Node node;
        node.class_type = value["class_type"].get<std::string>();
        node.inputs = value["inputs"];
        if (value.contains("_meta")) {
            node.meta = value["_meta"];
        }
        nodes_[key] = node;
    }
    GENAI_DEBUG("Parsed %zu nodes", nodes_.size());

    return true;
}

// ============================================================================
// Validation
// ============================================================================

PromptValidationResult ComfyUIJsonParser::validate_prompt(const std::string& prompt_id) {
    PromptValidationResult result;
    result.success = true;

    // Debug: Print all nodes
    GENAI_DEBUG("Total nodes: %zu", nodes_.size());
    for (const auto& [node_id, node] : nodes_) {
        GENAI_DEBUG("Node ID: %s, Class Type: %s", node_id.c_str(), node.class_type.c_str());
    }

    // Check that all nodes have class_type
    for (const auto& [node_id, node] : nodes_) {
        if (node.class_type.empty()) {
            result.success = false;
            result.error.type = "invalid_prompt";
            result.error.message = "Cannot execute because a node is missing the class_type property.";
            result.error.details = "Node ID '#" + node_id + "'";
            return result;
        }

        // Check if class_type exists in mappings
        if (node_class_mappings_.find(node.class_type) == node_class_mappings_.end()) {
            result.success = false;
            result.error.type = "invalid_prompt";
            result.error.message = "Cannot execute because node " + node.class_type + " does not exist.";
            result.error.details = "Node ID '#" + node_id + "'";
            return result;
        }
    }

    // Find output nodes
    std::set<std::string> outputs;
    for (const auto& [node_id, node] : nodes_) {
        const auto& class_info = node_class_mappings_[node.class_type];
        if (class_info.is_output_node) {
            outputs.insert(node_id);
        }
    }

    if (outputs.empty()) {
        result.success = false;
        result.error.type = "prompt_no_outputs";
        result.error.message = "Prompt has no outputs";
        result.error.details = "";
        return result;
    }

    // Validate each output node
    std::set<std::string> good_outputs;
    std::map<std::string, NodeValidationResult> validated;

    for (const auto& output_node_id : outputs) {
        try {
            auto node_result = validate_node_inputs(prompt_id, output_node_id, validated);

            if (node_result.valid) {
                good_outputs.insert(output_node_id);
            } else {
                result.output_errors.push_back({output_node_id, node_result.errors});

                for (const auto& [nid, vres] : validated) {
                    if (!vres.valid && !vres.errors.empty()) {
                        if (result.node_errors.find(nid) == result.node_errors.end()) {
                            json node_error;
                            node_error["errors"] = json::array();
                            for (const auto& err : vres.errors) {
                                json jerr;
                                jerr["type"] = err.type;
                                jerr["message"] = err.message;
                                jerr["details"] = err.details;
                                jerr["extra_info"] = err.extra_info;
                                node_error["errors"].push_back(jerr);
                            }
                            node_error["dependent_outputs"] = json::array();
                            node_error["class_type"] = nodes_[nid].class_type;
                            result.node_errors[nid] = node_error;
                        }
                        result.node_errors[nid]["dependent_outputs"].push_back(output_node_id);
                    }
                }
            }
        } catch (const std::exception& e) {
            result.output_errors.push_back({output_node_id, {{
                "exception_during_validation",
                "Exception when validating node",
                std::string(e.what()),
                {}
            }}});
        }
    }

    if (good_outputs.empty()) {
        result.success = false;
        result.error.type = "prompt_outputs_failed_validation";
        result.error.message = "Prompt outputs failed validation";

        std::stringstream details;
        for (const auto& [output_id, errors] : result.output_errors) {
            for (const auto& err : errors) {
                details << err.message << ": " << err.details << "\n";
            }
        }
        result.error.details = details.str();
        return result;
    }

    return result;
}

NodeValidationResult ComfyUIJsonParser::validate_node_inputs(
    const std::string& prompt_id,
    const std::string& node_id,
    std::map<std::string, NodeValidationResult>& validated) {

    if (validated.find(node_id) != validated.end()) {
        return validated[node_id];
    }

    NodeValidationResult result;
    result.valid = true;
    result.node_id = node_id;

    const auto& node = nodes_[node_id];
    const auto& class_info = node_class_mappings_[node.class_type];

    // Check required inputs
    for (const auto& [input_name, input_info] : class_info.required_inputs) {
        if (node.inputs.find(input_name) == node.inputs.end()) {
            result.valid = false;
            ValidationError error;
            error.type = "required_input_missing";
            error.message = "Required input is missing";
            error.details = input_name;
            error.extra_info["input_name"] = input_name;
            result.errors.push_back(error);
            continue;
        }
    }

    // Validate all inputs
    std::map<std::string, InputTypeInfo> all_inputs = class_info.required_inputs;
    all_inputs.insert(class_info.optional_inputs.begin(), class_info.optional_inputs.end());

    for (const auto& [input_name, input_info] : all_inputs) {
        if (node.inputs.find(input_name) == node.inputs.end()) {
            continue;
        }

        const auto& input_value = node.inputs.at(input_name);

        // Check if it's a linked input
        if (input_value.is_array()) {
            if (input_value.size() != 2) {
                result.valid = false;
                ValidationError error;
                error.type = "bad_linked_input";
                error.message = "Bad linked input, must be a length-2 list of [node_id, slot_index]";
                error.details = input_name;
                error.extra_info["input_name"] = input_name;
                error.extra_info["received_value"] = input_value;
                result.errors.push_back(error);
                continue;
            }

            std::string linked_node_id = input_value[0].get<std::string>();
            int slot_index = input_value[1].get<int>();

            if (nodes_.find(linked_node_id) == nodes_.end()) {
                result.valid = false;
                ValidationError error;
                error.type = "invalid_linked_node";
                error.message = "Linked node does not exist";
                error.details = "Node " + node_id + " input '" + input_name + "' links to non-existent node " + linked_node_id;
                error.extra_info["input_name"] = input_name;
                error.extra_info["linked_node"] = input_value;
                result.errors.push_back(error);
                continue;
            }

            const auto& linked_node = nodes_[linked_node_id];
            const auto& linked_class_info = node_class_mappings_[linked_node.class_type];

            if (slot_index < 0 || slot_index >= static_cast<int>(linked_class_info.return_types.size())) {
                result.valid = false;
                ValidationError error;
                error.type = "invalid_slot_index";
                error.message = "Slot index out of range";
                error.details = "Node " + linked_node_id + " output slot " + std::to_string(slot_index);
                error.extra_info["input_name"] = input_name;
                error.extra_info["linked_node"] = input_value;
                result.errors.push_back(error);
                continue;
            }

            std::string received_type = linked_class_info.return_types[slot_index];

            if (!validate_node_input(received_type, input_info.type)) {
                result.valid = false;
                ValidationError error;
                error.type = "return_type_mismatch";
                error.message = "Return type mismatch between linked nodes";
                error.details = input_name + ", received_type(" + received_type + ") mismatch input_type(" + input_info.type + ")";
                error.extra_info["input_name"] = input_name;
                error.extra_info["received_type"] = received_type;
                error.extra_info["linked_node"] = input_value;
                result.errors.push_back(error);
                continue;
            }

            // Recursively validate linked node
            try {
                auto linked_result = validate_node_inputs(prompt_id, linked_node_id, validated);
                if (!linked_result.valid) {
                    result.valid = false;
                }
            } catch (const std::exception& e) {
                result.valid = false;
                ValidationError error;
                error.type = "exception_during_inner_validation";
                error.message = "Exception when validating inner node";
                error.details = std::string(e.what());
                error.extra_info["input_name"] = input_name;
                error.extra_info["linked_node"] = input_value;
                result.errors.push_back(error);
            }
        } else {
            // Direct value input - validate type
            bool type_valid = true;

            try {
                if (input_info.type == "INT") {
                    if (!input_value.is_number_integer()) {
                        type_valid = false;
                    } else {
                        int64_t val = input_value.get<int64_t>();
                        if (input_info.extra_info.find("min") != input_info.extra_info.end()) {
                            int64_t min_val = std::get<int>(input_info.extra_info.at("min"));
                            if (val < min_val) {
                                result.valid = false;
                                ValidationError error;
                                error.type = "value_smaller_than_min";
                                error.message = "Value " + std::to_string(val) + " smaller than min of " + std::to_string(min_val);
                                error.details = input_name;
                                error.extra_info["input_name"] = input_name;
                                error.extra_info["received_value"] = input_value;
                                result.errors.push_back(error);
                            }
                        }
                        if (input_info.extra_info.find("max") != input_info.extra_info.end()) {
                            int64_t max_val = std::get<int>(input_info.extra_info.at("max"));
                            if (val > max_val) {
                                result.valid = false;
                                ValidationError error;
                                error.type = "value_bigger_than_max";
                                error.message = "Value " + std::to_string(val) + " bigger than max of " + std::to_string(max_val);
                                error.details = input_name;
                                error.extra_info["input_name"] = input_name;
                                error.extra_info["received_value"] = input_value;
                                result.errors.push_back(error);
                            }
                        }
                    }
                } else if (input_info.type == "FLOAT") {
                    if (!input_value.is_number()) {
                        type_valid = false;
                    } else {
                        double val = input_value.get<double>();
                        if (input_info.extra_info.find("min") != input_info.extra_info.end()) {
                            double min_val = std::get<double>(input_info.extra_info.at("min"));
                            if (val < min_val) {
                                result.valid = false;
                                ValidationError error;
                                error.type = "value_smaller_than_min";
                                error.message = "Value " + std::to_string(val) + " smaller than min of " + std::to_string(min_val);
                                error.details = input_name;
                                error.extra_info["input_name"] = input_name;
                                error.extra_info["received_value"] = input_value;
                                result.errors.push_back(error);
                            }
                        }
                        if (input_info.extra_info.find("max") != input_info.extra_info.end()) {
                            double max_val = std::get<double>(input_info.extra_info.at("max"));
                            if (val > max_val) {
                                result.valid = false;
                                ValidationError error;
                                error.type = "value_bigger_than_max";
                                error.message = "Value " + std::to_string(val) + " bigger than max of " + std::to_string(max_val);
                                error.details = input_name;
                                error.extra_info["input_name"] = input_name;
                                error.extra_info["received_value"] = input_value;
                                result.errors.push_back(error);
                            }
                        }
                    }
                } else if (input_info.type == "STRING") {
                    if (!input_value.is_string()) {
                        type_valid = false;
                    }
                } else if (input_info.type == "BOOLEAN") {
                    if (!input_value.is_boolean()) {
                        type_valid = false;
                    }
                }

                if (!type_valid) {
                    result.valid = false;
                    ValidationError error;
                    error.type = "invalid_input_type";
                    error.message = "Failed to convert an input value to a " + input_info.type + " value";
                    error.details = input_name + ", " + input_value.dump();
                    error.extra_info["input_name"] = input_name;
                    error.extra_info["received_value"] = input_value;
                    result.errors.push_back(error);
                }
            } catch (const std::exception& e) {
                result.valid = false;
                ValidationError error;
                error.type = "invalid_input_type";
                error.message = "Failed to validate input value";
                error.details = input_name + ", " + std::string(e.what());
                error.extra_info["input_name"] = input_name;
                error.extra_info["received_value"] = input_value;
                result.errors.push_back(error);
            }
        }
    }

    validated[node_id] = result;
    return result;
}

bool ComfyUIJsonParser::validate_node_input(const std::string& received_type, const std::string& input_type) {
    if (received_type == input_type) {
        return true;
    }

    // Allow wildcards
    if (input_type == "*" || received_type == "*") {
        return true;
    }

    return false;
}

// ============================================================================
// Node Class Registration
// ============================================================================

void ComfyUIJsonParser::register_node_class(
    const std::string& class_name,
    const std::map<std::string, InputTypeInfo>& required_inputs,
    const std::map<std::string, InputTypeInfo>& optional_inputs,
    const std::vector<std::string>& return_types,
    bool is_output_node) {

    NodeClassInfo info;
    info.required_inputs = required_inputs;
    info.optional_inputs = optional_inputs;
    info.return_types = return_types;
    info.is_output_node = is_output_node;
    node_class_mappings_[class_name] = info;
}

std::string ComfyUIJsonParser::get_validation_errors_string(const PromptValidationResult& validation_result) const {
    std::stringstream ss;

    // Get node title for the main error message if available
    std::string main_error_node_title;
    std::string details = validation_result.error.details;
    size_t start = details.find("#");
    size_t end = details.find("'", start);
    if (start != std::string::npos && end != std::string::npos) {
        std::string error_node_id = details.substr(start + 1, end - start - 1);
        auto it = nodes_.find(error_node_id);
        if (it != nodes_.end()) {
            if (it->second.meta.contains("title")) {
                main_error_node_title = " (title: " + it->second.meta["title"].get<std::string>() + ")";
            } else {
                main_error_node_title = " (class: " + it->second.class_type + ")";
            }
        }
    }

    GENAI_ERR("Prompt validation failed: %s", validation_result.error.message.c_str());
    GENAI_ERR("  Type: %s", validation_result.error.type.c_str());
    GENAI_ERR("  Details: %s%s", validation_result.error.details.c_str(), main_error_node_title.c_str());

    ss << "[" << validation_result.error.type << "] " << validation_result.error.message
       << ": " << validation_result.error.details << main_error_node_title << "\n";

    // Print node errors with titles
    for (const auto& [node_id, node_error] : validation_result.node_errors) {
        std::string title = node_error.contains("class_type")
            ? node_error["class_type"].get<std::string>()
            : "unknown";
        auto it = nodes_.find(node_id);
        if (it != nodes_.end() && it->second.meta.contains("title")) {
            title = it->second.meta["title"].get<std::string>();
        }
        GENAI_ERR("  Node #%s (%s):", node_id.c_str(), title.c_str());
        ss << "  Node #" << node_id << " (" << title << "):\n";

        // Print detailed errors for this node
        if (node_error.contains("errors") && node_error["errors"].is_array()) {
            for (const auto& err : node_error["errors"]) {
                std::string err_type = err.contains("type") ? err["type"].get<std::string>() : "";
                std::string err_msg = err.contains("message") ? err["message"].get<std::string>() : "";
                std::string err_details = err.contains("details") ? err["details"].get<std::string>() : "";
                GENAI_ERR("    [%s] %s: %s", err_type.c_str(), err_msg.c_str(), err_details.c_str());
                ss << "    [" << err_type << "] " << err_msg << ": " << err_details << "\n";
            }
        }
    }

    return ss.str();
}

// ============================================================================
// Default Node Classes Initialization
}  // namespace comfyui
}  // namespace module
}  // namespace genai
}  // namespace ov
