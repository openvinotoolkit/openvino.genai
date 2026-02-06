// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file comfyui.hpp
 * @brief ComfyUI to OpenVINO GenAI ModulePipeline converter
 *
 * This header file contains:
 * 1. ComfyUIJsonParser - Parse and validate ComfyUI API JSON files
 * 2. WorkflowToApiConverter - Convert ComfyUI workflow JSON to API JSON format
 * 3. ComfyUIToGenAIConverter - Convert API JSON to OpenVINO GenAI YAML pipeline
 *
 * Usage:
 * @code
 *   #include "comfyui.hpp"
 *
 *   // Option 1: Direct conversion from JSON file to YAML string
 *   comfyui::ComfyUIToGenAIConverter converter;
 *   std::string yaml = converter.get_yaml_for_pipeline("workflow.json", options);
 *   ov::genai::module::ModulePipeline pipeline(yaml);
 *
 *   // Option 2: Step by step with validation
 *   comfyui::ComfyUIJsonParser parser;
 *   parser.load_json_file("workflow.json");  // Supports both workflow and API format
 *   auto result = parser.validate_prompt("my_workflow");
 *   if (result.success) {
 *       std::string yaml = converter.convert_to_yaml(parser.get_api_json(), options);
 *   }
 * @endcode
 */

#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <variant>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <openvino/runtime/properties.hpp>

namespace ov {
namespace genai {
namespace module {
namespace comfyui {

// Use ordered_json to preserve insertion order
using json = nlohmann::ordered_json;

// ============================================================================
// Forward Declarations
// ============================================================================

class ComfyUIJsonParser;
class WorkflowToApiConverter;
class ComfyUIToGenAIConverter;

// ============================================================================
// Common Types
// ============================================================================

// Input type information
struct InputTypeInfo {
    std::string type;  // INT, FLOAT, STRING, BOOLEAN, or custom types
    std::string category;  // "required" or "optional"
    std::map<std::string, std::variant<int, double, std::string, std::vector<std::string>>> extra_info;
};

// Node definition
struct Node {
    std::string class_type;
    std::map<std::string, json> inputs;
    json meta;
    std::string node_id_str;
};

// Validation error
struct ValidationError {
    std::string type;
    std::string message;
    std::string details;
    std::map<std::string, json> extra_info;
};

// Validation result for a node
struct NodeValidationResult {
    bool valid;
    std::vector<ValidationError> errors;
    std::string node_id;
};

// Node class information (moved from ComfyUIJsonParser private section)
struct NodeClassInfo {
    std::map<std::string, InputTypeInfo> required_inputs;
    std::map<std::string, InputTypeInfo> optional_inputs;
    std::vector<std::string> return_types;
    bool is_output_node;
};

// Prompt validation result
struct PromptValidationResult {
    bool success;
    ValidationError error;  // Main error if validation failed
    std::vector<std::pair<std::string, std::vector<ValidationError>>> output_errors;
    std::map<std::string, json> node_errors;
};

// ============================================================================
// ComfyUIJsonParser - Parse and validate ComfyUI JSON files
// ============================================================================

/**
 * @brief ComfyUI JSON Parser and Validator
 *
 * Parses ComfyUI API JSON files and validates the node connections and inputs.
 * Also supports workflow JSON format by automatically converting to API format.
 */
class ComfyUIJsonParser {
public:
    ComfyUIJsonParser();
    ~ComfyUIJsonParser();

    // Load and parse JSON file
    bool load_json_file(const std::filesystem::path& file_path);

    // Parse JSON from string
    bool parse_json_string(const std::string& json_str);

    // Validate the loaded prompt
    PromptValidationResult validate_prompt(const std::string& prompt_id = "");

    // Get the parsed nodes
    const std::map<std::string, Node>& get_nodes() const { return nodes_; }

    // Get the parsed API JSON (after workflow->API conversion if needed)
    const json& get_api_json() const { return prompt_; }

    // Get the API JSON as string
    std::string get_api_json_string() const { return prompt_.dump(2); }

    // Register a node class type with its input types
    void register_node_class(const std::string& class_name,
                            const std::map<std::string, InputTypeInfo>& required_inputs,
                            const std::map<std::string, InputTypeInfo>& optional_inputs,
                            const std::vector<std::string>& return_types,
                            bool is_output_node = false);

    /**
     * @brief Log validation errors and return error string
     * @param validation_result The validation result from validate_prompt()
     * @return Error details as formatted string
     */
    std::string get_validation_errors_string(const PromptValidationResult& validation_result) const;

    // Check if JSON is workflow format (has "nodes" array) or API format
    bool is_workflow_json(const json& j) const;

    // Get the source JSON format type
    enum class JsonFormat {
        UNKNOWN,
        WORKFLOW,  // Has "nodes" array
        API        // Direct node dictionary
    };
    JsonFormat get_json_format() const { return json_format_; }

private:
    // Process parsed JSON: detect format, convert if needed, and populate nodes_
    bool process_json(const json& source_json);

    // Validate individual node inputs
    NodeValidationResult validate_node_inputs(
        const std::string& prompt_id,
        const std::string& node_id,
        std::map<std::string, NodeValidationResult>& validated);

    // Validate input value type
    bool validate_input_type(const json& value, const std::string& expected_type);

    // Check if linked node output type matches input type
    bool validate_node_input(const std::string& received_type, const std::string& input_type);

private:
    json prompt_;  // The parsed JSON prompt
    std::map<std::string, Node> nodes_;  // Parsed nodes
    JsonFormat json_format_;  // Source JSON format

    // Node class mappings: class_type -> (required_inputs, optional_inputs, return_types, is_output_node)
    std::map<std::string, NodeClassInfo> node_class_mappings_;
};

// ============================================================================
// WorkflowToApiConverter - Convert workflow JSON to API JSON
// ============================================================================

/**
 * @brief Custom comparator for node IDs to maintain proper order
 * Regular nodes (e.g., "9", "58") come before subgraph nodes (e.g., "57:11")
 * Within each group, sort numerically
 */
struct NodeIdComparator {
    bool operator()(const std::string& a, const std::string& b) const {
        bool a_is_composite = a.find(':') != std::string::npos;
        bool b_is_composite = b.find(':') != std::string::npos;

        // Regular nodes come before composite nodes
        if (!a_is_composite && b_is_composite) return true;
        if (a_is_composite && !b_is_composite) return false;

        // Both are the same type, compare numerically
        if (!a_is_composite) {
            // Both are regular nodes
            return std::stoi(a) < std::stoi(b);
        } else {
            // Both are composite nodes (e.g., "57:11" vs "57:13")
            size_t colon_a = a.find(':');
            size_t colon_b = b.find(':');

            int parent_a = std::stoi(a.substr(0, colon_a));
            int parent_b = std::stoi(b.substr(0, colon_b));

            if (parent_a != parent_b) {
                return parent_a < parent_b;
            }

            int child_a = std::stoi(a.substr(colon_a + 1));
            int child_b = std::stoi(b.substr(colon_b + 1));
            return child_a < child_b;
        }
    }
};

/**
 * @brief Workflow JSON to API JSON converter
 *
 * Converts ComfyUI workflow JSON format to API JSON format.
 *
 * Workflow JSON structure (from ComfyUI UI):
 * {
 *   "nodes": [
 *     {
 *       "id": 9,
 *       "type": "SaveImage",
 *       "inputs": [{"name": "images", "type": "IMAGE", "link": 54}],
 *       "widgets_values": ["ComfyUI"]
 *     }
 *   ],
 *   "links": [[54, 28, 0, 9, 0, "IMAGE"]],
 *   ...
 * }
 *
 * API JSON structure (for execution):
 * {
 *   "9": {
 *     "inputs": {
 *       "filename_prefix": "ComfyUI",
 *       "images": ["28", 0]
 *     },
 *     "class_type": "SaveImage"
 *   }
 * }
 */
class WorkflowToApiConverter {
public:
    /**
     * @brief Convert workflow JSON to API JSON format
     * @param workflow_json Workflow JSON object
     * @return API JSON object
     */
    static json convert(const nlohmann::json& workflow_json);

    /**
     * @brief Convert workflow to list of nodes (preserves insertion order)
     * Regular nodes come first (in workflow order), then subgraph nodes
     */
    static std::vector<std::pair<std::string, json>> convert_to_node_map(const nlohmann::json& workflow_json);

    /**
     * @brief Load workflow JSON from file and convert to API JSON
     * @param workflow_path Path to workflow JSON file
     * @param api_path Path to save API JSON file
     * @return true if successful
     */
    static bool convert_file(const std::string& workflow_path, const std::string& api_path);

private:
    /**
     * @brief Check if a string is a UUID (contains hyphens in UUID format)
     */
    static bool is_uuid(const std::string& str);

    /**
     * @brief Expand subgraph nodes into node_list
     */
    static void expand_subgraph(
        const nlohmann::json& parent_node,
        const nlohmann::json& subgraph,
        std::vector<std::pair<std::string, json>>& node_list,
        const std::unordered_map<int, nlohmann::json>& link_map,
        const std::unordered_map<int, int>& bypass_map
    );

    /**
     * @brief Get widget input names for a given node type
     */
    static std::vector<std::string> get_widget_input_names(const std::string& node_type, size_t widget_count);
};

// ============================================================================
// ComfyUIToGenAIConverter - Convert ComfyUI API JSON to OpenVINO Modular GenAI Pipeline YAML
// ============================================================================

/**
 * @brief Conversion options for ComfyUI to GenAI YAML converter
 */
struct ConversionOptions {
    std::string model_path = "./models/";  // Base path for model files
    std::string device = "CPU"; // default device
    std::string model_type = "unknown";  // Model type (e.g., "wan2.1", "zimage")
    int tile_size = 0;  // VAE decoder tile size (sample_size), 0 means use value from JSON
    int use_tiling = -1;  // -1=auto (default for model type), 0=disable, 1=enable
    // Note: use_tiled_vae is now auto-detected from VAEDecodeSwitcher's select_decoder input
};

/**
 * @brief ComfyUI API JSON to OpenVINO GenAI YAML converter
 *
 * Converts ComfyUI API JSON files to OpenVINO GenAI modular pipeline YAML configuration.
 *
 * A Sample ZImage Mapping rules as example:
 * - KSampler (node 3) -> DenoiserLoopModule
 * - CLIPTextEncode (nodes 6, 7) -> ClipTextEncoderModule
 * - SaveImage (node 9) -> SaveImageModule
 * - EmptySD3LatentImage (node 13) -> RandomLatentImageModule (width, height)
 * - UNETLoader (node 16) -> DenoiserLoopModule params.model_path
 * - VAELoader (node 17) -> VAEDecoderModule params.model_path
 * - CLIPLoader (node 18) -> ClipTextEncoderModule params.model_path
 * - VAEDecodeSwitcher (node 28) -> VAEDecoderTilingModule
 */
class ComfyUIToGenAIConverter {
public:
    // Alias for backward compatibility
    using ConversionOptions = comfyui::ConversionOptions;

    ComfyUIToGenAIConverter() = default;

    /**
     * @brief Convert ComfyUI API JSON file to OpenVINO GenAI YAML
     * @param api_json_path Path to the ComfyUI API JSON file
     * @param output_yaml_path Path for the output YAML file
     * @param options Conversion options
     * @return true if conversion successful
     */
    bool convert_file(const std::string& api_json_path,
                      const std::string& output_yaml_path,
                      const ConversionOptions& options = ConversionOptions());

    /**
     * @brief Convert ComfyUI API JSON to OpenVINO GenAI YAML string
     * @param api_json The parsed API JSON
     * @param options Conversion options
     * @return YAML content as string
     */
    std::string convert_to_yaml(const json& api_json,
                                const ConversionOptions& options = ConversionOptions());

    /**
     * @brief Convert ComfyUI API JSON to OpenVINO GenAI YAML string and extract pipeline inputs
     *
     * This overload extracts default input values from the ComfyUI JSON nodes
     * that can be used directly with pipeline.generate(inputs).
     *
     * Extracted inputs:
     * - "prompt": from CLIPTextEncode node's text input (std::string)
     * - "guidance_scale": from KSampler node's cfg input (float)
     * - "num_inference_steps": from KSampler node's steps input (int)
     * - "width": from EmptySD3LatentImage node's width input (int)
     * - "height": from EmptySD3LatentImage node's height input (int)
     * - "max_sequence_length": default value (int)
     *
     * @param api_json The parsed API JSON
     * @param[out] pipeline_inputs Output map containing extracted input values
     * @param options Conversion options
     * @return YAML content as string
     *
     * Example usage:
     * @code
     *   ComfyUIToGenAIConverter converter;
     *   std::map<std::string, std::any> extracted;
     *   std::string yaml = converter.convert_to_yaml(api_json, extracted, options);
     *
     *   // Use ov::AnyMap directly for pipeline.generate()
     *   ov::AnyMap inputs;
     *   inputs["prompt"] = extracted["prompt"].as<std::string>();
     *   inputs["width"] = extracted["width"].as<int>();
     *   // ... or override with custom values
     *
     *   ModulePipeline pipeline(yaml);
     *   pipeline.generate(inputs);
     * @endcode
     */
    std::string convert_to_yaml(const json& api_json,
                                ov::AnyMap& pipeline_inputs,
                                const ConversionOptions& options = ConversionOptions());

    /**
     * @brief Get YAML content for ModulePipeline from ComfyUI JSON file
     *
     * This is a convenience method that combines parsing and conversion.
     * The returned string can be directly used as input to:
     *   ModulePipeline(const std::string& config_yaml_content)
     *
     * @param json_file_path Path to ComfyUI JSON file (API or workflow format)
     * @param options Conversion options
     * @return YAML content string, or empty string on error
     *
     * Example usage:
     * @code
     *   ComfyUIToGenAIConverter converter;
     *   ComfyUIToGenAIConverter::ConversionOptions opts;
     *   opts.model_path = "/path/to/model/";
     *
     *   std::string yaml_content = converter.get_yaml_for_pipeline("workflow.json", opts);
     *   if (!yaml_content.empty()) {
     *       ov::genai::module::ModulePipeline pipeline(yaml_content);
     *       // ... use pipeline
     *   }
     * @endcode
     */
    std::string get_yaml_for_pipeline(const std::string& json_file_path,
                                      const ConversionOptions& options = ConversionOptions());

    // Node data extracted from API JSON (public for node handlers)
    struct NodeInfo {
        std::string class_type;
        json inputs;
        std::string title;
        std::string node_id_str;
    };

    // Pipeline context - stores found nodes and their JSON data for direct YAML generation
    struct PipelineParams {
        // Stored node data by class_type (for direct JSON->YAML conversion)
        std::map<std::string, std::vector<NodeInfo>> nodes_by_type;

        // Found node types - automatically set when nodes are processed
        std::set<std::string> found_node_types;

        // Helper method to check if a node type was found
        bool has(const std::string& node_type) const {
            return found_node_types.count(node_type) > 0;
        }

        // Helper method to mark a node type as found and store its data
        void store_node(const std::string& node_type, const NodeInfo& node) {
            found_node_types.insert(node_type);
            nodes_by_type[node_type].push_back(node);
        }

        // Get first node of a type (most common case)
        const NodeInfo* get_node(const std::string& node_type) const {
            auto it = nodes_by_type.find(node_type);
            if (it != nodes_by_type.end() && !it->second.empty()) {
                return &it->second[0];
            }
            return nullptr;
        }

        // Get all nodes of a type
        const std::vector<NodeInfo>* get_nodes(const std::string& node_type) const {
            auto it = nodes_by_type.find(node_type);
            if (it != nodes_by_type.end()) {
                return &it->second;
            }
            return nullptr;
        }

        // Helper to get JSON value with default
        template<typename T>
        T get_value(const std::string& node_type, const std::string& input_name, T default_val) const {
            if (auto* node = get_node(node_type)) {
                if (node->inputs.contains(input_name)) {
                    return node->inputs[input_name].get<T>();
                }
            }
            return default_val;
        }
    };

private:
    // Generate YAML with extracted params output for pipeline_inputs
    std::string generate_yaml(const json& api_json,
                              PipelineParams& out_params,
                              const ConversionOptions& options);
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Log validation errors from PromptValidationResult
 *
 * This function logs detailed validation error information using GENAI_DEBUG.
 * It handles main errors, output errors, and node-specific errors.
 *
 * @param result The validation result to log
 */
void log_validation_errors(const PromptValidationResult& result);

/**
 * @brief Create ConversionOptions from pipeline_inputs AnyMap
 *
 * Extracts "model_path" and "device" from pipeline_inputs with defaults.
 * Default model_path is "./models/", default device is "CPU".
 *
 * @param pipeline_inputs The input map containing optional model_path and device
 * @return ConversionOptions with extracted or default values
 */
ConversionOptions create_conversion_options(const ov::AnyMap& pipeline_inputs);

/**
 * @brief Log parsed nodes information
 *
 * Logs the count and details of parsed nodes using GENAI_INFO and GENAI_DEBUG.
 *
 * @param nodes The map of parsed nodes
 */
void log_parsed_nodes(const std::map<std::string, Node>& nodes);

}  // namespace comfyui
}  // namespace module
}  // namespace genai
}  // namespace ov
