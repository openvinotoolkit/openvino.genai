// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file yaml_module_generators.hpp
 * @brief YAML module generators for ComfyUI to GenAI conversion
 *
 * This file provides a class-based registry for YAML module generators.
 * Each generator is a class derived from YamlModuleGeneratorBase that converts
 * a ComfyUI node type to its corresponding YAML module.
 *
 * To add a new generator:
 * 1. Create a class derived from YamlModuleGeneratorBase
 * 2. Implement the generate() method
 * 3. Register it in yaml_module_generators.cpp using register_generator()
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <yaml-cpp/yaml.h>
#include "comfyui.hpp"

namespace ov {
namespace genai {
namespace module {
namespace comfyui {

// Forward declaration
class YamlModuleGeneratorBase;

/**
 * @brief Context passed to generators during YAML generation
 *
 * Contains all information needed for a generator to produce YAML output.
 */
struct YamlGeneratorContext {
    YAML::Node& pipeline_modules;       // YAML node for pipeline_modules section
    YAML::Node& root;                   // Root YAML node for sub_modules etc.
    const ComfyUIToGenAIConverter::PipelineParams& params;  // Pipeline parameters containing all stored nodes
    const ConversionOptions& options;   // Conversion options (device, model path, etc.)
    const ComfyUIToGenAIConverter::NodeInfo& current_node;  // Current node being processed
    std::string model_type;             // Model type detected during generation (e.g., "wan2.1", "zimage")

    YamlGeneratorContext(
        YAML::Node& pm,
        YAML::Node& r,
        const ComfyUIToGenAIConverter::PipelineParams& p,
        const ConversionOptions& o,
        const ComfyUIToGenAIConverter::NodeInfo& n,
        const std::string& mt = "unknown")
        : pipeline_modules(pm), root(r), params(p), options(o), current_node(n), model_type(mt) {}
};

/**
 * @brief Base class for all YAML module generators
 *
 * Each generator handles one or more ComfyUI node types and converts them
 * to corresponding YAML module definitions.
 */
class YamlModuleGeneratorBase {
public:
    virtual ~YamlModuleGeneratorBase() = default;

    /**
     * @brief Generate YAML module for the current node
     * @param ctx Generator context containing all necessary information
     */
    virtual void generate(YamlGeneratorContext& ctx) = 0;
};

// ============================================================================
// Concrete Generator Classes
// ============================================================================

/**
 * @brief Generator for EmptySD3LatentImage -> RandomLatentImageModule
 */
class RandomLatentImageModuleGenerator : public YamlModuleGeneratorBase {
public:
    void generate(YamlGeneratorContext& ctx) override;
};

/**
 * @brief Generator for EmptyHunyuanLatentVideo -> RandomLatentImageModule
 * Maps: width->width, height->height, length->num_frames, batch_size->batch_size
 */
class HunyuanLatentVideoModuleGenerator : public YamlModuleGeneratorBase {
public:
    void generate(YamlGeneratorContext& ctx) override;
};

/**
 * @brief Generator for CLIPTextEncode -> ClipTextEncoderModule
 */
class ClipTextEncoderModuleGenerator : public YamlModuleGeneratorBase {
public:
    void generate(YamlGeneratorContext& ctx) override;
};

/**
 * @brief Generator for KSampler -> DenoiserLoopModule
 */
class DenoiserLoopModuleGenerator : public YamlModuleGeneratorBase {
public:
    void generate(YamlGeneratorContext& ctx) override;
};

/**
 * @brief Generator for VAEDecode -> VAEDecoderModule
 */
class VAEDecoderModuleGenerator : public YamlModuleGeneratorBase {
public:
    void generate(YamlGeneratorContext& ctx) override;
};

/**
 * @brief Generator for VAEDecodeSwitcher -> VAEDecoderModule or VAEDecoderTilingModule
 */
class VAEDecoderTilingModuleGenerator : public YamlModuleGeneratorBase {
public:
    void generate(YamlGeneratorContext& ctx) override;
};

/**
 * @brief Generator for SaveImage -> SaveImageModule
 */
class SaveImageModuleGenerator : public YamlModuleGeneratorBase {
public:
    void generate(YamlGeneratorContext& ctx) override;
};

/**
 * @brief Generator for SaveAnimatedWEBP -> SaveVideoModule
 * Maps: filename_prefix->filename_prefix, fps->fps, quality->quality
 */
class SaveVideoModuleGenerator : public YamlModuleGeneratorBase {
public:
    void generate(YamlGeneratorContext& ctx) override;
};

// ============================================================================
// Registry for YAML Module Generators
// ============================================================================

/**
 * @brief Registry for YAML module generators
 *
 * This class manages the mapping from ComfyUI node types to YAML generators.
 * Each generator is called when its corresponding node type is encountered
 * during iteration over api_json.items().
 */
class YamlModuleGeneratorRegistry {
public:
    /**
     * @brief Get the singleton instance
     */
    static YamlModuleGeneratorRegistry& instance();

    /**
     * @brief Initialize all default generators
     */
    void initialize_defaults();

    /**
     * @brief Register a generator for a specific class_type
     * @param class_type ComfyUI node class_type (e.g., "KSampler", "CLIPTextEncode")
     * @param generator Shared pointer to the generator
     */
    void register_generator(const std::string& class_type, std::shared_ptr<YamlModuleGeneratorBase> generator);

    /**
     * @brief Get generator for a specific node type
     * @param node_type ComfyUI class_type
     * @return Pointer to generator, or nullptr if not found
     */
    YamlModuleGeneratorBase* get_generator(const std::string& node_type) const;

    /**
     * @brief Check if a generator exists for the given node type
     * @param node_type ComfyUI class_type
     * @return true if generator exists
     */
    bool has_generator(const std::string& node_type) const;

    /**
     * @brief Generate the result module (always called last)
     */
    static void generate_result_module(
        YAML::Node& pipeline_modules,
        YAML::Node& root,
        const ComfyUIToGenAIConverter::PipelineParams& params,
        const ConversionOptions& options,
        const std::string& model_type = "unknown");

private:
    YamlModuleGeneratorRegistry() = default;
    ~YamlModuleGeneratorRegistry() = default;
    YamlModuleGeneratorRegistry(const YamlModuleGeneratorRegistry&) = delete;
    YamlModuleGeneratorRegistry& operator=(const YamlModuleGeneratorRegistry&) = delete;

    // Map from node_type to generator
    std::unordered_map<std::string, std::shared_ptr<YamlModuleGeneratorBase>> generators_;

    // Keep ownership of all generators
    std::vector<std::shared_ptr<YamlModuleGeneratorBase>> generator_storage_;

    bool initialized_ = false;
};

// ============================================================================
// Helper Functions for YAML Node Creation
// ============================================================================

/**
 * @brief Create an input port node
 */
YAML::Node create_input_node(const std::string& name, const std::string& source, const std::string& type);

/**
 * @brief Create an output port node
 */
YAML::Node create_output_node(const std::string& name, const std::string& type);

}  // namespace comfyui
}  // namespace module
}  // namespace genai
}  // namespace ov
