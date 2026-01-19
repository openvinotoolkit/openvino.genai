// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file node_class_registry.hpp
 * @brief Node class registry for ComfyUI nodes
 *
 * This file provides a centralized registry for all supported ComfyUI node classes.
 * To add support for a new node class, add a new registration in node_class_registry.cpp
 */

#pragma once

#include <string>
#include <map>
#include <vector>
#include <variant>
#include <functional>

// Include comfyui.hpp for InputTypeInfo and NodeClassInfo definitions
#include "comfyui.hpp"

namespace ov {
namespace genai {
namespace module {
namespace comfyui {

/**
 * @brief Node class registry for managing ComfyUI node class definitions
 *
 * This class provides a centralized way to register and retrieve node class
 * definitions. It supports both built-in node classes and custom extensions.
 */
class NodeClassRegistry {
public:
    using ExtraInfoMap = std::map<std::string, std::variant<int, double, std::string, std::vector<std::string>>>;

    /**
     * @brief Get the singleton instance of the registry
     */
    static NodeClassRegistry& instance();

    /**
     * @brief Initialize all default node classes
     * Call this once during parser construction
     */
    void initialize_defaults();

    /**
     * @brief Register a node class
     * @param class_name The ComfyUI class type name (e.g., "KSampler", "CLIPTextEncode")
     * @param info The node class information
     */
    void register_node_class(const std::string& class_name, const NodeClassInfo& info);

    /**
     * @brief Get all registered node class mappings
     * @return Reference to the node class mappings
     */
    const std::map<std::string, NodeClassInfo>& get_mappings() const;

    /**
     * @brief Check if a node class is registered
     * @param class_name The class type name to check
     * @return true if registered, false otherwise
     */
    bool has_node_class(const std::string& class_name) const;

    /**
     * @brief Get a node class info by name
     * @param class_name The class type name
     * @return Pointer to NodeClassInfo, or nullptr if not found
     */
    const NodeClassInfo* get_node_class(const std::string& class_name) const;

private:
    NodeClassRegistry() = default;
    ~NodeClassRegistry() = default;
    NodeClassRegistry(const NodeClassRegistry&) = delete;
    NodeClassRegistry& operator=(const NodeClassRegistry&) = delete;

    std::map<std::string, NodeClassInfo> node_class_mappings_;
    bool initialized_ = false;

    // Helper for creating extra_info maps
    static ExtraInfoMap make_extra_info(
        std::initializer_list<std::pair<const std::string, std::variant<int, double, std::string, std::vector<std::string>>>> init);

    // Registration functions for each category of nodes
    void register_sampler_nodes();
    void register_text_encoder_nodes();
    void register_latent_nodes();
    void register_vae_nodes();
    void register_loader_nodes();
    void register_output_nodes();
    void register_utility_nodes();
};

}  // namespace comfyui
}  // namespace module
}  // namespace genai
}  // namespace ov
