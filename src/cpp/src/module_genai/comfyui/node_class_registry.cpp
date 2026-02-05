// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file node_class_registry.cpp
 * @brief Implementation of NodeClassRegistry
 *
 * This file contains all supported ComfyUI node class registrations.
 * To add a new node class:
 * 1. Find the appropriate register_*_nodes() function
 * 2. Add a new NodeClassInfo block following the existing pattern
 * 3. Define required_inputs, optional_inputs, return_types, and is_output_node
 */

#include "node_class_registry.hpp"

namespace ov {
namespace genai {
namespace module {
namespace comfyui {

// ============================================================================
// Singleton Instance
// ============================================================================

NodeClassRegistry& NodeClassRegistry::instance() {
    static NodeClassRegistry registry;
    return registry;
}

// ============================================================================
// Initialization
// ============================================================================

void NodeClassRegistry::initialize_defaults() {
    if (initialized_) return;

    register_sampler_nodes();
    register_text_encoder_nodes();
    register_latent_nodes();
    register_vae_nodes();
    register_loader_nodes();
    register_output_nodes();
    register_utility_nodes();

    initialized_ = true;
}

// ============================================================================
// Public Methods
// ============================================================================

void NodeClassRegistry::register_node_class(const std::string& class_name, const NodeClassInfo& info) {
    node_class_mappings_[class_name] = info;
}

const std::map<std::string, NodeClassInfo>& NodeClassRegistry::get_mappings() const {
    return node_class_mappings_;
}

bool NodeClassRegistry::has_node_class(const std::string& class_name) const {
    return node_class_mappings_.find(class_name) != node_class_mappings_.end();
}

const NodeClassInfo* NodeClassRegistry::get_node_class(const std::string& class_name) const {
    auto it = node_class_mappings_.find(class_name);
    return it != node_class_mappings_.end() ? &it->second : nullptr;
}

// ============================================================================
// Helper Functions
// ============================================================================

NodeClassRegistry::ExtraInfoMap NodeClassRegistry::make_extra_info(
    std::initializer_list<std::pair<const std::string, std::variant<int, double, std::string, std::vector<std::string>>>> init) {
    return ExtraInfoMap(init);
}

// ============================================================================
// Sampler Nodes
// ============================================================================

void NodeClassRegistry::register_sampler_nodes() {
    // KSampler - Main sampling node for image generation
    {
        NodeClassInfo info;
        info.required_inputs["model"] = InputTypeInfo{"MODEL", "required", {}};
        info.required_inputs["seed"] = InputTypeInfo{"INT", "required", make_extra_info({{"default", 0}, {"min", 0}})};
        info.required_inputs["steps"] = InputTypeInfo{"INT", "required", make_extra_info({{"default", 20}, {"min", 1}, {"max", 10000}})};
        info.required_inputs["cfg"] = InputTypeInfo{"FLOAT", "required", make_extra_info({{"default", 8.0}, {"min", 0.0}, {"max", 100.0}})};
        info.required_inputs["sampler_name"] = InputTypeInfo{"STRING", "required", {}};
        info.required_inputs["scheduler"] = InputTypeInfo{"STRING", "required", {}};
        info.required_inputs["positive"] = InputTypeInfo{"CONDITIONING", "required", {}};
        info.required_inputs["negative"] = InputTypeInfo{"CONDITIONING", "required", {}};
        info.required_inputs["latent_image"] = InputTypeInfo{"LATENT", "required", {}};
        info.required_inputs["denoise"] = InputTypeInfo{"FLOAT", "required", make_extra_info({{"default", 1.0}, {"min", 0.0}, {"max", 1.0}})};
        info.return_types = {"LATENT"};
        info.is_output_node = false;
        register_node_class("KSampler", info);
    }

    // Add more sampler nodes here as needed
    // Example: KSamplerAdvanced, SamplerCustom, etc.
}

// ============================================================================
// Text Encoder Nodes
// ============================================================================

void NodeClassRegistry::register_text_encoder_nodes() {
    // CLIPTextEncode - Encodes text prompts using CLIP
    {
        NodeClassInfo info;
        info.required_inputs["text"] = InputTypeInfo{"STRING", "required", {}};
        info.required_inputs["clip"] = InputTypeInfo{"CLIP", "required", {}};
        info.return_types = {"CONDITIONING"};
        info.is_output_node = false;
        register_node_class("CLIPTextEncode", info);
    }

    // ConditioningZeroOut - Zeros out conditioning
    {
        NodeClassInfo info;
        info.required_inputs["conditioning"] = InputTypeInfo{"CONDITIONING", "required", {}};
        info.return_types = {"CONDITIONING"};
        info.is_output_node = false;
        register_node_class("ConditioningZeroOut", info);
    }

    // Add more text encoder nodes here as needed
    // Example: CLIPTextEncodeSDXL, CLIPTextEncodeSD3, etc.
}

// ============================================================================
// Latent Image Nodes
// ============================================================================

void NodeClassRegistry::register_latent_nodes() {
    // EmptySD3LatentImage - Creates empty latent image for SD3
    {
        NodeClassInfo info;
        info.required_inputs["width"] = InputTypeInfo{"INT", "required", make_extra_info({{"default", 1024}, {"min", 16}, {"max", 16384}})};
        info.required_inputs["height"] = InputTypeInfo{"INT", "required", make_extra_info({{"default", 1024}, {"min", 16}, {"max", 16384}})};
        info.required_inputs["batch_size"] = InputTypeInfo{"INT", "required", make_extra_info({{"default", 1}, {"min", 1}, {"max", 4096}})};
        info.return_types = {"LATENT"};
        info.is_output_node = false;
        register_node_class("EmptySD3LatentImage", info);
    }

    // EmptyHunyuanLatentVideo - Creates empty latent video for HunyuanVideo
    {
        NodeClassInfo info;
        info.required_inputs["width"] = InputTypeInfo{"INT", "required", make_extra_info({{"default", 848}, {"min", 16}, {"max", 16384}, {"step", 16}})};
        info.required_inputs["height"] = InputTypeInfo{"INT", "required", make_extra_info({{"default", 480}, {"min", 16}, {"max", 16384}, {"step", 16}})};
        info.required_inputs["length"] = InputTypeInfo{"INT", "required", make_extra_info({{"default", 25}, {"min", 1}, {"max", 16384}, {"step", 4}})};
        info.required_inputs["batch_size"] = InputTypeInfo{"INT", "required", make_extra_info({{"default", 1}, {"min", 1}, {"max", 4096}})};
        info.return_types = {"LATENT"};
        info.is_output_node = false;
        register_node_class("EmptyHunyuanLatentVideo", info);
    }

    // Add more latent nodes here as needed
    // Example: EmptyLatentImage, LatentUpscale, etc.
}

// ============================================================================
// VAE Nodes
// ============================================================================

void NodeClassRegistry::register_vae_nodes() {
    // VAEDecode - Decodes latent to image
    {
        NodeClassInfo info;
        info.required_inputs["samples"] = InputTypeInfo{"LATENT", "required", {}};
        info.required_inputs["vae"] = InputTypeInfo{"VAE", "required", {}};
        info.return_types = {"IMAGE"};
        info.is_output_node = false;
        register_node_class("VAEDecode", info);
    }

    // VAEDecodeSwitcher - Decodes latent with tiling option
    {
        NodeClassInfo info;
        info.required_inputs["samples"] = InputTypeInfo{"LATENT", "required", {}};
        info.required_inputs["vae"] = InputTypeInfo{"VAE", "required", {}};
        info.required_inputs["select_decoder"] = InputTypeInfo{"STRING", "required", {}};
        info.optional_inputs["tile_size"] = InputTypeInfo{"INT", "optional", make_extra_info({{"default", 512}, {"min", 320}, {"max", 4096}})};
        info.optional_inputs["overlap"] = InputTypeInfo{"INT", "optional", make_extra_info({{"default", 64}, {"min", 0}, {"max", 256}})};
        info.optional_inputs["temporal_size"] = InputTypeInfo{"INT", "optional", make_extra_info({{"default", 64}})};
        info.optional_inputs["temporal_overlap"] = InputTypeInfo{"INT", "optional", make_extra_info({{"default", 8}})};
        info.return_types = {"IMAGE"};
        info.is_output_node = false;
        register_node_class("VAEDecodeSwitcher", info);
    }

    // Add more VAE nodes here as needed
    // Example: VAEEncode, VAEDecodeTiled, etc.
}

// ============================================================================
// Loader Nodes
// ============================================================================

void NodeClassRegistry::register_loader_nodes() {
    // UNETLoader - Loads UNET model
    {
        NodeClassInfo info;
        info.required_inputs["unet_name"] = InputTypeInfo{"STRING", "required", {}};
        info.optional_inputs["weight_dtype"] = InputTypeInfo{"STRING", "optional", make_extra_info({{"default", std::string("default")}})};
        info.return_types = {"MODEL"};
        info.is_output_node = false;
        register_node_class("UNETLoader", info);
    }

    // VAELoader - Loads VAE model
    {
        NodeClassInfo info;
        info.required_inputs["vae_name"] = InputTypeInfo{"STRING", "required", {}};
        info.return_types = {"VAE"};
        info.is_output_node = false;
        register_node_class("VAELoader", info);
    }

    // CLIPLoader - Loads CLIP model
    {
        NodeClassInfo info;
        info.required_inputs["clip_name"] = InputTypeInfo{"STRING", "required", {}};
        info.required_inputs["type"] = InputTypeInfo{"STRING", "required", {}};
        info.optional_inputs["device"] = InputTypeInfo{"STRING", "optional", make_extra_info({{"default", std::string("default")}})};
        info.return_types = {"CLIP"};
        info.is_output_node = false;
        register_node_class("CLIPLoader", info);
    }

    // Add more loader nodes here as needed
    // Example: CheckpointLoaderSimple, LoraLoader, etc.
}

// ============================================================================
// Output Nodes
// ============================================================================

void NodeClassRegistry::register_output_nodes() {
    // SaveImage - Saves generated image to disk
    {
        NodeClassInfo info;
        info.required_inputs["images"] = InputTypeInfo{"IMAGE", "required", {}};
        info.optional_inputs["filename_prefix"] = InputTypeInfo{"STRING", "optional", make_extra_info({{"default", std::string("ComfyUI")}})};
        info.return_types = {};
        info.is_output_node = true;
        register_node_class("SaveImage", info);
    }

    // SaveAnimatedWEBP - Saves animated WEBP image
    {
        NodeClassInfo info;
        info.required_inputs["images"] = InputTypeInfo{"IMAGE", "required", {}};
        info.optional_inputs["filename_prefix"] = InputTypeInfo{"STRING", "optional", make_extra_info({{"default", std::string("ComfyUI")}})};
        info.optional_inputs["fps"] = InputTypeInfo{"FLOAT", "optional", make_extra_info({{"default", 6.0}, {"min", 0.01}, {"max", 1000.0}})};
        info.optional_inputs["lossless"] = InputTypeInfo{"BOOLEAN", "optional", make_extra_info({{"default", std::string("true")}})};
        info.optional_inputs["quality"] = InputTypeInfo{"INT", "optional", make_extra_info({{"default", 80}, {"min", 0}, {"max", 100}})};
        info.optional_inputs["method"] = InputTypeInfo{"STRING", "optional", make_extra_info({{"default", std::string("default")}})};
        info.return_types = {};
        info.is_output_node = true;
        register_node_class("SaveAnimatedWEBP", info);
    }

    // SaveWEBM - Saves video as WEBM format
    {
        NodeClassInfo info;
        info.required_inputs["images"] = InputTypeInfo{"IMAGE", "required", {}};
        info.optional_inputs["filename_prefix"] = InputTypeInfo{"STRING", "optional", make_extra_info({{"default", std::string("ComfyUI")}})};
        info.optional_inputs["codec"] = InputTypeInfo{"STRING", "optional", make_extra_info({{"default", std::string("vp9")}})};
        info.optional_inputs["fps"] = InputTypeInfo{"FLOAT", "optional", make_extra_info({{"default", 24.0}, {"min", 0.01}, {"max", 1000.0}})};
        info.optional_inputs["crf"] = InputTypeInfo{"FLOAT", "optional", make_extra_info({{"default", 32.0}, {"min", 0.0}, {"max", 63.0}})};
        info.return_types = {};
        info.is_output_node = true;
        register_node_class("SaveWEBM", info);
    }

    // Add more output nodes here as needed
    // Example: PreviewImage, SaveImageWebsocket, etc.
}

// ============================================================================
// Utility Nodes
// ============================================================================

void NodeClassRegistry::register_utility_nodes() {
    // PrimitiveStringMultiline - Multiline string input
    {
        NodeClassInfo info;
        info.required_inputs["value"] = InputTypeInfo{"STRING", "required", {}};
        info.return_types = {"STRING"};
        info.is_output_node = false;
        register_node_class("PrimitiveStringMultiline", info);
    }

    // StringConcatenate - Concatenates two strings
    {
        NodeClassInfo info;
        info.required_inputs["string_a"] = InputTypeInfo{"STRING", "required", {}};
        info.required_inputs["string_b"] = InputTypeInfo{"STRING", "required", {}};
        info.optional_inputs["delimiter"] = InputTypeInfo{"STRING", "optional", make_extra_info({{"default", std::string("")}})};
        info.return_types = {"STRING"};
        info.is_output_node = false;
        register_node_class("StringConcatenate", info);
    }

    // ModelSamplingAuraFlow - Modifies model sampling for AuraFlow
    {
        NodeClassInfo info;
        info.required_inputs["model"] = InputTypeInfo{"MODEL", "required", {}};
        info.required_inputs["shift"] = InputTypeInfo{"FLOAT", "required", make_extra_info({{"default", 1.73}, {"min", 0.0}, {"max", 100.0}})};
        info.return_types = {"MODEL"};
        info.is_output_node = false;
        register_node_class("ModelSamplingAuraFlow", info);
    }

    // ModelSamplingSD3 - Modifies model sampling for SD3
    {
        NodeClassInfo info;
        info.required_inputs["model"] = InputTypeInfo{"MODEL", "required", {}};
        info.required_inputs["shift"] = InputTypeInfo{"FLOAT", "required", make_extra_info({{"default", 3.0}, {"min", 0.0}, {"max", 100.0}})};
        info.return_types = {"MODEL"};
        info.is_output_node = false;
        register_node_class("ModelSamplingSD3", info);
    }

    // Add more utility nodes here as needed
    // Example: Reroute, Note, PrimitiveNumber, etc.
}

}  // namespace comfyui
}  // namespace module
}  // namespace genai
}  // namespace ov
