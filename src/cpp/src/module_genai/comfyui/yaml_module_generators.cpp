// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * @file yaml_module_generators.cpp
 * @brief Implementation of YAML module generators
 *
 * This file contains all YAML module generator classes for converting ComfyUI nodes
 * to GenAI pipeline YAML format.
 *
 * To add a new generator:
 * 1. Create a class derived from YamlModuleGeneratorBase in the header
 * 2. Implement the generate() method in this file
 * 3. Register it in initialize_defaults()
 */

#include "yaml_module_generators.hpp"
#include "logger.hpp"

namespace ov {
namespace genai {
namespace module {
namespace comfyui {

// ============================================================================
// Helper Functions
// ============================================================================

YAML::Node create_input_node(const std::string& name, const std::string& source, const std::string& type) {
    YAML::Node node;
    node["name"] = name;
    node["source"] = source;
    node["type"] = type;
    return node;
}

YAML::Node create_output_node(const std::string& name, const std::string& type) {
    YAML::Node node;
    node["name"] = name;
    node["type"] = type;
    return node;
}

// ============================================================================
// Singleton Instance
// ============================================================================

YamlModuleGeneratorRegistry& YamlModuleGeneratorRegistry::instance() {
    static YamlModuleGeneratorRegistry registry;
    return registry;
}

// ============================================================================
// Registry Methods
// ============================================================================

void YamlModuleGeneratorRegistry::initialize_defaults() {
    if (initialized_) return;

    // Register generators by ComfyUI class_type for easy lookup
    // Format: { "class_type", generator_instance }
    static const std::vector<std::pair<std::string, std::shared_ptr<YamlModuleGeneratorBase>>> default_generators = {
        {"EmptySD3LatentImage",  std::make_shared<RandomLatentImageModuleGenerator>()},
        {"EmptyHunyuanLatentVideo", std::make_shared<HunyuanLatentVideoModuleGenerator>()},
        {"CLIPTextEncode",       std::make_shared<ClipTextEncoderModuleGenerator>()},
        {"KSampler",             std::make_shared<DenoiserLoopModuleGenerator>()},
        {"VAEDecode",            std::make_shared<VAEDecoderModuleGenerator>()},
        {"VAEDecodeSwitcher",    std::make_shared<VAEDecoderTilingModuleGenerator>()},
        {"SaveImage",            std::make_shared<SaveImageModuleGenerator>()},
        {"SaveAnimatedWEBP",     std::make_shared<SaveVideoModuleGenerator>()},
    };

    for (const auto& [class_type, generator] : default_generators) {
        register_generator(class_type, generator);
    }

    initialized_ = true;
}

void YamlModuleGeneratorRegistry::register_generator(
    const std::string& class_type,
    std::shared_ptr<YamlModuleGeneratorBase> generator) {
    // Store ownership
    generator_storage_.push_back(generator);
    // Map class_type to generator
    generators_[class_type] = generator;
}

YamlModuleGeneratorBase* YamlModuleGeneratorRegistry::get_generator(const std::string& node_type) const {
    auto it = generators_.find(node_type);
    if (it != generators_.end()) {
        return it->second.get();
    }
    return nullptr;
}

bool YamlModuleGeneratorRegistry::has_generator(const std::string& node_type) const {
    return generators_.find(node_type) != generators_.end();
}

// ============================================================================
// RandomLatentImageModule Generator (EmptySD3LatentImage)
// ============================================================================

void RandomLatentImageModuleGenerator::generate(YamlGeneratorContext& ctx) {
    const auto& node = ctx.current_node;
    GENAI_DEBUG("[EmptySD3LatentImage] Processing node: node_id_str=%s, title=%s",
                node.node_id_str.c_str(), node.title.c_str());

    std::string module_name = node.node_id_str;
    GENAI_DEBUG("[YAML] Adding RandomLatentImageModule (%s)", module_name.c_str());

    YAML::Node module = ctx.pipeline_modules[module_name];
    module["device"] = ctx.options.device;

    YAML::Node inputs;
    inputs.push_back(create_input_node("width", "pipeline_params.width", "Int"));
    inputs.push_back(create_input_node("height", "pipeline_params.height", "Int"));
    inputs.push_back(create_input_node("batch_size", "pipeline_params.batch_size", "Int"));
    inputs.push_back(create_input_node("seed", "pipeline_params.seed", "Int"));
    module["inputs"] = inputs;

    YAML::Node outputs;
    outputs.push_back(create_output_node("latents", "OVTensor"));
    module["outputs"] = outputs;

    module["params"]["model_path"] = ctx.options.model_path;
    module["type"] = "RandomLatentImageModule";
}

// ============================================================================
// HunyuanLatentVideoModule Generator (EmptyHunyuanLatentVideo)
// ============================================================================

void HunyuanLatentVideoModuleGenerator::generate(YamlGeneratorContext& ctx) {
    const auto& node = ctx.current_node;
    GENAI_DEBUG("[EmptyHunyuanLatentVideo] Processing node: node_id_str=%s, title=%s",
                node.node_id_str.c_str(), node.title.c_str());

    std::string module_name = node.node_id_str;
    GENAI_DEBUG("[YAML] Adding RandomLatentImageModule (%s) for HunyuanLatentVideo", module_name.c_str());

    YAML::Node module = ctx.pipeline_modules[module_name];
    module["device"] = ctx.options.device;

    // Map: width->width, height->height, length->num_frames, batch_size->batch_size
    YAML::Node inputs;
    inputs.push_back(create_input_node("width", "pipeline_params.width", "Int"));
    inputs.push_back(create_input_node("height", "pipeline_params.height", "Int"));
    inputs.push_back(create_input_node("num_frames", "pipeline_params.num_frames", "Int"));
    inputs.push_back(create_input_node("batch_size", "pipeline_params.batch_size", "Int"));
    inputs.push_back(create_input_node("seed", "pipeline_params.seed", "Int"));
    module["inputs"] = inputs;

    YAML::Node outputs;
    outputs.push_back(create_output_node("latents", "OVTensor"));
    module["outputs"] = outputs;

    module["params"]["model_path"] = ctx.options.model_path;
    module["type"] = "RandomLatentImageModule";
}

// ============================================================================
// ClipTextEncoderModule Generator (CLIPTextEncode)
// ============================================================================

void ClipTextEncoderModuleGenerator::generate(YamlGeneratorContext& ctx) {
    // CLIPTextEncode -> ClipTextEncoderModule
    const auto& node = ctx.current_node;
    GENAI_DEBUG("[CLIP] Processing node: node_id_str=%s, title=%s",
                node.node_id_str.c_str(), node.title.c_str());

    std::string module_name = node.node_id_str;
    bool is_negative = (node.title.find("Negative") != std::string::npos);

    GENAI_DEBUG("[YAML] Adding ClipTextEncoderModule (%s) - %s",
                module_name.c_str(), is_negative ? "negative" : "positive");

    YAML::Node module = ctx.pipeline_modules[module_name];
    module["device"] = ctx.options.device;

    YAML::Node inputs;
    if (is_negative) {
        // For negative prompt encoder, use "prompt" input name but source from negative_prompt
        // This is because ClipTextEncoderModule treats "prompt" input as the main input
        inputs.push_back(create_input_node("prompt", "pipeline_params.negative_prompt", "String"));
    } else {
        inputs.push_back(create_input_node("prompt", "pipeline_params.prompt", "String"));
    }
    inputs.push_back(create_input_node("guidance_scale", "pipeline_params.guidance_scale", "Float"));
    inputs.push_back(create_input_node("max_sequence_length", "pipeline_params.max_sequence_length", "Int"));
    module["inputs"] = inputs;

    YAML::Node outputs;
    // ClipTextEncoderModule always outputs to "prompt_embeds" when only prompt input is provided
    outputs.push_back(create_output_node("prompt_embeds", "VecOVTensor"));
    module["outputs"] = outputs;

    module["params"]["model_path"] = ctx.options.model_path;
    module["type"] = "ClipTextEncoderModule";
}

// ============================================================================
// DenoiserLoopModule Generator (KSampler)
// ============================================================================

void DenoiserLoopModuleGenerator::generate(YamlGeneratorContext& ctx) {
    // KSampler -> DenoiserLoopModule
    const auto& node = ctx.current_node;
    GENAI_DEBUG("[KSampler] Processing node: node_id_str=%s, title=%s",
                node.node_id_str.c_str(), node.title.c_str());

    // Get referenced module names from CLIPTextEncode nodes
    std::string clip_positive_module_name;
    std::string clip_negative_module_name;
    if (auto* clip_nodes = ctx.params.get_nodes("CLIPTextEncode")) {
        for (const auto& clip_node : *clip_nodes) {
            if (clip_node.title.find("Negative") != std::string::npos) {
                clip_negative_module_name = clip_node.node_id_str;
            } else {
                clip_positive_module_name = clip_node.node_id_str;
            }
        }
    }

    std::string latent_module_name;
    bool is_video_latent = false;
    if (auto* latent_node = ctx.params.get_node("EmptySD3LatentImage")) {
        latent_module_name = latent_node->node_id_str;
    } else if (auto* latent_node = ctx.params.get_node("EmptyHunyuanLatentVideo")) {
        latent_module_name = latent_node->node_id_str;
        is_video_latent = true;
    }

    std::string module_name = node.node_id_str;
    GENAI_DEBUG("[YAML] Adding DenoiserLoopModule (%s)", module_name.c_str());

    YAML::Node module = ctx.pipeline_modules[module_name];
    module["device"] = ctx.options.device;

    YAML::Node inputs;
    if (!clip_positive_module_name.empty()) {
        inputs.push_back(create_input_node("prompt_embeds", clip_positive_module_name + ".prompt_embeds", "VecOVTensor"));
    }
    if (!clip_negative_module_name.empty()) {
        // Negative encoder also outputs to "prompt_embeds" (not "negative_prompt_embeds")
        inputs.push_back(create_input_node("prompt_embeds_negative", clip_negative_module_name + ".prompt_embeds", "VecOVTensor"));
    }
    if (!latent_module_name.empty()) {
        inputs.push_back(create_input_node("latents", latent_module_name + ".latents", "OVTensor"));
    }
    // Add guidance_scale input for CFG
    inputs.push_back(create_input_node("guidance_scale", "pipeline_params.guidance_scale", "Float"));
    inputs.push_back(create_input_node("num_inference_steps", "pipeline_params.num_inference_steps", "Int"));
    inputs.push_back(create_input_node("width", "pipeline_params.width", "Int"));
    inputs.push_back(create_input_node("height", "pipeline_params.height", "Int"));
    if (is_video_latent) {
        inputs.push_back(create_input_node("num_frames", "pipeline_params.num_frames", "Int"));
    }
    module["inputs"] = inputs;

    YAML::Node outputs;
    outputs.push_back(create_output_node("latents", "OVTensor"));
    module["outputs"] = outputs;

    module["params"]["model_path"] = ctx.options.model_path;
    module["type"] = "DenoiserLoopModule";
}

// ============================================================================
// VAEDecoderModule Generator (VAEDecode)
// ============================================================================

void VAEDecoderModuleGenerator::generate(YamlGeneratorContext& ctx) {
    // VAEDecode -> VAEDecoderModule (handles both VAEDecode and VAEDecodeSwitcher)
    const auto& node = ctx.current_node;
    GENAI_DEBUG("[VAEDecode] Processing node: node_id_str=%s, title=%s",
                node.node_id_str.c_str(), node.title.c_str());

    // Get referenced module name from KSampler
    std::string ksampler_module_name;
    if (auto* ksampler_node = ctx.params.get_node("KSampler")) {
        ksampler_module_name = ksampler_node->node_id_str;
    }

    std::string module_name = node.node_id_str;

    // Determine whether to enable tiling:
    // - use_tiling = 1: enable (only for Wan 2.1)
    // - use_tiling = 0: disable
    // - use_tiling = -1 (auto): enable for wan2.1 models only
    // Note: ZImage uses VAEDecodeSwitcher node for tiling, not this parameter
    bool enable_tiling = false;
    if (ctx.model_type == "wan2.1") {
        if (ctx.options.use_tiling == 1) {
            enable_tiling = true;
            GENAI_DEBUG("[YAML] VAE tiling enabled by user flag for Wan 2.1");
        } else if (ctx.options.use_tiling == 0) {
            enable_tiling = false;
            GENAI_DEBUG("[YAML] VAE tiling disabled by user flag for Wan 2.1");
        } else {
            // Auto mode: enable for Wan 2.1 by default
            enable_tiling = true;
            GENAI_DEBUG("[YAML] VAE tiling auto-enabled for Wan 2.1");
        }
    } else {
        // Non-Wan 2.1 models (e.g., ZImage): --use_tiling is ignored
        // ZImage uses VAEDecodeSwitcher node for tiling control
        if (ctx.options.use_tiling == 1) {
            GENAI_WARN("[YAML] --use_tiling is only supported for Wan 2.1 models. "
                       "For ZImage, use VAEDecodeSwitcher node with select_decoder='tiled'.");
        }
        enable_tiling = false;
    }

    // For Wan 2.1 with tiling, use VAEDecoderTilingModule
    if (enable_tiling && ctx.model_type == "wan2.1") {
        GENAI_DEBUG("[YAML] Adding VAEDecoderTilingModule (%s) for Wan 2.1", module_name.c_str());

        YAML::Node module = ctx.pipeline_modules[module_name];
        module["type"] = "VAEDecoderTilingModule";
        module["model_type"] = "wan2.1";
        module["device"] = "CPU";  // Tiling module runs on CPU

        YAML::Node inputs;
        if (!ksampler_module_name.empty()) {
            inputs.push_back(create_input_node("latent", ksampler_module_name + ".latents", "OVTensor"));
        }
        module["inputs"] = inputs;

        YAML::Node outputs;
        outputs.push_back(create_output_node("video", "OVTensor"));
        module["outputs"] = outputs;

        // Tiling parameters
        if (ctx.options.tile_size > 0) {
            module["params"]["tile_sample_min_height"] = std::to_string(ctx.options.tile_size);
            module["params"]["tile_sample_min_width"] = std::to_string(ctx.options.tile_size);
            // Stride = tile_size * 0.75 (25% overlap)
            int stride = static_cast<int>(ctx.options.tile_size * 0.75);
            module["params"]["tile_sample_stride_height"] = std::to_string(stride);
            module["params"]["tile_sample_stride_width"] = std::to_string(stride);
        } else {
            // Default tiling parameters: 256x256 tiles with 64 pixel overlap (stride=192)
            module["params"]["tile_sample_min_height"] = "256";
            module["params"]["tile_sample_min_width"] = "256";
            module["params"]["tile_sample_stride_height"] = "192";
            module["params"]["tile_sample_stride_width"] = "192";
        }
        module["params"]["spatial_compression_ratio"] = "8";
        module["params"]["sub_module_name"] = "vae_decoder";

        // Add sub_module for VAE decoder
        YAML::Node sub_module;
        sub_module["name"] = "vae_decoder";

        YAML::Node vae_decoder;
        vae_decoder["device"] = ctx.options.device;
        vae_decoder["model_type"] = "wan2.1";

        // Note: VAEDecoderTilingModule passes "latent" for VIDEO mode,
        // but VAEDecoderModule expects "latents". We need to update
        // either the module or the YAML to match.
        // For now, use "latent" as input since VAEDecoderTilingModule passes that.
        YAML::Node sub_inputs;
        YAML::Node input_node;
        input_node["name"] = "latent";
        input_node["type"] = "OVTensor";
        sub_inputs.push_back(input_node);
        vae_decoder["inputs"] = sub_inputs;

        YAML::Node sub_outputs;
        sub_outputs.push_back(create_output_node("image", "OVTensor"));
        vae_decoder["outputs"] = sub_outputs;

        vae_decoder["params"]["model_path"] = ctx.options.model_path;
        vae_decoder["type"] = "VAEDecoderModule";

        sub_module["vae_decoder"] = vae_decoder;
        ctx.root["sub_modules"].push_back(sub_module);
        return;
    }

    // Standard VAEDecoderModule for non-tiling or non-Wan 2.1 cases
    GENAI_DEBUG("[YAML] Adding VAEDecoderModule (%s)", module_name.c_str());

    YAML::Node module = ctx.pipeline_modules[module_name];
    module["device"] = ctx.options.device;

    YAML::Node inputs;
    if (!ksampler_module_name.empty()) {
        inputs.push_back(create_input_node("latents", ksampler_module_name + ".latents", "OVTensor"));
    }
    module["inputs"] = inputs;

    YAML::Node outputs;
    outputs.push_back(create_output_node("image", "OVTensor"));
    module["outputs"] = outputs;

    module["params"]["model_path"] = ctx.options.model_path;

    module["type"] = "VAEDecoderModule";
}

// ============================================================================
// VAEDecoderTilingModule Generator (VAEDecodeSwitcher)
// ============================================================================

void VAEDecoderTilingModuleGenerator::generate(YamlGeneratorContext& ctx) {
    const auto& node = ctx.current_node;
    GENAI_DEBUG("[VAEDecodeSwitcher] Processing node: node_id_str=%s, title=%s",
                node.node_id_str.c_str(), node.title.c_str());

    // Get referenced module name from KSampler
    std::string ksampler_module_name;
    if (auto* ksampler_node = ctx.params.get_node("KSampler")) {
        ksampler_module_name = ksampler_node->node_id_str;
    }

    // Check if tiled mode
    bool use_tiled = false;
    if (node.inputs.contains("select_decoder")) {
        use_tiled = (node.inputs["select_decoder"].get<std::string>() == "tiled");
    }

    std::string module_name = node.node_id_str;

    YAML::Node module = ctx.pipeline_modules[module_name];
    module["device"] = ctx.options.device;

    YAML::Node inputs;
    if (!ksampler_module_name.empty()) {
        inputs.push_back(create_input_node("latents", ksampler_module_name + ".latents", "OVTensor"));
    }
    module["inputs"] = inputs;

    YAML::Node outputs;
    outputs.push_back(create_output_node("image", "OVTensor"));
    module["outputs"] = outputs;

    module["params"]["model_path"] = ctx.options.model_path;

    if (use_tiled) {
        GENAI_DEBUG("[YAML] Adding VAEDecoderTilingModule (%s) - tiled mode", module_name.c_str());
        module["params"]["sub_module_name"] = "vae_decoder_submodule";
        module["params"]["tile_overlap_factor"] = "0.25";

        // Set default sample_size from options or node.inputs (will be overridden at runtime if tile_size input > 0)
        int default_tile_size = 0;
        if (ctx.options.tile_size > 0) {
            default_tile_size = ctx.options.tile_size;
        } else if (node.inputs.contains("tile_size")) {
            default_tile_size = node.inputs["tile_size"].get<int>();
        }
        if (default_tile_size > 0) {
            module["params"]["sample_size"] = std::to_string(default_tile_size);
        }

        module["type"] = "VAEDecoderTilingModule";

        YAML::Node sub_module;
        sub_module["name"] = "vae_decoder_submodule";

        YAML::Node vae_decoder;
        vae_decoder["device"] = ctx.options.device;

        YAML::Node sub_inputs;
        YAML::Node input_node;
        input_node["name"] = "latents";
        input_node["type"] = "OVTensor";
        sub_inputs.push_back(input_node);
        vae_decoder["inputs"] = sub_inputs;

        YAML::Node sub_outputs;
        sub_outputs.push_back(create_output_node("image", "OVTensor"));
        vae_decoder["outputs"] = sub_outputs;

        vae_decoder["params"]["enable_postprocess"] = "false";
        vae_decoder["params"]["model_path"] = ctx.options.model_path;
        vae_decoder["type"] = "VAEDecoderModule";

        sub_module["vae_decoder"] = vae_decoder;
        ctx.root["sub_modules"].push_back(sub_module);
    } else {
        GENAI_DEBUG("[YAML] Adding VAEDecoderModule (%s) - standard mode", module_name.c_str());
        module["type"] = "VAEDecoderModule";
    }
}

// ============================================================================
// SaveImageModule Generator (SaveImage)
// ============================================================================

void SaveImageModuleGenerator::generate(YamlGeneratorContext& ctx) {
    // SaveImage -> SaveImageModule
    const auto& node = ctx.current_node;
    GENAI_DEBUG("[SaveImage] Processing node: node_id_str=%s, title=%s",
                node.node_id_str.c_str(), node.title.c_str());

    // Get referenced module name from VAE
    std::string vae_module_name;
    if (auto* vae_node = ctx.params.get_node("VAEDecodeSwitcher")) {
        vae_module_name = vae_node->node_id_str;
    } else if (auto* vae_node = ctx.params.get_node("VAEDecode")) {
        vae_module_name = vae_node->node_id_str;
    }

    std::string module_name = node.node_id_str;
    GENAI_DEBUG("[YAML] Adding SaveImageModule (%s)", module_name.c_str());

    std::string filename_prefix = "ComfyUI";
    if (node.inputs.contains("filename_prefix")) {
        filename_prefix = node.inputs["filename_prefix"].get<std::string>();
    }

    YAML::Node module = ctx.pipeline_modules[module_name];
    module["device"] = "CPU";

    YAML::Node inputs;
    if (!vae_module_name.empty()) {
        inputs.push_back(create_input_node("raw_data", vae_module_name + ".image", "OVTensor"));
    }
    module["inputs"] = inputs;

    YAML::Node outputs;
    outputs.push_back(create_output_node("saved_image", "String"));
    module["outputs"] = outputs;

    module["params"]["filename_prefix"] = filename_prefix;
    module["type"] = "SaveImageModule";
}

// ============================================================================
// SaveVideoModule Generator (SaveAnimatedWEBP -> SaveVideoModule)
// ============================================================================

void SaveVideoModuleGenerator::generate(YamlGeneratorContext& ctx) {
    const auto& node = ctx.current_node;
    GENAI_DEBUG("[SaveAnimatedWEBP] Processing node: node_id_str=%s, title=%s",
                node.node_id_str.c_str(), node.title.c_str());

    // Get referenced module name from VAE
    std::string vae_module_name;
    bool is_wan21_tiling = false;
    if (auto* vae_node = ctx.params.get_node("VAEDecodeSwitcher")) {
        vae_module_name = vae_node->node_id_str;
    } else if (auto* vae_node = ctx.params.get_node("VAEDecode")) {
        vae_module_name = vae_node->node_id_str;
        // Check if this is Wan 2.1 with tiling (output is "video" not "image")
        if (ctx.model_type == "wan2.1" && ctx.options.use_tiling != 0) {
            is_wan21_tiling = true;
        }
    }

    std::string module_name = node.node_id_str;
    GENAI_DEBUG("[YAML] Adding SaveVideoModule (%s)", module_name.c_str());

    // Extract parameters from node inputs
    std::string filename_prefix = "ComfyUI";
    if (node.inputs.contains("filename_prefix")) {
        filename_prefix = node.inputs["filename_prefix"].get<std::string>();
    }

    int fps = 6;
    if (node.inputs.contains("fps")) {
        fps = static_cast<int>(node.inputs["fps"].get<double>());
    }

    int quality = 80;
    if (node.inputs.contains("quality")) {
        quality = node.inputs["quality"].get<int>();
    }

    YAML::Node module = ctx.pipeline_modules[module_name];
    module["device"] = "CPU";

    YAML::Node inputs;
    if (!vae_module_name.empty()) {
        // Wan 2.1 tiling outputs "video", others output "image"
        std::string output_name = is_wan21_tiling ? "video" : "image";
        inputs.push_back(create_input_node("raw_data", vae_module_name + "." + output_name, "OVTensor"));
    }
    module["inputs"] = inputs;

    YAML::Node outputs;
    outputs.push_back(create_output_node("saved_video", "String"));
    outputs.push_back(create_output_node("saved_videos", "VecString"));
    module["outputs"] = outputs;

    module["params"]["filename_prefix"] = filename_prefix;
    module["params"]["fps"] = std::to_string(fps);
    module["params"]["quality"] = std::to_string(quality);
    module["type"] = "SaveVideoModule";
}

// ============================================================================
// Result Module Generator (always called last)
// ============================================================================

void YamlModuleGeneratorRegistry::generate_result_module(
    YAML::Node& pipeline_modules,
    YAML::Node& root,
    const ComfyUIToGenAIConverter::PipelineParams& params,
    const ConversionOptions& options,
    const std::string& model_type) {

    GENAI_DEBUG("[YAML] Adding ResultModule (pipeline_result)");
    YAML::Node module = pipeline_modules["pipeline_result"];

    // Get referenced module names
    std::string vae_module_name;
    bool is_wan21_tiling = false;
    if (auto* vae_node = params.get_node("VAEDecodeSwitcher")) {
        vae_module_name = vae_node->node_id_str;
    } else if (auto* vae_node = params.get_node("VAEDecode")) {
        vae_module_name = vae_node->node_id_str;
        // Check if this is Wan 2.1 with tiling (output is "video" not "image")
        if (model_type == "wan2.1" && options.use_tiling != 0) {
            is_wan21_tiling = true;
        }
    }

    std::string save_image_module_name;
    if (auto* save_node = params.get_node("SaveImage")) {
        save_image_module_name = save_node->node_id_str;
    }

    std::string save_video_module_name;
    if (auto* save_node = params.get_node("SaveAnimatedWEBP")) {
        save_video_module_name = save_node->node_id_str;
    }

    YAML::Node inputs;
    // Priority: SaveAnimatedWEBP > SaveImage > VAE (only one input, fallback logic)
    if (!save_video_module_name.empty()) {
        inputs.push_back(create_input_node("saved_video", save_video_module_name + ".saved_video", "String"));
    } else if (!save_image_module_name.empty()) {
        inputs.push_back(create_input_node("saved_image", save_image_module_name + ".saved_image", "String"));
    } else if (!vae_module_name.empty()) {
        // Wan 2.1 tiling outputs "video", others output "image"
        std::string output_name = is_wan21_tiling ? "video" : "image";
        inputs.push_back(create_input_node(output_name, vae_module_name + "." + output_name, "OVTensor"));
    }
    module["inputs"] = inputs;

    YAML::Node outputs;
    if (!save_video_module_name.empty()) {
        outputs.push_back(create_output_node("saved_video", "String"));
    } else {
        outputs.push_back(create_output_node("saved_image", "String"));
    }
    module["outputs"] = outputs;

    module["type"] = "ResultModule";
}

}  // namespace comfyui
}  // namespace module
}  // namespace genai
}  // namespace ov
