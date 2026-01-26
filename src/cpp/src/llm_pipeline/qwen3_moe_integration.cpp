// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_moe_integration.hpp"
#include "qwen3_moe_config.hpp"
#include "qwen3_moe_graph_builder.hpp"
#include "qwen3_moe_weight_manager.hpp"
#include "layer_selection_strategy.hpp"
#include "../utils.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ov {
namespace genai {

namespace {

/**
 * @brief Helper function to check if a file exists.
 */
bool file_exists(const std::filesystem::path& path) {
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

/**
 * @brief Helper function to check if a directory exists.
 */
bool directory_exists(const std::filesystem::path& path) {
    return std::filesystem::exists(path) && std::filesystem::is_directory(path);
}

/**
 * @brief Helper function to find config.json in model path.
 */
std::filesystem::path find_config_json(const std::filesystem::path& model_path) {
    if (std::filesystem::is_directory(model_path)) {
        auto config_path = model_path / "config.json";
        if (file_exists(config_path)) {
            return config_path;
        }
        throw std::runtime_error("config.json not found in directory: " + model_path.string());
    } else if (file_exists(model_path) && model_path.filename() == "config.json") {
        return model_path;
    } else {
        throw std::runtime_error("Invalid model path: " + model_path.string());
    }
}

/**
 * @brief Helper function to find weights in model path.
 */
std::filesystem::path find_weights_path(const std::filesystem::path& model_path) {
    if (std::filesystem::is_directory(model_path)) {
        // Look for common weight file patterns
        std::vector<std::string> weight_patterns = {
            "model.safetensors",
            "model-00001-of-00001.safetensors",
            "pytorch_model.bin",
            "model.bin"
        };
        
        for (const auto& pattern : weight_patterns) {
            auto weight_path = model_path / pattern;
            if (file_exists(weight_path)) {
                return weight_path;
            }
        }
        
        // If no single file found, return directory (for sharded weights)
        return model_path;
    } else if (file_exists(model_path)) {
        return model_path;
    } else {
        throw std::runtime_error("Weights not found in path: " + model_path.string());
    }
}

} // anonymous namespace

std::shared_ptr<ov::Model> build_qwen3_moe_model(
    const std::filesystem::path& config_path,
    const std::filesystem::path& weights_path,
    const std::string& device) {
    
    // 1. Parse configuration
    auto config_file = find_config_json(config_path);
    Qwen3MoeConfig config(config_file);
    
    // 2. Load weights
    Qwen3MoeWeightManager weight_mgr;
    auto weights_file = find_weights_path(weights_path);
    auto weights = weight_mgr.load_weights(weights_file, config);
    
    // 3. Create graph builder
    Qwen3MoeGraphBuilder builder(config);
    
    // 4. Pass weights to builder
    builder.set_weights(weights);
    
    // 5. Build model
    auto model = builder.build_graph();
    
    // 6. Validate model
    if (!model) {
        throw std::runtime_error("Failed to build Qwen3-MoE model");
    }
    
    // Check that model has expected inputs and outputs
    auto inputs = model->inputs();
    auto outputs = model->outputs();
    
    if (inputs.empty()) {
        throw std::runtime_error("Model has no inputs");
    }
    
    if (outputs.empty()) {
        throw std::runtime_error("Model has no outputs");
    }
    
    // 7. Return model
    return model;
}

ov::CompiledModel load_and_compile_qwen3_moe(
    const std::filesystem::path& model_path,
    const std::string& device,
    const ov::AnyMap& config) {
    
    // 1. Create OpenVINO Core
    ov::Core& core = ov::genai::utils::singleton_core();
    
    // 2. Build model
    auto model = build_qwen3_moe_model(model_path, model_path, device);
    
    // 3. Apply optimizations from config
    ov::AnyMap compilation_config = config;
    
    // Set default precision hints if not specified
    if (compilation_config.find(ov::hint::inference_precision.name()) == compilation_config.end()) {
        // Use f16 for better performance on most devices
        compilation_config[ov::hint::inference_precision.name()] = ov::element::f16;
    }
    
    // Enable dynamic shapes if needed
    if (compilation_config.find(ov::hint::dynamic_quantization_group_size.name()) == compilation_config.end()) {
        // Dynamic shapes are enabled by default in the model
    }
    
    // 4. Compile model
    auto compiled = core.compile_model(model, device, compilation_config);
    
    // 5. Return compiled model
    return compiled;
}

std::shared_ptr<LLMPipeline> create_qwen3_moe_pipeline(
    const std::filesystem::path& model_path,
    const std::string& device,
    const ov::AnyMap& properties,
    const ov::genai::GenerationConfig& generation_config) {
    
    // 1. Build and compile model
    auto compiled = load_and_compile_qwen3_moe(model_path, device, properties);
    
    // 2. Load tokenizer
    ov::genai::Tokenizer tokenizer(model_path, properties);
    
    // 3. Create LLM pipeline
    // Use the compiled model and tokenizer to create pipeline
    auto infer_request = compiled.create_infer_request();
    
    // Determine generation config
    ov::genai::GenerationConfig final_config = generation_config;
    
    // If generation config is default, try to load from file
    if (final_config.max_new_tokens == SIZE_MAX && final_config.max_length == SIZE_MAX) {
        auto gen_config_path = model_path / "generation_config.json";
        if (file_exists(gen_config_path)) {
            final_config = ov::genai::GenerationConfig(gen_config_path);
        }
    }
    
    // Create pipeline from infer request and tokenizer
    auto pipeline = std::make_shared<LLMPipeline>(infer_request, tokenizer, final_config);
    
    // 4. Return pipeline
    return pipeline;
}

Qwen3MoeModelMetadata get_qwen3_moe_metadata(
    const std::filesystem::path& model_path) {
    
    // 1. Parse config to get model structure
    auto config_file = find_config_json(model_path);
    Qwen3MoeConfig config(config_file);
    
    Qwen3MoeModelMetadata metadata;
    
    // 2. Set basic metadata
    metadata.model_type = "qwen3_moe";
    metadata.vocab_size = config.vocab_size;
    metadata.hidden_size = config.hidden_size;
    metadata.num_hidden_layers = config.num_hidden_layers;
    metadata.num_experts = config.num_experts;
    metadata.num_experts_per_tok = config.num_experts_per_tok;
    
    // 3. Compute total parameters
    size_t total_params = 0;
    
    // Embedding parameters
    total_params += static_cast<size_t>(config.vocab_size) * config.hidden_size;
    
    // Per-layer parameters
    for (int layer_idx = 0; layer_idx < config.num_hidden_layers; ++layer_idx) {
        // Attention parameters
        size_t attn_params = 0;
        attn_params += static_cast<size_t>(config.num_attention_heads) * 
                       (config.hidden_size / config.num_attention_heads) * config.hidden_size; // Q projection
        attn_params += static_cast<size_t>(config.num_key_value_heads) * 
                       (config.hidden_size / config.num_attention_heads) * config.hidden_size; // K projection
        attn_params += static_cast<size_t>(config.num_key_value_heads) * 
                       (config.hidden_size / config.num_attention_heads) * config.hidden_size; // V projection
        attn_params += static_cast<size_t>(config.num_attention_heads) * 
                       (config.hidden_size / config.num_attention_heads) * config.hidden_size; // O projection
        
        // Q/K norm parameters (head_dim for each)
        size_t head_dim = config.hidden_size / config.num_attention_heads;
        attn_params += head_dim * 2; // Q norm and K norm
        
        total_params += attn_params;
        
        // Layer norm parameters
        total_params += config.hidden_size * 2; // input_layernorm and post_attention_layernorm
        
        // MLP or MoE parameters
        if (config.is_moe_layer(layer_idx)) {
            // MoE parameters
            size_t moe_params = 0;
            
            // Router parameters
            moe_params += static_cast<size_t>(config.num_experts) * config.hidden_size;
            
            // Expert parameters (3D tensors)
            // gate_up_proj: [num_experts, 2*moe_intermediate_size, hidden_size]
            moe_params += static_cast<size_t>(config.num_experts) * 
                         (2 * config.moe_intermediate_size) * config.hidden_size;
            
            // down_proj: [num_experts, hidden_size, moe_intermediate_size]
            moe_params += static_cast<size_t>(config.num_experts) * 
                         config.hidden_size * config.moe_intermediate_size;
            
            total_params += moe_params;
        } else {
            // Standard MLP parameters
            size_t mlp_params = 0;
            mlp_params += static_cast<size_t>(config.intermediate_size) * config.hidden_size; // gate_proj
            mlp_params += static_cast<size_t>(config.intermediate_size) * config.hidden_size; // up_proj
            mlp_params += static_cast<size_t>(config.hidden_size) * config.intermediate_size; // down_proj
            
            total_params += mlp_params;
        }
    }
    
    // Final norm parameters
    total_params += config.hidden_size;
    
    // LM head parameters (if not tied with embeddings)
    if (!config.tie_word_embeddings) {
        total_params += static_cast<size_t>(config.vocab_size) * config.hidden_size;
    }
    
    metadata.num_parameters = total_params;
    
    // 4. Compute active parameters (parameters used per forward pass)
    size_t active_params = 0;
    
    // Embedding parameters (always active)
    active_params += static_cast<size_t>(config.vocab_size) * config.hidden_size;
    
    // Per-layer active parameters
    for (int layer_idx = 0; layer_idx < config.num_hidden_layers; ++layer_idx) {
        // Attention parameters (always active)
        size_t attn_params = 0;
        attn_params += static_cast<size_t>(config.num_attention_heads) * 
                       (config.hidden_size / config.num_attention_heads) * config.hidden_size;
        attn_params += static_cast<size_t>(config.num_key_value_heads) * 
                       (config.hidden_size / config.num_attention_heads) * config.hidden_size;
        attn_params += static_cast<size_t>(config.num_key_value_heads) * 
                       (config.hidden_size / config.num_attention_heads) * config.hidden_size;
        attn_params += static_cast<size_t>(config.num_attention_heads) * 
                       (config.hidden_size / config.num_attention_heads) * config.hidden_size;
        
        size_t head_dim = config.hidden_size / config.num_attention_heads;
        attn_params += head_dim * 2;
        
        active_params += attn_params;
        
        // Layer norm parameters (always active)
        active_params += config.hidden_size * 2;
        
        // MLP or MoE parameters
        if (config.is_moe_layer(layer_idx)) {
            // MoE active parameters (only num_experts_per_tok experts are active)
            size_t moe_active_params = 0;
            
            // Router parameters (always active)
            moe_active_params += static_cast<size_t>(config.num_experts) * config.hidden_size;
            
            // Active expert parameters
            size_t params_per_expert = (2 * config.moe_intermediate_size * config.hidden_size) +
                                      (config.hidden_size * config.moe_intermediate_size);
            moe_active_params += static_cast<size_t>(config.num_experts_per_tok) * params_per_expert;
            
            active_params += moe_active_params;
        } else {
            // Standard MLP parameters (all active)
            size_t mlp_params = 0;
            mlp_params += static_cast<size_t>(config.intermediate_size) * config.hidden_size;
            mlp_params += static_cast<size_t>(config.intermediate_size) * config.hidden_size;
            mlp_params += static_cast<size_t>(config.hidden_size) * config.intermediate_size;
            
            active_params += mlp_params;
        }
    }
    
    // Final norm parameters (always active)
    active_params += config.hidden_size;
    
    // LM head parameters (always active)
    if (!config.tie_word_embeddings) {
        active_params += static_cast<size_t>(config.vocab_size) * config.hidden_size;
    } else {
        active_params += static_cast<size_t>(config.vocab_size) * config.hidden_size;
    }
    
    metadata.num_active_parameters = active_params;
    
    // 5. Count layer types using LayerSelectionStrategy
    LayerSelectionStrategy strategy(config);
    int num_moe = 0;
    int num_mlp = 0;
    
    for (int layer_idx = 0; layer_idx < config.num_hidden_layers; ++layer_idx) {
        if (strategy.is_moe_layer(layer_idx)) {
            num_moe++;
        } else {
            num_mlp++;
        }
    }
    
    metadata.num_moe_layers = num_moe;
    metadata.num_mlp_layers = num_mlp;
    
    // 6. Return metadata
    return metadata;
}

bool validate_qwen3_moe_checkpoint(
    const std::filesystem::path& checkpoint_path) {
    
    try {
        // 1. Check that checkpoint path exists
        if (!std::filesystem::exists(checkpoint_path)) {
            return false;
        }
        
        // 2. Check for config.json
        auto config_path = checkpoint_path / "config.json";
        if (!file_exists(config_path)) {
            return false;
        }
        
        // 3. Validate config.json format
        try {
            Qwen3MoeConfig config(config_path);
            config.validate();
        } catch (const std::exception&) {
            return false;
        }
        
        // 4. Check for weight files
        bool has_weights = false;
        
        // Check for safetensors format
        if (file_exists(checkpoint_path / "model.safetensors") ||
            file_exists(checkpoint_path / "model-00001-of-00001.safetensors")) {
            has_weights = true;
        }
        
        // Check for PyTorch format
        if (file_exists(checkpoint_path / "pytorch_model.bin") ||
            file_exists(checkpoint_path / "model.bin")) {
            has_weights = true;
        }
        
        // Check for GGUF format
        if (file_exists(checkpoint_path / "model.gguf")) {
            has_weights = true;
        }
        
        if (!has_weights) {
            return false;
        }
        
        // 5. Check for tokenizer files
        bool has_tokenizer = false;
        
        if (file_exists(checkpoint_path / "tokenizer.json") ||
            file_exists(checkpoint_path / "tokenizer_config.json")) {
            has_tokenizer = true;
        }
        
        // Tokenizer is optional for validation, but log warning if missing
        if (!has_tokenizer) {
            // Note: In production, you might want to log this warning
            // For now, we don't fail validation
        }
        
        return true;
        
    } catch (const std::exception&) {
        return false;
    }
}

void convert_hf_checkpoint_to_ov(
    const std::filesystem::path& hf_path,
    const std::filesystem::path& ov_path,
    bool compress_weights) {
    
    // 1. Validate HuggingFace checkpoint
    if (!validate_qwen3_moe_checkpoint(hf_path)) {
        throw std::runtime_error("Invalid HuggingFace checkpoint: " + hf_path.string());
    }
    
    // 2. Create output directory
    if (!directory_exists(ov_path)) {
        std::filesystem::create_directories(ov_path);
    }
    
    // 3. Load HuggingFace checkpoint
    auto config_file = find_config_json(hf_path);
    Qwen3MoeConfig config(config_file);
    
    Qwen3MoeWeightManager weight_mgr;
    auto weights_file = find_weights_path(hf_path);
    auto weights = weight_mgr.load_weights(weights_file, config);
    
    // 4. Build OpenVINO model
    Qwen3MoeGraphBuilder builder(config);
    builder.set_weights(weights);
    auto model = builder.build_graph();
    
    // 5. Apply weight compression if requested
    if (compress_weights) {
        // Note: Weight compression would be implemented here
        // This could use OpenVINO's weight compression utilities
        // For now, we save the model as-is
    }
    
    // 6. Serialize model to OpenVINO IR format
    auto model_xml_path = ov_path / "openvino_model.xml";
    auto model_bin_path = ov_path / "openvino_model.bin";
    
    ov::serialize(model, model_xml_path.string(), model_bin_path.string());
    
    // 7. Copy configuration files
    auto src_config = hf_path / "config.json";
    auto dst_config = ov_path / "config.json";
    std::filesystem::copy_file(src_config, dst_config, std::filesystem::copy_options::overwrite_existing);
    
    // 8. Copy tokenizer files if they exist
    std::vector<std::string> tokenizer_files = {
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json"
    };
    
    for (const auto& file : tokenizer_files) {
        auto src_file = hf_path / file;
        if (file_exists(src_file)) {
            auto dst_file = ov_path / file;
            std::filesystem::copy_file(src_file, dst_file, std::filesystem::copy_options::overwrite_existing);
        }
    }
    
    // 9. Copy generation_config.json if it exists
    auto src_gen_config = hf_path / "generation_config.json";
    if (file_exists(src_gen_config)) {
        auto dst_gen_config = ov_path / "generation_config.json";
        std::filesystem::copy_file(src_gen_config, dst_gen_config, std::filesystem::copy_options::overwrite_existing);
    }
}

void register_qwen3_moe_model_type() {
    // Note: Currently OpenVINO GenAI does not have a formal model type registry.
    // This function is reserved for future use when such infrastructure is added.
    // 
    // When implemented, this would:
    // 1. Register "qwen3_moe" as a recognized model type
    // 2. Associate it with the build_qwen3_moe_model builder function
    // 3. Set default generation config for Qwen3-MoE models
    // 4. Register model detection patterns (e.g., checking config.json for model_type="qwen3_moe")
    
    // Placeholder implementation - does nothing for now
}

}  // namespace genai
}  // namespace ov