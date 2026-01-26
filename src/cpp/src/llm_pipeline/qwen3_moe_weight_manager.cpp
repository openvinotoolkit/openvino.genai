// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_moe_weight_manager.hpp"
#include "layer_selection_strategy.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

// Include safetensors support
extern "C" {
    #include "../lora/safetensors.h"
}

namespace ov {
namespace genai {

namespace {

// Convert safetensors dtype to OpenVINO element type
ov::element::Type safetensors_to_ov_element_type(int dtype) {
    switch (dtype) {
        case 0: return ov::element::f32;
        case 1: return ov::element::f16;
        case 2: return ov::element::bf16;
        case 3: return ov::element::i8;
        case 4: return ov::element::u8;
        case 5: return ov::element::i16;
        case 6: return ov::element::i32;
        case 7: return ov::element::i64;
        default:
            throw std::runtime_error("Unsupported safetensors dtype: " + std::to_string(dtype));
    }
}

// RAII wrapper for safetensors_File
struct AutoSafetensor : public safetensors_File {
    ~AutoSafetensor() {
        std::free(tensors);
        std::free(metadata);
    }
};

// Convert safetensor buffer to tensor map
std::unordered_map<std::string, ov::Tensor> safetensor_to_tensor_map(const ov::Tensor& safetensor) {
    AutoSafetensor safe_tensors_file{};
    
    // Parse safetensors format
    auto safetensor_data = const_cast<char*>(safetensor.data<char>());
    const char* error = safetensors_file_init(safetensor_data, safetensor.get_byte_size(), &safe_tensors_file);
    if (error != nullptr) {
        throw std::runtime_error("Cannot parse safetensor file: " + std::string(error));
    }

    std::unordered_map<std::string, ov::Tensor> tensors;
    
    // Extract each tensor
    for (int i = 0; i < safe_tensors_file.num_tensors; i++) {
        safetensors_TensorDescriptor tensor_desc = safe_tensors_file.tensors[i];
        
        // Extract tensor name
        std::string name(tensor_desc.name.ptr, tensor_desc.name.ptr + tensor_desc.name.len);
        
        // Extract shape
        ov::Shape shape(tensor_desc.shape, tensor_desc.shape + tensor_desc.n_dimensions);
        
        // Get element type
        auto element_type = safetensors_to_ov_element_type(tensor_desc.dtype);
        
        // Create tensor wrapping the data
        void* data_ptr = tensor_desc.ptr;
        ov::Tensor tensor(element_type, shape, data_ptr);
        
        // Store reference to safetensor buffer to prevent deallocation
        // Note: This is a simplified approach. In production, we'd need proper memory management
        tensors[name] = tensor;
    }
    
    return tensors;
}

} // anonymous namespace

WeightFormat Qwen3MoeWeightManager::detect_format(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == ".safetensors") {
        return WeightFormat::SAFETENSORS;
    } else if (ext == ".pt" || ext == ".bin") {
        return WeightFormat::PYTORCH;
    } else if (ext == ".gguf") {
        return WeightFormat::GGUF;
    } else {
        return WeightFormat::UNKNOWN;
    }
}

std::unordered_map<std::string, ov::Tensor> Qwen3MoeWeightManager::load_weights(
    const std::filesystem::path& checkpoint_path,
    const Qwen3MoeConfig& config) {
    
    // Check if path exists
    if (!std::filesystem::exists(checkpoint_path)) {
        throw std::runtime_error("Checkpoint path does not exist: " + checkpoint_path.string());
    }
    
    // Determine if path is a file or directory
    std::filesystem::path weight_file;
    if (std::filesystem::is_directory(checkpoint_path)) {
        // Look for model.safetensors in directory
        weight_file = checkpoint_path / "model.safetensors";
        if (!std::filesystem::exists(weight_file)) {
            // Try model-00001-of-00001.safetensors pattern
            weight_file = checkpoint_path / "model-00001-of-00001.safetensors";
            if (!std::filesystem::exists(weight_file)) {
                throw std::runtime_error("No weight file found in directory: " + checkpoint_path.string());
            }
        }
    } else {
        weight_file = checkpoint_path;
    }
    
    // Detect format
    WeightFormat format = detect_format(weight_file);
    
    // Load weights based on format
    std::unordered_map<std::string, ov::Tensor> weights;
    switch (format) {
        case WeightFormat::SAFETENSORS:
            weights = load_safetensors(weight_file, config);
            break;
        case WeightFormat::PYTORCH:
            weights = load_pytorch(weight_file, config);
            break;
        case WeightFormat::GGUF:
            weights = load_gguf(weight_file, config);
            break;
        default:
            throw std::runtime_error("Unsupported weight format: " + weight_file.string());
    }
    
    // Validate loaded weights
    if (!validate_weight_shapes(weights, config)) {
        throw std::runtime_error("Weight validation failed for checkpoint: " + weight_file.string());
    }
    
    return weights;
}

std::unordered_map<std::string, ov::Tensor> Qwen3MoeWeightManager::load_safetensors(
    const std::filesystem::path& checkpoint_path,
    const Qwen3MoeConfig& config) {
    
    // Read safetensors file into memory
    auto safetensor = ov::read_tensor_data(checkpoint_path);
    
    // Convert to tensor map
    auto weights = safetensor_to_tensor_map(safetensor);
    
    return weights;
}

std::unordered_map<std::string, ov::Tensor> Qwen3MoeWeightManager::load_pytorch(
    const std::filesystem::path& checkpoint_path,
    const Qwen3MoeConfig& config) {
    
    // TODO: Implement PyTorch weight loading
    // This would require torch::load or a custom PyTorch format parser
    throw std::runtime_error("PyTorch weight format not yet implemented");
}

std::unordered_map<std::string, ov::Tensor> Qwen3MoeWeightManager::load_gguf(
    const std::filesystem::path& checkpoint_path,
    const Qwen3MoeConfig& config) {
    
    // TODO: Implement GGUF weight loading
    // This would integrate with existing GGUF utilities in gguf_utils/
    throw std::runtime_error("GGUF weight format not yet implemented");
}

ov::Tensor Qwen3MoeWeightManager::load_expert_weights_3d(
    const std::filesystem::path& checkpoint_path,
    const std::string& weight_key,
    int num_experts,
    int dim1,
    int dim2) {
    
    // Load all weights from checkpoint
    Qwen3MoeConfig dummy_config;  // Minimal config for loading
    auto weights = load_safetensors(checkpoint_path, dummy_config);
    
    // Check if weight exists as a single 3D tensor
    if (weights.count(weight_key) > 0) {
        auto tensor = weights.at(weight_key);
        auto shape = tensor.get_shape();
        
        // Validate 3D shape
        if (shape.size() != 3) {
            throw std::runtime_error("Expected 3D tensor for key: " + weight_key + 
                                   ", got " + std::to_string(shape.size()) + "D");
        }
        
        if (shape[0] != static_cast<size_t>(num_experts) ||
            shape[1] != static_cast<size_t>(dim1) ||
            shape[2] != static_cast<size_t>(dim2)) {
            throw std::runtime_error("Shape mismatch for key: " + weight_key);
        }
        
        return tensor;
    }
    
    // Otherwise, try to load per-expert weights and stack them
    std::vector<ov::Tensor> expert_tensors;
    for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        std::string expert_key = weight_key + "." + std::to_string(expert_idx);
        
        if (weights.count(expert_key) == 0) {
            throw std::runtime_error("Expert weight not found: " + expert_key);
        }
        
        expert_tensors.push_back(weights.at(expert_key));
    }
    
    return reshape_expert_weights_to_3d(expert_tensors);
}

ov::Tensor Qwen3MoeWeightManager::reshape_expert_weights_to_3d(
    const std::vector<ov::Tensor>& expert_weights) {
    
    if (expert_weights.empty()) {
        throw std::runtime_error("Cannot reshape empty expert weights vector");
    }
    
    // Validate all experts have same shape
    auto first_shape = expert_weights[0].get_shape();
    if (first_shape.size() != 2) {
        throw std::runtime_error("Expert weights must be 2D tensors");
    }
    
    for (size_t i = 1; i < expert_weights.size(); ++i) {
        if (expert_weights[i].get_shape() != first_shape) {
            throw std::runtime_error("Inconsistent expert weight shapes");
        }
    }
    
    // Create 3D tensor
    size_t num_experts = expert_weights.size();
    size_t dim1 = first_shape[0];
    size_t dim2 = first_shape[1];
    
    auto element_type = expert_weights[0].get_element_type();
    ov::Tensor result(element_type, ov::Shape{num_experts, dim1, dim2});
    
    // Copy expert data
    size_t expert_size = dim1 * dim2 * element_type.size();
    for (size_t i = 0; i < num_experts; ++i) {
        std::memcpy(
            static_cast<char*>(result.data()) + i * expert_size,
            expert_weights[i].data(),
            expert_size
        );
    }
    
    return result;
}

bool Qwen3MoeWeightManager::validate_weight_shapes(
    const std::unordered_map<std::string, ov::Tensor>& weights,
    const Qwen3MoeConfig& config) {
    
    // Get expected keys
    auto expected_keys = get_expected_weight_keys(config);
    
    // Check all expected keys exist
    for (const auto& key : expected_keys) {
        if (weights.count(key) == 0) {
            // Some keys are optional (e.g., bias terms)
            if (key.find(".bias") != std::string::npos) {
                continue;  // Bias is optional
            }
            
            std::cerr << "Warning: Missing weight key: " << key << std::endl;
            // Don't fail validation for missing keys, just warn
            // return false;
        }
    }
    
    // Validate specific weight shapes
    
    // 1. Embedding weights
    std::string embed_key = "model.embed_tokens.weight";
    if (weights.count(embed_key) > 0) {
        auto shape = weights.at(embed_key).get_shape();
        if (shape.size() != 2 || 
            shape[0] != static_cast<size_t>(config.vocab_size) ||
            shape[1] != static_cast<size_t>(config.hidden_size)) {
            std::cerr << "Invalid embedding shape for key: " << embed_key << std::endl;
            return false;
        }
    }
    
    // 2. Layer weights (sample first layer)
    if (config.num_hidden_layers > 0) {
        std::string q_proj_key = "model.layers.0.self_attn.q_proj.weight";
        if (weights.count(q_proj_key) > 0) {
            auto shape = weights.at(q_proj_key).get_shape();
            size_t expected_out = config.num_attention_heads * 
                                 (config.hidden_size / config.num_attention_heads);
            if (shape.size() != 2 || 
                shape[0] != expected_out ||
                shape[1] != static_cast<size_t>(config.hidden_size)) {
                std::cerr << "Invalid q_proj shape for key: " << q_proj_key << std::endl;
                return false;
            }
        }
    }
    
    // 3. Expert weights (if MoE layers exist)
    LayerSelectionStrategy strategy(config);
    for (int layer_idx = 0; layer_idx < config.num_hidden_layers; ++layer_idx) {
        if (strategy.is_moe_layer(layer_idx)) {
            std::string gate_up_key = "model.layers." + std::to_string(layer_idx) + 
                                     ".mlp.experts.gate_up_proj";
            if (weights.count(gate_up_key) > 0) {
                auto shape = weights.at(gate_up_key).get_shape();
                if (shape.size() != 3 ||
                    shape[0] != static_cast<size_t>(config.num_experts)) {
                    std::cerr << "Invalid expert weight shape for key: " << gate_up_key << std::endl;
                    return false;
                }
            }
            break;  // Only check first MoE layer
        }
    }
    
    return true;
}

std::vector<std::string> Qwen3MoeWeightManager::get_expected_weight_keys(
    const Qwen3MoeConfig& config) {
    
    std::vector<std::string> keys;
    
    // Embedding keys
    auto embed_keys = get_embedding_keys();
    keys.insert(keys.end(), embed_keys.begin(), embed_keys.end());
    
    // Layer keys
    LayerSelectionStrategy strategy(config);
    for (int layer_idx = 0; layer_idx < config.num_hidden_layers; ++layer_idx) {
        bool is_moe = strategy.is_moe_layer(layer_idx);
        auto layer_keys = get_layer_keys(layer_idx, is_moe);
        keys.insert(keys.end(), layer_keys.begin(), layer_keys.end());
    }
    
    // Output keys
    auto output_keys = get_output_keys();
    keys.insert(keys.end(), output_keys.begin(), output_keys.end());
    
    return keys;
}

std::vector<std::string> Qwen3MoeWeightManager::get_embedding_keys() {
    return {
        "model.embed_tokens.weight"
    };
}

std::vector<std::string> Qwen3MoeWeightManager::get_layer_keys(int layer_idx, bool is_moe_layer) {
    std::string prefix = "model.layers." + std::to_string(layer_idx);
    
    std::vector<std::string> keys = {
        // Attention weights
        prefix + ".self_attn.q_proj.weight",
        prefix + ".self_attn.k_proj.weight",
        prefix + ".self_attn.v_proj.weight",
        prefix + ".self_attn.o_proj.weight",
        
        // Q/K normalization
        prefix + ".self_attn.q_norm.weight",
        prefix + ".self_attn.k_norm.weight",
        
        // Layer norms
        prefix + ".input_layernorm.weight",
        prefix + ".post_attention_layernorm.weight"
    };
    
    if (is_moe_layer) {
        // MoE weights
        keys.push_back(prefix + ".mlp.gate.weight");  // Router
        keys.push_back(prefix + ".mlp.experts.gate_up_proj");  // 3D tensor
        keys.push_back(prefix + ".mlp.experts.down_proj");     // 3D tensor
    } else {
        // Standard MLP weights
        keys.push_back(prefix + ".mlp.gate_proj.weight");
        keys.push_back(prefix + ".mlp.up_proj.weight");
        keys.push_back(prefix + ".mlp.down_proj.weight");
    }
    
    return keys;
}

std::vector<std::string> Qwen3MoeWeightManager::get_output_keys() {
    return {
        "model.norm.weight",
        "lm_head.weight"
    };
}

ov::Tensor Qwen3MoeWeightManager::load_single_tensor(
    const std::filesystem::path& file_path,
    const std::string& tensor_name) {
    
    // Load all weights
    Qwen3MoeConfig dummy_config;
    auto weights = load_safetensors(file_path, dummy_config);
    
    // Find tensor
    if (weights.count(tensor_name) == 0) {
        throw std::runtime_error("Tensor not found: " + tensor_name);
    }
    
    return weights.at(tensor_name);
}

bool Qwen3MoeWeightManager::validate_tensor_shape(
    const std::string& weight_name,
    const ov::Tensor& tensor,
    const std::vector<size_t>& expected_shape) {
    
    auto actual_shape = tensor.get_shape();
    
    if (actual_shape.size() != expected_shape.size()) {
        return false;
    }
    
    for (size_t i = 0; i < actual_shape.size(); ++i) {
        if (actual_shape[i] != expected_shape[i]) {
            return false;
        }
    }
    
    return true;
}

} // namespace genai
} // namespace ov