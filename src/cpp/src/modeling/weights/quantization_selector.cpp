// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling/weights/quantization_selector.hpp"
#include <algorithm>
#include <iostream>

namespace ov {
namespace genai {
namespace modeling {
namespace weights {

QuantizationSelector::QuantizationSelector(const QuantizationConfig& config)
    : config_(config) {
    // Apply legacy options to selection config
    config_.apply_legacy_options();
    // exclude patterns default
    config_.selection.exclude_patterns.push_back("*visual.patch_embed*");
}

bool QuantizationSelector::should_quantize(const std::string& name, 
                                          const ov::Shape& shape,
                                          ov::element::Type dtype) const {
    // Skip if quantization is disabled
    if (!config_.enabled()) {
        return false;
    }
    
    const auto& sel = config_.selection;
    
    // Only quantize 2D weight matrices if configured
    if (sel.only_2d_weights && shape.size() != 2) {
        if (sel.verbose) {
            std::cout << "[QuantizationSelector] Skipping " << name << " - not 2D" << std::endl;
        }
        return false;
    }
    
    // Only quantize FP16/FP32/BF16/FP8 weights if dtype is specified
    if (dtype != ov::element::undefined) {
        if (dtype != ov::element::f16 && dtype != ov::element::f32 && 
            dtype != ov::element::bf16 && dtype != ov::element::f8e4m3) {
            if (sel.verbose) {
                std::cout << "[QuantizationSelector] Skipping " << name << " - not float type" << std::endl;
            }
            return false;
        }
    }
    
    // 1. Check size threshold
    size_t weight_elements = 1;
    for (auto dim : shape) {
        weight_elements *= dim;
    }
    
    if (weight_elements < sel.min_weight_size) {
        if (sel.verbose) {
            std::cout << "[QuantizationSelector] Skipping " << name << " - too small (" 
                      << weight_elements << " < " << sel.min_weight_size << ")" << std::endl;
        }
        return false;
    }
    
    if (sel.max_weight_size > 0 && weight_elements > sel.max_weight_size) {
        if (sel.verbose) {
            std::cout << "[QuantizationSelector] Skipping " << name << " - too large (" 
                      << weight_elements << " > " << sel.max_weight_size << ")" << std::endl;
        }
        return false;
    }
    
    // 2. Check explicit exclude list (highest priority)
    if (std::find(sel.exclude_weights.begin(), sel.exclude_weights.end(), name) 
        != sel.exclude_weights.end()) {
        if (sel.verbose) {
            std::cout << "[QuantizationSelector] Skipping " << name << " - in exclude list" << std::endl;
        }
        return false;
    }
    
    // 3. Check explicit include list (second priority)
    if (!sel.include_weights.empty()) {
        if (std::find(sel.include_weights.begin(), sel.include_weights.end(), name) 
            != sel.include_weights.end()) {
            if (sel.verbose) {
                std::cout << "[QuantizationSelector] Quantizing " << name << " - in include list" << std::endl;
            }
            return true;
        }
        // If include list is specified but weight not in it, continue checking other criteria
    }
    
    // 4. Check exclude patterns
    if (sel.matches_pattern(name, sel.exclude_patterns)) {
        if (sel.verbose) {
            std::cout << "[QuantizationSelector] Skipping " << name << " - matches exclude pattern" << std::endl;
        }
        return false;
    }
    
    // 5. Check include patterns
    if (!sel.include_patterns.empty()) {
        if (!sel.matches_pattern(name, sel.include_patterns)) {
            if (sel.verbose) {
                std::cout << "[QuantizationSelector] Skipping " << name << " - doesn't match include pattern" << std::endl;
            }
            return false;  // If patterns specified but didn't match, skip
        }
    }
    
    // 6. Check layer range
    if (sel.layer_range.has_value()) {
        auto layer_idx = sel.extract_layer_index(name);
        if (layer_idx.has_value()) {
            int idx = layer_idx.value();
            if (idx < sel.layer_range->first || idx > sel.layer_range->second) {
                if (sel.verbose) {
                    std::cout << "[QuantizationSelector] Skipping " << name << " - layer " 
                              << idx << " outside range [" << sel.layer_range->first 
                              << ", " << sel.layer_range->second << "]" << std::endl;
                }
                return false;
            }
        }
    }
    
    // 7. Check type-based flags
    bool is_attention = name.find("attn") != std::string::npos || 
                        name.find("attention") != std::string::npos ||
                        name.find("q_proj") != std::string::npos ||
                        name.find("k_proj") != std::string::npos ||
                        name.find("v_proj") != std::string::npos ||
                        name.find("o_proj") != std::string::npos;
    bool is_mlp = name.find("mlp") != std::string::npos || 
                  name.find("fc") != std::string::npos ||
                  name.find("feed_forward") != std::string::npos ||
                  name.find("gate_proj") != std::string::npos ||
                  name.find("up_proj") != std::string::npos ||
                  name.find("down_proj") != std::string::npos;
    bool is_embedding = name.find("embed") != std::string::npos;
    bool is_lm_head = name.find("lm_head") != std::string::npos ||
                      name.find("output") != std::string::npos;
    bool is_norm = name.find("norm") != std::string::npos ||
                   name.find("ln") != std::string::npos ||
                   name.find("layer_norm") != std::string::npos;
    bool is_bias = name.find(".bias") != std::string::npos;
    bool is_moe = name.find("moe.gate_exps") != std::string::npos ||
                  name.find("moe.up_exps") != std::string::npos ||
                  name.find("moe.down_exps") != std::string::npos ||
                  name.find(".mlp.experts.") != std::string::npos;
    bool is_router = name.find(".mlp.gate.weight") != std::string::npos ||
                     name.find(".moe.gate_inp") != std::string::npos ||
                     name.find("shared_expert_gate") != std::string::npos;  // Skip shared_expert_gate (small BF16 weight)

    
    // Skip bias and norm by default
    if (is_bias || (is_norm && !sel.quantize_norm)) {
        if (sel.verbose) {
            std::cout << "[QuantizationSelector] Skipping " << name << " - bias or norm" << std::endl;
        }
        return false;
    }
    
    if (is_attention && !sel.quantize_attention) {
        if (sel.verbose) {
            std::cout << "[QuantizationSelector] Skipping " << name << " - attention disabled" << std::endl;
        }
        return false;
    }
    if (is_mlp && !sel.quantize_mlp) {
        if (sel.verbose) {
            std::cout << "[QuantizationSelector] Skipping " << name << " - MLP disabled" << std::endl;
        }
        return false;
    }
    if (is_embedding && !sel.quantize_embeddings) {
        if (sel.verbose) {
            std::cout << "[QuantizationSelector] Skipping " << name << " - embeddings disabled" << std::endl;
        }
        return false;
    }
    if (is_lm_head && !sel.quantize_lm_head) {
        if (sel.verbose) {
            std::cout << "[QuantizationSelector] Skipping " << name << " - LM head disabled" << std::endl;
        }
        return false;
    }
    if (is_moe && !sel.quantize_moe) {
        if (sel.verbose) {
            std::cout << "[QuantizationSelector] Skipping " << name << " - moe disabled" << std::endl;
        }
        return false;
    }
    if (is_router && !sel.quantize_routers) {
        if (sel.verbose) {
            std::cout << "[QuantizationSelector] Skipping " << name << " - router/gate input" << std::endl;
        }
        return false;  // never quantize router/gate input
    }

    // 8. Default: quantize if it's a recognized type
    bool should_quantize = is_attention || is_mlp || is_moe || 
                        is_router || is_embedding ||is_lm_head;

    if (sel.verbose) {
        if (should_quantize) {
            std::cout << "[QuantizationSelector] Quantizing " << name << " - recognized type" << std::endl;
        } else {
            std::cout << "[QuantizationSelector] Skipping " << name << " - unrecognized type" << std::endl;
        }
    }
    
    return should_quantize;
}

bool QuantizationSelector::is_sensitive_layer(const std::string& name) const {
    bool is_embedding = name.find("embed") != std::string::npos;
    bool is_lm_head = name.find("lm_head") != std::string::npos;
        
    return is_lm_head || is_embedding;
}

QuantizationConfig::Mode QuantizationSelector::get_quantization_mode(
    const std::string& name,
    const ov::Shape& shape,
    ov::element::Type dtype) const {
    
    // First check if we should quantize at all
    if (!should_quantize(name, shape, dtype)) {
        return QuantizationConfig::Mode::NONE;
    }
    
    // NNCF-compatible logic:
    // - If backup_mode != mode: sensitive layers (lm_head, embeddings, routers) use backup_mode
    // - If backup_mode == mode: all layers use same mode (like NNCF --all-layers)
    // - If backup_mode == NONE: sensitive layers return NONE (handled by should_quantize)
    bool is_sensitive = is_sensitive_layer(name);
    bool use_backup = is_sensitive && (config_.backup_mode != config_.mode);
    
    if (use_backup) {
        // Use backup mode (typically INT8_ASYM for better accuracy)
        if (config_.selection.verbose) {
            std::cout << "[QuantizationSelector] " << name 
                      << " -> backup mode - sensitive layer" << std::endl;
        }
        return config_.backup_mode;
    }
    
    // Use primary mode (typically INT4 for efficiency)
    if (config_.selection.verbose) {
        std::cout << "[QuantizationSelector] " << name 
                  << " -> primary mode - regular layer" << std::endl;
    }
    return config_.mode;
}

int QuantizationSelector::get_group_size(const std::string& name) const {
    // Determine group_size based on the actual quantization mode for this layer
    // Check if this layer uses backup_mode (sensitive layers like lm_head, embeddings)
    bool is_sensitive = is_sensitive_layer(name);
    bool uses_backup = is_sensitive && (config_.backup_mode != config_.mode);
    
    if (uses_backup) {
        // Backup mode (typically INT8) uses per-channel quantization by default
        // This matches NNCF behavior where sensitive layers use INT8 per-channel
        return -1;  // Per-channel for backup mode
    }
    
    // Primary mode uses user-configured group_size
    return config_.group_size;
}

}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov
