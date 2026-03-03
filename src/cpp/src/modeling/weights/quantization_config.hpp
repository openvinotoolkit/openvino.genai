// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <optional>
#include <limits>
#include <iostream>

namespace ov {
namespace genai {
namespace modeling {
namespace weights {

/**
 * @brief Custom weight selection configuration for quantization
 * 
 * Provides flexible control over which weights to quantize using multiple strategies:
 * - Pattern-based (wildcard matching)
 * - Layer-based (layer index ranges)
 * - Type-based (attention, mlp, embeddings, etc.)
 * - Explicit lists (include/exclude specific weights)
 * - Size-based (minimum/maximum weight size)
 * 
 * Selection priority (highest to lowest):
 * 1. Size thresholds (applied first)
 * 2. Explicit exclude list
 * 3. Explicit include list
 * 4. Exclude patterns
 * 5. Include patterns
 * 6. Layer range
 * 7. Type-based flags
 */
struct WeightSelectionConfig {
    // Pattern-based selection (supports wildcards: *, ?)
    std::vector<std::string> include_patterns;  // Quantize weights matching these patterns
    std::vector<std::string> exclude_patterns;  // Don't quantize weights matching these patterns
    
    // Layer-based selection (e.g., only quantize layers 5-15)
    std::optional<std::pair<int, int>> layer_range;  // {start_layer, end_layer} inclusive
    
    // Type-based selection (default behavior if no patterns specified)
    bool quantize_attention = true;     // q_proj, k_proj, v_proj, o_proj
    bool quantize_mlp = true;           // gate_proj, up_proj, down_proj
    bool quantize_moe = true;           // MoE expert weights
    bool quantize_embeddings = false;   // embed_tokens
    bool quantize_lm_head = false;      // lm_head
    bool quantize_norm = false;         // Normalization layers (usually not beneficial)
    bool quantize_routers = false;      // MoE router/gate weights (usually not beneficial)
    
    // Explicit weight lists (highest priority)
    std::vector<std::string> include_weights;  // Exact weight names to quantize
    std::vector<std::string> exclude_weights;  // Exact weight names to skip
    
    // Size-based filtering
    size_t min_weight_size = 0;         // Only quantize weights with >= this many elements
    size_t max_weight_size = std::numeric_limits<size_t>::max();  // Only quantize weights with <= this many elements
    
    // Advanced options
    bool only_2d_weights = false;        // Only quantize 2D weight matrices
    bool verbose = false;               // Print quantization decisions
    
    /**
     * @brief Helper: Check if a weight name matches any pattern
     * @param name Weight name to check
     * @param patterns List of wildcard patterns (* = any sequence, ? = single char)
     * @return true if name matches any pattern
     */
    bool matches_pattern(const std::string& name, const std::vector<std::string>& patterns) const;
    
    /**
     * @brief Helper: Extract layer index from weight name (e.g., "layers[5].mlp.weight" -> 5)
     * @param name Weight name
     * @return Layer index if found, std::nullopt otherwise
     */
    std::optional<int> extract_layer_index(const std::string& name) const;
};

/**
 * @brief Quantization configuration for inflight quantization
 * 
 * This configuration is format-agnostic and can be used by any weight finalizer
 * that supports quantization (Safetensors, GGUF, etc.)
 * 
 * NNCF-compatible behavior:
 * - Primary mode (INT4) for most transformer layers (attention, MLP)
 * - Backup mode (INT8_ASYM by default) for sensitive layers (lm_head, embeddings)
 * - Set backup_mode=primary_mode to use same mode for all layers
 * - Set backup_mode=NONE to skip quantizing sensitive layers
 */
struct QuantizationConfig {
    enum class Mode {
        NONE,          // No quantization
        INT4_SYM,      // INT4 symmetric
        INT4_ASYM,     // INT4 asymmetric  
        INT8_SYM,      // INT8 symmetric
        INT8_ASYM      // INT8 asymmetric
    };
    
    // Primary quantization mode (for most layers)
    Mode mode = Mode::NONE;
    int group_size = 128;              // Group size for group-wise quantization
    
    // Backup mode for sensitive layers (NNCF-style)
    // - backup_mode != NONE: lm_head and embeddings use backup_mode
    // - backup_mode == mode: all layers use same mode (like NNCF --all-layers)
    // - backup_mode == NONE: sensitive layers NOT quantized (keep FP16)
    Mode backup_mode = Mode::INT8_ASYM;
    
    // Legacy options (kept for backward compatibility)
    bool quantize_embeddings = false;  // Whether to quantize embeddings (deprecated, use backup_mode)
    bool quantize_lm_head = false;     // Whether to quantize LM head (deprecated, use backup_mode)
    
    // Custom weight selection (new!)
    WeightSelectionConfig selection;
    
    bool enabled() const { return mode != Mode::NONE; }
    
    bool is_primary_4bit() const {
        return mode == Mode::INT4_SYM || mode == Mode::INT4_ASYM;
    }
    
    int bits() const {
        return is_primary_4bit() ? 4 : 8;
    }
    
    int backup_bits() const {
        return (backup_mode == Mode::INT4_SYM || backup_mode == Mode::INT4_ASYM) ? 4 : 8;
    }
    
    /**
     * @brief Convenience: Set selection from legacy options
     * 
     * NNCF-style behavior:
     * - When backup_mode != NONE: always quantize embeddings and lm_head (with backup mode)
     * - When backup_mode == mode: all layers use same mode (like NNCF --all-layers)
     * - When backup_mode == NONE: sensitive layers not quantized
     */
    void apply_legacy_options() {
        // NNCF-style: Always quantize embeddings/lm_head when backup_mode is available
        // They will use backup_mode (INT8_ASYM by default) instead of primary mode
        bool use_backup_quantization = (backup_mode != Mode::NONE);
        
        if (selection.verbose) {
            std::cout << "[QuantizationConfig] Applying legacy options:" << std::endl;
            std::cout << "  quantize_embeddings and lm_head: " << (use_backup_quantization ? "true" : "false") << std::endl;
        }

        selection.quantize_embeddings = use_backup_quantization;
        selection.quantize_lm_head = use_backup_quantization;
    }
};

/**
 * @brief Parse quantization configuration from environment variables
 * 
 * NNCF-compatible quantization with mixed precision support.
 * 
 * Environment variables:
 *   OV_GENAI_INFLIGHT_QUANT_MODE: Primary quantization mode (INT4_SYM, INT4_ASYM, INT8_SYM, INT8_ASYM)
 *   OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE: Group size for INT4 quantization (default: 128)
 *   OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE: Backup mode for sensitive layers (default: INT8_ASYM)
 *                                         Set to primary mode for all-layers quantization
 *                                         Set to NONE to skip quantizing sensitive layers
 *   OV_GENAI_INFLIGHT_QUANT_VERBOSE: If "1" or "true", print detailed quantization decisions
 *   OV_GENAI_INFLIGHT_QUANT_INCLUDE: Comma-separated include patterns
 *   OV_GENAI_INFLIGHT_QUANT_EXCLUDE: Comma-separated exclude patterns
 *   OV_GENAI_INFLIGHT_QUANT_LAYER_RANGE: Layer range (e.g., "10-20")
 *   OV_GENAI_INFLIGHT_QUANT_WEIGHT_NAMES: Comma-separated explicit weight names
 *   OV_GENAI_INFLIGHT_QUANT_MIN_SIZE: Minimum weight size in elements
 *   OV_GENAI_INFLIGHT_QUANT_MAX_SIZE: Maximum weight size in elements
 * 
 * NNCF-compatible default behavior (when mode=INT4_SYM, backup_mode=INT8_ASYM):
 *   - Attention/MLP layers: INT4_SYM with group_size=128
 *   - lm_head: INT8_ASYM per-channel
 *   - embeddings: INT8_ASYM per-channel
 *   - norm layers: not quantized
 * 
 * @return QuantizationConfig parsed from environment variables
 */
QuantizationConfig parse_quantization_config_from_env();

}  // namespace weights
}  // namespace modeling
}  // namespace genai
}  // namespace ov
