// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "qwen3_moe_config.hpp"
#include <vector>
#include <string>

namespace ov {
namespace genai {

/**
 * @brief Strategy class for determining layer type selection (MLP vs MoE) in Qwen3-MoE model.
 * 
 * This class encapsulates the logic for determining whether each decoder layer should use
 * a standard MLP block or a sparse Mixture-of-Experts (MoE) block. The selection is based
 * on two configuration parameters:
 * 
 * 1. mlp_only_layers: Explicit list of layer indices that should use MLP (overrides sparse_step)
 * 2. decoder_sparse_step: Frequency of MoE layers (e.g., step=2 means every 2nd layer is MoE)
 * 
 * Layer Selection Logic:
 * - If layer_idx is in mlp_only_layers: use MLP (regardless of sparse_step)
 * - Otherwise, if decoder_sparse_step > 0 and num_experts > 0:
 *   - Use MoE if (layer_idx + 1) % decoder_sparse_step == 0
 *   - Use MLP otherwise
 * - If decoder_sparse_step <= 0 or num_experts <= 0: all layers use MLP
 * 
 * Examples:
 * - decoder_sparse_step=2, num_layers=24, mlp_only_layers=[]
 *   -> MoE at layers [1,3,5,7,9,11,13,15,17,19,21,23] (12 MoE, 12 MLP)
 * 
 * - decoder_sparse_step=2, num_layers=24, mlp_only_layers=[0,23]
 *   -> MoE at layers [1,3,5,7,9,11,13,15,17,19,21] (11 MoE, 13 MLP)
 *   -> Layers 0 and 23 use MLP even though they would be MoE by sparse_step rule
 * 
 * - decoder_sparse_step=1, num_layers=24, mlp_only_layers=[]
 *   -> All layers use MoE (24 MoE, 0 MLP)
 * 
 * The class precomputes the layer type schedule during construction for efficient lookup.
 */
class LayerSelectionStrategy {
public:
    /**
     * @brief Construct LayerSelectionStrategy from model configuration.
     * 
     * Precomputes the layer type schedule for all layers based on the configuration.
     * Validates the configuration consistency during construction.
     * 
     * @param config Qwen3-MoE model configuration containing layer selection parameters
     * @throws ov::Exception if configuration is invalid
     */
    explicit LayerSelectionStrategy(const Qwen3MoeConfig& config);

    /**
     * @brief Determine if a specific layer should use MoE block.
     * 
     * @param layer_idx Index of the layer (0-based, range: [0, num_hidden_layers))
     * @return true if the layer uses sparse MoE block, false if it uses standard MLP
     * @throws ov::Exception if layer_idx is out of range
     */
    bool is_moe_layer(int layer_idx) const;

    /**
     * @brief Get the complete layer type schedule as a string vector.
     * 
     * Returns a vector where each element is either "moe" or "mlp" indicating
     * the layer type for the corresponding layer index.
     * 
     * @return Vector of layer type strings, size = num_hidden_layers
     *         Example: ["mlp", "moe", "mlp", "moe", ...] for decoder_sparse_step=2
     */
    std::vector<std::string> get_layer_type_schedule() const;

    /**
     * @brief Validate the layer selection configuration.
     * 
     * Checks:
     * - All mlp_only_layers indices are in valid range [0, num_hidden_layers)
     * - No duplicate indices in mlp_only_layers
     * - decoder_sparse_step results in at least one MoE layer (if > 0)
     * - Warns if configuration results in no MoE layers
     * 
     * @return true if configuration is valid, false otherwise
     */
    bool validate() const;

    /**
     * @brief Get summary statistics of the layer type schedule.
     * 
     * @return Pair of (num_moe_layers, num_mlp_layers)
     */
    std::pair<int, int> get_layer_statistics() const;

private:
    const Qwen3MoeConfig& config_;           ///< Reference to model configuration
    std::vector<bool> layer_types_;          ///< Precomputed layer types (true=MoE, false=MLP)

    /**
     * @brief Precompute layer types for all layers.
     * 
     * Called during construction to build the layer_types_ vector.
     */
    void precompute_layer_types();

    /**
     * @brief Check if a layer index is in mlp_only_layers.
     * 
     * @param layer_idx Layer index to check
     * @return true if layer_idx is in mlp_only_layers
     */
    bool is_in_mlp_only_layers(int layer_idx) const;
};

}  // namespace genai
}  // namespace ov