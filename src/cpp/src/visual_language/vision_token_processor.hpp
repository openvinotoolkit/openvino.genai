// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <vector>

#include "openvino/runtime/tensor.hpp"
#include "visual_language/cdpruner/cdpruner.hpp"
#include "visual_language/cdpruner/cdpruner_config.hpp"

namespace ov::genai {

// Forward declarations
class EncodedImage;
namespace utils {
class KVCacheState;
}

/**
 * @brief Context for pruning pipeline execution.
 * Contains all necessary data and references for the complete pruning workflow.
 */
struct PruningContext {
    // Input tensors
    const ov::Tensor& input_ids;
    const ov::Tensor& text_embeds;
    const ov::Tensor& merged_visual_embeddings;

    // Image metadata
    const std::vector<ov::genai::EncodedImage>& images;
    const std::vector<std::array<size_t, 3>>& images_grid_thw;
    const std::vector<size_t>& images_sequence;
    const std::vector<size_t>& tokens_per_image;

    // Token IDs
    int64_t image_pad_token_id;
    int64_t vision_start_token_id = -1;  // -1 means not used (optional)
    int64_t vision_end_token_id = -1;    // -1 means not used (optional)

    // Configuration
    size_t spatial_merge_size = 1;  // 1 means no merge (default)
};

/**
 * @brief Vision token processor for optimizing visual features.
 *
 * This class provides a separate abstraction for post-processing visual tokens,
 * including pruning, compression, and other optimization techniques.
 * Currently implements visual token pruning using CDPruner algorithm.
 */
class VisionTokenProcessor {
public:
    /**
     * @brief Construct a new Vision Token Processor
     * @param device Device to use for processing operations
     */
    explicit VisionTokenProcessor(const std::string& device);

    /**
     * @brief Process (prune) visual features based on text features
     * @param visual_features Vector of visual feature tensors to process
     * @param text_features Text features for relevance calculation
     * @return Processed (pruned) visual features tensor
     */
    ov::Tensor process(const std::vector<ov::Tensor>& visual_features, const ov::Tensor& text_features);

    /**
     * @brief Check if processor is available and ready
     * @return true if processor is initialized and available
     */
    bool is_available() const {
        return m_pruner != nullptr;
    }

    /**
     * @brief Set processor configuration
     * @param config New configuration to apply
     */
    void set_config(const cdpruner::Config& config);

    /**
     * @brief Get current processor configuration
     * @return Current configuration
     */
    cdpruner::Config get_config() const;

    /**
     * @brief Get statistics from last processing operation
     * @return Pruning statistics if available
     */
    std::optional<cdpruner::PruningStatistics> get_last_statistics() const;

    /**
     * @brief Get indices of selected tokens from last processing
     * @return Vector of selected token indices for each frame/image
     */
    std::vector<std::vector<size_t>> get_last_selected_tokens() const;

    /**
     * @brief Result structure for CDPruner visual token pruning pipeline.
     * Contains all necessary information about the pruning operation and its results.
     */
    struct PruningResult {
        bool is_pruned = false;                                ///< Whether pruning was actually applied
        size_t original_visual_tokens = 0;                     ///< Original number of visual tokens before pruning
        size_t pruned_visual_tokens = 0;                       ///< Number of visual tokens after pruning
        ov::Tensor pruned_embeddings;                          ///< Pruned visual embeddings tensor
        ov::Tensor pruned_input_ids;                           ///< Input IDs with pruned visual tokens removed
        ov::Tensor pruned_text_embeds;                         ///< Text embeddings with pruned visual positions removed
        std::vector<std::vector<bool>> keep_flags_per_region;  ///< Keep flags for each visual region
        std::optional<int64_t> updated_rope_delta;  ///< Updated rope_delta value (optional, only for RoPE models)
    };

    /**
     * @brief Extract text features for CDPruner relevance calculation.
     */
    ov::Tensor extract_text_features(const ov::Tensor& text_embeds,
                                     const ov::Tensor& input_ids,
                                     int64_t image_pad_token_id,
                                     int64_t vision_start_token_id,
                                     int64_t vision_end_token_id) const;

    /**
     * @brief Convert visual features to CDPruner batch format.
     * @param vision_embeds Input embeddings [total_tokens, hidden_dim]
     * @param chunk_count Number of chunks (1 for single batch, N for per-image)
     * @param tokens_per_image Token count for each image (when chunk_count > 1)
     * @return Vector of tensors, each [1, num_tokens, hidden_dim]
     */
    std::vector<ov::Tensor> convert_visual_features(const ov::Tensor& vision_embeds,
                                                    size_t chunk_count,
                                                    const std::vector<size_t>& tokens_per_image) const;

    /**
     * @brief Adjust position IDs after visual token pruning.
     * @param position_ids_inout The position IDs to adjust (modified in-place)
     * @param input_ids The input token IDs for sequence traversal
     * @param images_grid_thw Grid dimensions for each image
     * @param images_sequence Image sequence ordering
     * @param image_pad_token_id Token ID for image padding
     * @param vision_start_token_id Token ID for vision start marker
     * @param spatial_merge_size Spatial merge size for coordinate conversion
     * @param keep_flags_per_region_out Output: keep flags for each vision region
     */
    void adjust_position_ids(ov::Tensor& position_ids_inout,
                             const ov::Tensor& input_ids,
                             const std::vector<std::array<size_t, 3>>& images_grid_thw,
                             const std::vector<size_t>& images_sequence,
                             int64_t image_pad_token_id,
                             int64_t vision_start_token_id,
                             size_t spatial_merge_size,
                             std::vector<std::vector<bool>>& keep_flags_per_region_out) const;

    /**
     * @brief Update 3D position IDs for Qwen2VL-style models (3D RoPE).
     */
    ov::Tensor update_position_ids_3d(const ov::Tensor& original_position_ids,
                                      const ov::Tensor& input_ids,
                                      int64_t vision_start_token_id,
                                      int64_t image_pad_token_id,
                                      const std::vector<std::array<size_t, 3>>& reordered_images_grid_thw,
                                      const std::vector<std::vector<size_t>>& kept_indices_per_image,
                                      size_t spatial_merge_size,
                                      std::vector<std::vector<bool>>& keep_flags_out) const;

    /**
     * @brief Update 1D position IDs for LLaVA-style models.
     */
    ov::Tensor update_position_ids_1d(const ov::Tensor& original_position_ids,
                                      const ov::Tensor& input_ids,
                                      int64_t vision_start_token_id,
                                      int64_t image_pad_token_id,
                                      const std::vector<std::array<size_t, 3>>& reordered_images_grid_thw,
                                      const std::vector<std::vector<size_t>>& kept_indices_per_image,
                                      std::vector<std::vector<bool>>& keep_flags_out) const;

    /**
     * @brief Generate pruned input_ids based on keep_flags.
     */
    ov::Tensor generate_pruned_input_ids(const ov::Tensor& input_ids,
                                         const std::vector<std::vector<bool>>& keep_flags_per_region,
                                         int64_t image_pad_token_id,
                                         int64_t vision_start_token_id,
                                         int64_t vision_end_token_id) const;

    /**
     * @brief Generate pruned text embeddings by removing filtered image_pad positions.
     */
    ov::Tensor generate_pruned_text_embeds(const ov::Tensor& input_ids,
                                           const ov::Tensor& text_embeds,
                                           int64_t image_pad_token_id,
                                           int64_t vision_start_token_id,
                                           int64_t vision_end_token_id,
                                           const std::vector<std::vector<bool>>& keep_flags_per_region) const;

    /**
     * @brief Execute the complete pruning pipeline.
     *
     * This method orchestrates the entire pruning workflow including:
     * - Text feature extraction
     * - Visual feature conversion
     * - Token pruning
     * - Position IDs adjustment
     * - Input IDs and embeddings regeneration
     * - KV cache update
     *
     * @param context PruningContext containing input data and optional rope_delta pointer
     * @param position_ids Position IDs tensor (modified in-place)
     * @param kv_cache_state KV cache state (modified)
     * @param prev_hist_length Previous history length for KV cache
     * @return PruningResult with pruned tensors and metadata
     */
    PruningResult execute_full_pipeline(const PruningContext& context,
                                        ov::Tensor& position_ids,
                                        utils::KVCacheState& kv_cache_state,
                                        size_t prev_hist_length);

private:
    /// @brief CDPruner instance for token pruning (lazy initialized)
    std::unique_ptr<cdpruner::CDPruner> m_pruner;
    /// @brief Configuration storage (used before pruner is created; pruner becomes source of truth after creation)
    /// Device is stored in m_config.device
    cdpruner::Config m_config;
};

}  // namespace ov::genai
