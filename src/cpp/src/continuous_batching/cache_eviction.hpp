// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include <vector>
#include <cstdlib>
#include <cmath>

#include "openvino/openvino.hpp"
#include "continuous_batching/attention_output.hpp"
#include "openvino/genai/cache_eviction.hpp"
#include "continuous_batching/kvcrush.hpp"

namespace ov::genai {

/**
 * @brief Keeps track of the accumulated token scores across model inferences and their lifetime.
 */
class EvictionScoreManager {
public:
    EvictionScoreManager() = default;
    EvictionScoreManager(const EvictionScoreManager& rhs) = default;
    EvictionScoreManager& operator=(const EvictionScoreManager& rhs) = default;

    /**
     * Constructs an EvictionScoreManager.
     * @param block_size Block size of the KV cache to evict from.
     * @param num_decoder_layers Number of independent KV caches (each corresponding to a single attention layer) in the underlying LLM.
     * @param max_pool_window_size Window size for the max pooling step applied to the newly registered scores before aggregation.
     * @param aggregation_mode Aggregation mode for the scores across register calls.
     * @param ignore_first_n_blocks Number of blocks from the beginning of the per-token score vector, the scores for which will
     * be disregarded and never aggregated.
     * @param snapkv_window_size Window size for the SnapKV algorithm in effect. If non-zero, then by the start of the generation phase
     * for the tracked sequence (when the total number of `num_snapkv_scores` passed to each `register_new_token_scores` call reaches
     * the `snapkv_window_size`) the internal occurence counters will be:
     * `| S | S | ... | S | S - 1 | S - 2 | ... | 2 | 1 |`,
     * where `S` is equal to `snapkv_window_size`. In contrast, if this is set to 0, then the initial counter state would be
     * `| L | L - 1 | ... | 2 | 1 |`,
     * where L is the prompt size of the sequence in tokens.
     */
    explicit EvictionScoreManager(size_t block_size, size_t num_decoder_layers, size_t max_pool_window_size, AggregationMode aggregation_mode, size_t ignore_first_n_blocks = 0, size_t snapkv_window_size = 0) : m_block_size(block_size), m_num_decoder_layers(num_decoder_layers), m_scores(num_decoder_layers), m_cache_counter(num_decoder_layers), m_max_pool_window_size(max_pool_window_size), m_aggregation_mode(aggregation_mode), m_ignore_first_n_blocks(ignore_first_n_blocks), m_snapkv_window_size(snapkv_window_size), m_num_registered_snapkv_aggregated_scores(0) {}

    /**
     * Registers new token scores and aggregates them internally as necessary. The token scores provided may be corresponding not to all
     * tokens in the current sequence length, in which case the set of logical block indices must be provided for which the score entries
     * are missing.
     *
     * @param attention_scores_for_all_decoder_layers A vector of ov::Tensor, each ov::Tensor corresponding to the per-token attention
     * scores in a corresponding decoder layer.
     * @param skipped_logical_block_ids Logical block indices which had been skipped during inference call that produced the new scores, and
     * which are missing from the new scores.
     * @param num_snapkv_scores Number of latest token scores that were aggregated together when computing the registered score. If SnapKV is not used, this should be set to 0.
     */
    void register_new_token_scores(const AttentionScoresForEachDecoderLayer& attention_scores_for_all_decoder_layers, const std::set<size_t>& skipped_logical_block_ids, size_t num_snapkv_scores = 0);

    /**
     * Removes the scores from tracking for given block indices and given decoder layer.
     *
     * @param evicted_block_indices A vector of logical block indices, the scores for which should be removed from tracking.
     * @param decoder_layer_idx The index of the decoder layer for which the block scores must be removed.
     */
    void remove_scores(const std::vector<std::size_t>& evicted_block_indices, size_t decoder_layer_idx);

    /**
     * Adds two vectors of different length, treating the shorter one as values from which certain block-sized chunks had been
     * skipped on purpose and which should not impact the final sum.
     *
     * @param dst The destination vector.
     * @param src The source vector. Must be shorter by N * B values than dst, where N is the size of the skipped logical block ID set,
     * and B is the block size.
     * @param skipped_logical_block_ids The set of logical block IDs that had been "skipped" from the src values.
     */
    void add_with_skips(std::vector<double>& dst, const std::vector<double>& src, const std::set<size_t>& skipped_logical_block_ids) const;

    /**
     * @param layer_idx The decoder layer index.
     * @return Current length of the tracked scores in tokens for this decoder layer.
     */
    size_t get_current_scores_length_in_tokens(size_t layer_idx) const;

    /**
     * @return Current scores for all decoder layers (0-th dimension) and tokens (1-st dimension).
     */
    const std::vector<std::vector<double>>& get_scores() const;

    /**
     * @return Current token occurence counters for all decoder layers (0-th dimension) and tokens (1-st dimension).
     */
    const std::vector<std::vector<size_t>>& get_counters() const;

private:
    std::size_t m_block_size;
    std::size_t m_num_decoder_layers;
    std::vector<std::vector<double>> m_scores;
    std::vector<std::vector<size_t>> m_cache_counter;
    std::size_t m_max_pool_window_size;
    AggregationMode m_aggregation_mode;
    std::size_t m_ignore_first_n_blocks;
    std::size_t m_snapkv_window_size;
    std::size_t m_num_registered_snapkv_aggregated_scores;
};

class SnapKVScoreAggregationCalculator {
public:
    SnapKVScoreAggregationCalculator() = default;
    SnapKVScoreAggregationCalculator(const SnapKVScoreAggregationCalculator& rhs) = default;
    SnapKVScoreAggregationCalculator& operator=(const SnapKVScoreAggregationCalculator& rhs) = default;
    SnapKVScoreAggregationCalculator(size_t snapkv_window_size) : m_snapkv_window_size(snapkv_window_size) {}

    size_t get_num_token_scores_to_aggregate(size_t prompt_len, size_t num_scheduled_tokens, size_t num_processed_tokens);

private:
    size_t m_snapkv_window_size;

};

/**
 * @brief Determines blocks to be evicted from the KV cache of a sequence based on the importance score calculated from the
 * attention scores of each token at each attention layer in the LLM.
 *
 * The KV cache is conceptually divided into three areas as shown below:
 *
 * ```
 * --> *logical KV cache space in blocks*
 * | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
 * |<- start area->|<-   evictable area    ->|<- recent area ->|
 * ```
 *
 * The sizes of each areas are configurable. Once the sequence KV cache utilization is such that all three areas
 * are filled, the algorithm determines the blocks from the *evictable area* that should be freed from this sequence
 * based on the importance scores accumulated after each previous generation step in the pipeline. The least important
 * tokens according to this score are to be evicted. Only the tokens from the *evictable area* are evicted - the tokens
 * in the *start* and *recent* areas are never evicted, but throughout the eviction process the *recent* blocks naturally
 * move into the *evictable* area.
 *
 * Eviction only starts when at least one block *past* the *recent area* is completely filled, and the corresponding number
 * of blocks is selected to be evicted, so that the remaining blocks completely fit into the arena defined by the *start*,
 * *evictable* and *recent* areas. This effectively caps the cache usage for the sequence by the size of the arena (plus,
 * in general, one partially filled block past the recent area).
 *
 * Sizes of *start*, *evictable* and *recent* areas are configurable, but the *evictable* area size specifies the
 * _minimal_ size of the evictable area. When tokens overflow the eviction arena, the actual evictable area is
 * determined as the tokens between the fixed-size *start area* and the fixed-size *end area*, so at a given eviction step
 * there are in general more tokens considered for eviction than the specified *evictable* size.
 *
 */
class CacheEvictionAlgorithm {
public:
    /**
     * @brief A pair of indices specifying the logical block interval where the blocks may be evicted at this point in time.
     */
    class CacheEvictionRange : public std::pair<std::size_t, std::size_t> {
    public:
        CacheEvictionRange(std::size_t begin, std::size_t end) : std::pair<std::size_t, std::size_t>(begin, end) {}
        static const CacheEvictionRange& invalid() {
            static CacheEvictionRange inv(0, 0);
            return inv;
        }
    };
    CacheEvictionAlgorithm() = default;  // needed only to satisfy DefaultConstructible so that algo objects may be used as values in std::map

    /**
     * Constructs a CacheEvictionAlgorithm.
     * @param eviction_config The configuration struct for this algorithm.
     * @param block_size Block size of the KV cache to evict from.
     * @param num_decoder_layers Number of independent KV caches (each corresponding to a single attention layer) in the underlying LLM.
     */
    explicit CacheEvictionAlgorithm(const CacheEvictionConfig& eviction_config, size_t block_size, size_t num_decoder_layers, size_t max_pool_window_size);

    /**
     * @return Maximum cache size (in tokens) after each eviction step. Could be used as an estimate of the maximum per-sequence cache usage.
     */
    std::size_t get_max_cache_size_after_eviction() const;

    /**
     * @return Current logical range of evictable block indices.
     */
    CacheEvictionRange get_evictable_block_range() const;

    /**
     * Registers attention scores (for each layer) of each token in this sequence that is currently still represented
     * (i.e. not evicted) in the corresponding KV cache. Must be called after each generation step to properly keep track of
     * the tokens' lifetime in the KV cache and of the accumulated importance score of each token.
     * @param attention_scores_for_all_decoder_layers A vector with a size equal to the configured num_decoder_layers, where each entry is a
     * vector of per-token attention scores calculated within this layer.
     * @param skipped_logical_block_ids The set of logical indices that have been skipped from the scores as part of the sparse attention prefill process
     * @param num_snapkv_scores The number of SnapKV-aggregated scores in this score chunk. Set to 0 if SnapKV is not used
     * (i.e. eviction_config.snapkv_window_size == 0)
     */
    void register_new_token_scores(const AttentionScoresForEachDecoderLayer& attention_scores_for_all_decoder_layers, const std::set<size_t>& skipped_logical_block_ids, size_t num_snapkv_scores = 0);

    void register_new_token_scores(const AttentionScoresForEachDecoderLayer& attention_scores_across_decoder_layers_for_current_sequence, size_t num_snapkv_scores = 0);
    /**
     * Returns the per-layer sets of logical block indices that should be evicted according to the internally computed importance scores
     * and removes the corresponding blocks from the internal algorithm tracking.
     *
     * @return A vector with size equal to the configured num_decoder_layers, where each entry is a set of logical indices that are to be
     * evicted by the external cache-controlling mechanism.
     */
    std::vector<std::set<std::size_t>> evict_logical_blocks();


private:
    std::size_t get_num_blocks(std::size_t num_tokens) const;
    std::size_t get_num_blocks_to_evict(size_t decoder_layer_idx) const;
    std::size_t get_num_evictable_blocks(size_t decoder_layer_idx) const;

    CacheEvictionRange get_evictable_block_range(size_t layer_idx) const;

    std::vector<double> get_scores_for_all_evictable_blocks(size_t decoder_layer_idx) const;

    std::vector<std::size_t> get_indices_of_blocks_to_evict(const std::vector<double>& scores_for_each_evictable_block, size_t num_blocks_to_evict) const;

    void remove_scores_of_evicted_blocks(const std::vector<std::size_t>& evicted_block_indices, size_t decoder_layer_idx);

    CacheEvictionConfig m_eviction_config;
    KVCrushAlgorithm m_kvcrush_algo;
    std::size_t m_block_size;
    std::size_t m_num_evicted_tokens = 0;
    std::size_t m_num_decoder_layers;
    EvictionScoreManager m_score_manager;
};



/**
 * @brief Computes, based on the logical indices of the blocks to be evicted, the rotation coefficients for the
 * remaining cache blocks.
 *
 * The rotation assumes that the executed model applies rotary positional embedding (RoPE) during the execution of
 * the attention operation. Each cache block therefore has the RoPE values already "baked in", with positions equivalent
 * to the point in time when the cache block values were originally computed in one of the previous attention
 * operations. When blocks are evicted, the logical index space of the remaining blocks is in general no longer
 * contiguous with respect to the effective positions of tokens in the blocks. Cache rotation allows to remedy this by
 * effectively adjusting the RoPE positions of certain blocks in the cache after eviction, by additionally "rotating"
 * them (in the same sense as in RoPE) by such angles that the cache blocks in the logical index space are again
 * contiguous in terms of the RoPE positions. This is supposed to make the eviction process less impactful on the
 * accuracy of the generation.
 *
 * Currently only the basic RoPE method is supported (as applied in the Llama original models). Each model in general
 * may have its own RoPE method (e.g. non-linear/NTK frequency scaling), and ideally the cache rotation calculator
 * should be adjusted based on the specifics of the RoPE defined by the LLM.
 */
class CacheRotationCalculator {
public:
    /**
     * Constructs a CacheRotationCalculator.
     * @param block_size Block size of the KV cache to evict from.
     * @param max_context_length Maximum length possible for a sequence in the current pipeline.
     * @param kv_head_size The size (in elements) of the embedding dimension in the attention operation.
     * @param rope_theta The base RoPE angle used in the original LLM.
     */
    CacheRotationCalculator(size_t block_size,
                            size_t max_context_length,
                            size_t kv_head_size,
                            double rope_theta = 10000.0f);

    using RotationCoefficientsPerToken = std::vector<std::vector<float>>;  // dimensions: [BLOCK_SIZE, head_size / 2]

    /**
     * Basic output structure for the calculator.
     */
    struct BlockRotationData {
        bool operator==(const BlockRotationData& rhs) const {
            return (logical_block_idx == rhs.logical_block_idx) && (sines == rhs.sines) && (cosines == rhs.cosines);
        }
        size_t logical_block_idx;             /** Logical index of the block AFTER eviction to which the rotation
                                                 should be applied */
        size_t rotation_delta;                /** Delta, in token positions, that should be applied to block contents
                                                via rotation **/

        // Fields below are currently only used for testing purposes
        RotationCoefficientsPerToken sines;   /** The sine coefficients to be applied to this block's contents for
                                                 rotation, in order of the block's elements */
        RotationCoefficientsPerToken cosines; /** The cosine coefficients to be applied to this block's contents for
                                                 rotation, in order of the block's elements */
    };

    /**
     * Computes the rotation coefficients for the given state of the logical block space when eviction is about to take
     * place.
     * @param evicted_block_logical_indices The logical block indices that the prior cache eviction algorithm step
     * determined to be necessary to evict.
     * @param num_logical_blocks_before_eviction Number of logical blocks that the evicted-from sequence occupied before
     * the eviction step.
     * @param deltas_only If true, the sines and cosines fields in each returned BlockRotationData will be left empty.
     * @return A vector of per-block rotation data, including the indices of blocks after eviction that should be
     * rotated, and the pre-computed trigonometric coefficients necessary for rotation.
     */
    std::vector<BlockRotationData> get_rotation_data(const std::set<size_t>& evicted_block_logical_indices,
                                                             size_t num_logical_blocks_before_eviction,
                                                             bool deltas_only = true);

    /**
     * @return The size of the embedding dimension that this CacheRotationCalculator was initialized with.
     */
    size_t get_head_size() const {
        return m_head_size;
    }

    const std::vector<std::vector<float>>& get_sin_lut() const;
    const std::vector<std::vector<float>>& get_cos_lut() const;

private:
    size_t m_block_size;
    size_t m_head_size;
    std::vector<std::vector<float>> m_rope_sin_lut;  // dimensions: [ max_context_length, head_size / 2]
    std::vector<std::vector<float>> m_rope_cos_lut;  // dimensions: [ max_context_length, head_size / 2]
};

} // namespace ov::genai
