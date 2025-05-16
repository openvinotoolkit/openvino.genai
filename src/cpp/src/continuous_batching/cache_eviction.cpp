// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache_eviction.hpp"

namespace ov::genai {
    CacheEvictionAlgorithm::CacheEvictionAlgorithm(const CacheEvictionConfig &eviction_config, size_t block_size,
                                                   size_t num_decoder_layers) :
            m_eviction_config(eviction_config), m_block_size(block_size), m_num_decoder_layers(num_decoder_layers),
            m_cache_counter(num_decoder_layers), m_scores(num_decoder_layers) {
            OPENVINO_ASSERT(!(m_eviction_config.get_start_size() % m_block_size),
                            "CacheEvictionConfig.start_size in tokens must be a multiple of block size ", m_block_size);
            OPENVINO_ASSERT(!(m_eviction_config.get_recent_size() % m_block_size),
                            "CacheEvictionConfig.recent_size in tokens must be a multiple of block size ", m_block_size);
            OPENVINO_ASSERT(!(m_eviction_config.get_max_cache_size() % m_block_size),
                            "CacheEvictionConfig.max_cache_size in tokens must be a multiple of block size ", m_block_size);
            OPENVINO_ASSERT(m_num_decoder_layers, "num_decoder_layers must be non-zero");
    }

    std::size_t CacheEvictionAlgorithm::get_max_cache_size_after_eviction() const {
        // The cache layout after eviction should have blocks in all 3 areas (start, evictable and recent) fully filled,
        // and since we evict full blocks only from the middle, evictable part of the cache, then at least one block
        // past the "recent" area should be completely filled with fresh tokens before we can evict at least 1 block
        // from the evictable area
        return m_eviction_config.get_max_cache_size() + m_block_size - 1;
    }

    std::vector<std::size_t> CacheEvictionAlgorithm::get_indices_of_blocks_to_retain_using_kvcrush(size_t decoder_layer_idx, size_t k, std::vector<std::size_t>& evicted_block_indices)
    {
        const auto& keep_clus_eligible = evicted_block_indices;
        k = std::min(k, m_scores[decoder_layer_idx].size()); //k is number of tokens
        std::vector<size_t> indices(k);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(
            indices.begin(),
            indices.begin() + k/2,
            indices.end(),
            [&](size_t i, size_t j) {
                return m_scores[decoder_layer_idx][i] > m_scores[decoder_layer_idx][j];
            }
        );
        std::vector<int> indicators(k, 0);
        for (size_t i = 0; i < k/2; ++i) {
            indicators[indices[i]] = 1;
        }

        // Step 2: Create a random binary vector of size m_block_size as anchor point
        std::vector<int> anchor_point(m_block_size);
        // Initialize anchor_point based on anchor using switch-case
        switch (m_eviction_config.anchor_point) {
            case AnchorPoints::RANDOM:
            std::generate(anchor_point.begin(), anchor_point.end(), []() { return rand() % 2; });
            break;
            case AnchorPoints::ZEROS:
            std::fill(anchor_point.begin(), anchor_point.end(), 0);
            break;
            case AnchorPoints::ONES:
            std::fill(anchor_point.begin(), anchor_point.end(), 1);
            break;
            case AnchorPoints::MEAN: {
            // Mean here is 1 if the average of the indicators is greater than 0.5
            // and 0 otherwise
            double mean = std::accumulate(indicators.begin(), indicators.end(), 0.0) / k;
            for (size_t i = 0; i < m_block_size; ++i) {
                anchor_point[i] = (mean > 0.5) ? 1 : 0;
            }
            break;
            }
            case AnchorPoints::ALTERNATE:
            for (size_t i = 0; i < m_block_size; ++i) {
                anchor_point[i] = i % 2;
            }
            break;
            default:
            throw std::invalid_argument("Invalid anchor point type");
        }
        
        // Step 3: Calculate Hamming distances between anchor point and each block
        size_t num_blocks = k / m_block_size;
        std::vector<std::pair<size_t, size_t>> block_distances; // pair<hamming_distance, block_idx>
        block_distances.reserve(num_blocks);

        for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            size_t hamming_distance = 0;
            for (size_t j = 0; j < m_block_size; ++j) {
                size_t token_idx = block_idx * m_block_size + j;
                if (token_idx < k) {
                    // Use the indicators vector to determine the bit value of this position
                    int bit_value = indicators[token_idx];
                    if (bit_value != anchor_point[j]) {
                        hamming_distance++;
                    }
                }
            }
            block_distances.emplace_back(hamming_distance, block_idx);
        }

        // Step 4: Filter block indices that are in keep_clus_eligible
        std::vector<size_t> filtered_block_indices;
        filtered_block_indices.reserve(block_distances.size());

        for (const auto& entry : block_distances) {
            size_t block_idx = entry.second;
            // Check if block_idx is in keep_clus_eligible
            if (std::find(keep_clus_eligible.begin(), keep_clus_eligible.end(), block_idx) != keep_clus_eligible.end()) {
                filtered_block_indices.push_back(block_idx);
            }
        }
        // Sort filtered_block_indices based on Hamming distance
        std::sort(filtered_block_indices.begin(), filtered_block_indices.end(),
                    [&](size_t a, size_t b) {
                        return block_distances[a].first < block_distances[b].first;
                    });
        // select num_clusters blocks from filtered_block_indices, uniformly spaced
        size_t num_clusters = m_eviction_config.get_kvcrush_budget() / m_block_size;
        size_t step = filtered_block_indices.size() / num_clusters;
        std::vector<std::size_t> kvcrush_retained_block_indices;
        kvcrush_retained_block_indices.reserve(num_clusters);
        for (size_t i = 0; i < num_clusters; ++i) {
            size_t idx = i * step;
            if (idx < filtered_block_indices.size()) {
                kvcrush_retained_block_indices.push_back(filtered_block_indices[idx]);
            }
        }
        return kvcrush_retained_block_indices;
    }

    std::vector<std::set<std::size_t>> CacheEvictionAlgorithm::evict_logical_blocks() {
        // Returns the indices of logical KV cache blocks to evict (the rest is to be discarded) for each decoder layer in order.
        // The kept indices are determined using `attention_scores`, which is expected to be the
        // attention head scores that are already reduced by the batch and head dimensions, i.e. the shape of
        // `attention_scores` must be [num_new_tokens, current_seq_len], where `num_new_tokens` is the dimension
        // corresponding to the number of freshly generated tokens since the last cache eviction has taken place,
        // and the `current_seq_len` is the dimension corresponding to the current sequence length at this stage
        // in the generation process, i.e. the dimension over which the attention scores over individual previous
        // tokens was being computed.

        std::vector<std::set<size_t>> retval(m_num_decoder_layers);

        for (size_t decoder_layer_idx = 0; decoder_layer_idx < m_scores.size(); decoder_layer_idx++) {
            const auto &accumulated_scores_for_current_decoder_layer = m_scores[decoder_layer_idx];
            auto scores_length = accumulated_scores_for_current_decoder_layer.size();
            if (scores_length + m_eviction_config.get_start_size() <= get_max_cache_size_after_eviction()) {
                // KV cache is not yet filled, keep all currently occupied blocks
                continue;
            }

            // Only the blocks in the "intermediate" part of the logical KV cache will be considered for eviction
            auto scores_for_all_evictable_blocks = get_scores_for_all_evictable_blocks(decoder_layer_idx);
            size_t num_blocks_to_evict = get_num_blocks_to_evict(decoder_layer_idx);
            auto evicted_block_indices = get_indices_of_blocks_to_evict(scores_for_all_evictable_blocks, num_blocks_to_evict);

            //KVCrush: start 
            bool is_kvcrush_enabled = (m_eviction_config.get_kvcrush_budget() > 0) && (std::ceil(static_cast<double>(evicted_block_indices.size())) >= std::floor(static_cast<double>(m_eviction_config.get_kvcrush_budget()) / m_block_size));
            if(is_kvcrush_enabled)
            {
                size_t k = scores_for_all_evictable_blocks.size() * m_block_size;
                auto kvcrush_retained_block_indices = get_indices_of_blocks_to_retain_using_kvcrush(decoder_layer_idx, k, evicted_block_indices);

                // Remove the indcies in kvcrush_retained_block_indices from evicted_block_indices
                if (!kvcrush_retained_block_indices.empty()) {
                    // Convert both vectors to sets for efficient operations
                    std::unordered_set<std::size_t> retained_set(
                        kvcrush_retained_block_indices.begin(), 
                        kvcrush_retained_block_indices.end()
                    );
                    
                    // Create a new vector containing only elements not in retained_set
                    std::vector<std::size_t> filtered_evicted_indices;
                    filtered_evicted_indices.reserve(evicted_block_indices.size());
                    
                    for (const auto& idx : evicted_block_indices) {
                        if (retained_set.find(idx) == retained_set.end()) {
                            filtered_evicted_indices.push_back(idx);
                        }
                    }
                    // Replace the original vector with the filtered one
                    evicted_block_indices = std::move(filtered_evicted_indices);
                }
                
            }
            //KVCrush: end
            m_num_evicted_tokens += evicted_block_indices.size() * m_block_size;

            // No longer need to track the overall "heavy-hitter" attention scores for freshly evicted blocks
            remove_scores_of_evicted_blocks(evicted_block_indices, decoder_layer_idx);

            // Adjust indices to account for start area
            for (auto &idx: evicted_block_indices) idx += get_num_blocks(m_eviction_config.get_start_size());
            // auto remaining_block_indices = get_remaining_block_indices(evicted_block_indices);
            for (auto &idx: evicted_block_indices) retval[decoder_layer_idx].insert(idx);
        }
        return retval;
    }

    CacheEvictionAlgorithm::CacheEvictionRange CacheEvictionAlgorithm::get_evictable_block_range() const {
        return get_evictable_block_range(0);
    }

    CacheEvictionAlgorithm::CacheEvictionRange CacheEvictionAlgorithm::get_evictable_block_range(size_t layer_idx) const {
        std::size_t current_sequence_length = m_eviction_config.get_start_size() + m_scores[layer_idx].size();
        if (current_sequence_length <= get_max_cache_size_after_eviction()) {
            return CacheEvictionRange::invalid(); // purposely invalid range since no eviction can take place yet
        }
        std::size_t start = m_eviction_config.get_start_size() / m_block_size;
        std::size_t end = current_sequence_length / m_block_size - (m_eviction_config.get_recent_size() / m_block_size);
        return {start, end};
    }

    void CacheEvictionAlgorithm::register_new_token_scores(
            const AttentionScoresForEachDecoderLayer &attention_scores_for_all_decoder_layers) {
        for (size_t decoder_layer_idx = 0; decoder_layer_idx < m_cache_counter.size(); decoder_layer_idx++) {

            const auto &attention_scores = attention_scores_for_all_decoder_layers[decoder_layer_idx];
            // "Start" tokens are never evicted, won't track scores for these
            // "Recent" tokens are also not evicted just yet, but need to accumulate their scores since they may
            // ultimately move into the "intermediate" eviction region of cache
            // Taking the [1, start_size:seq_len] span of the attention scores:
            auto attn_shape = attention_scores.get_shape();
            size_t kv_cache_size_in_tokens = attn_shape[0];
            if (kv_cache_size_in_tokens <= m_eviction_config.get_start_size() + 1) {
                return;
            }

            auto hh_score = ov::Tensor(
                    attention_scores,
                    ov::Coordinate{m_eviction_config.get_start_size()},
                    ov::Coordinate{kv_cache_size_in_tokens}
            );

            auto &accumulated_scores_for_current_decoder_layer = m_scores[decoder_layer_idx];

            if (accumulated_scores_for_current_decoder_layer.empty()) {
                accumulated_scores_for_current_decoder_layer = std::vector<double>(hh_score.get_size());
                for (size_t idx = 0; idx < accumulated_scores_for_current_decoder_layer.size(); idx++) {
                    accumulated_scores_for_current_decoder_layer[idx] = hh_score.data<float>()[idx];
                }
                if (m_eviction_config.aggregation_mode == AggregationMode::NORM_SUM) {
                    // New sequence to track - will simulate that the tokens comprising the sequence were added one-by-one
                    // from the standpoint of the occurrence tracker
                    std::size_t new_scores_size = hh_score.get_size();
                    std::vector<std::size_t> counter(new_scores_size);
                    std::generate(counter.begin(), counter.begin() + new_scores_size,
                                  [&new_scores_size] { return new_scores_size--; });
                    m_cache_counter[decoder_layer_idx] = counter;
                }
            } else {
                size_t old_size_in_tokens = accumulated_scores_for_current_decoder_layer.size();
                size_t num_new_tokens = hh_score.get_size() - accumulated_scores_for_current_decoder_layer.size();
                if (m_eviction_config.aggregation_mode == AggregationMode::NORM_SUM) {
                    // Increment occurrence counts of all currently tracked cache blocks
                    auto &counter_for_current_decoder_layer = m_cache_counter[decoder_layer_idx];
                    for (auto it = counter_for_current_decoder_layer.begin();
                         it != counter_for_current_decoder_layer.end(); it++) {
                        *it += num_new_tokens;
                    }
                    // Add occurrence counts for new tokens like above
                    counter_for_current_decoder_layer.resize(hh_score.get_size());
                    for (size_t i = 0; i < num_new_tokens; i++) {
                        auto idx = old_size_in_tokens + i;
                        counter_for_current_decoder_layer[idx] = num_new_tokens - i;
                    }
                }
                accumulated_scores_for_current_decoder_layer.resize(hh_score.get_size());
                auto hh_score_data = hh_score.data<float>();
                for (size_t i = 0; i < hh_score.get_size(); ++i) {
                    accumulated_scores_for_current_decoder_layer[i] += hh_score_data[i];
                }
            }
        }
    }

    std::size_t CacheEvictionAlgorithm::get_num_blocks(std::size_t num_tokens) const {
        return static_cast<std::size_t>(std::ceil(((double) num_tokens) / m_block_size));
    }

    std::size_t CacheEvictionAlgorithm::get_num_evictable_blocks(size_t layer_idx) const {
        auto range = get_evictable_block_range(layer_idx);
        return range.second - range.first;
    }

    std::size_t CacheEvictionAlgorithm::get_num_blocks_to_evict(size_t layer_idx) const {
        auto num_evictable_blocks = get_num_evictable_blocks(layer_idx);
        std::size_t num_evictable_blocks_to_keep_after_eviction = get_num_blocks(m_eviction_config.get_evictable_size());
        if (num_evictable_blocks < num_evictable_blocks_to_keep_after_eviction) {
            return 0;
        }
        
        return num_evictable_blocks - num_evictable_blocks_to_keep_after_eviction;
    }

    std::vector<double> CacheEvictionAlgorithm::get_scores_for_all_evictable_blocks(size_t decoder_layer_idx) const {
        auto accumulated_scores_for_current_decoder_layer = m_scores[decoder_layer_idx];
        auto num_tracked_tokens = accumulated_scores_for_current_decoder_layer.size();
        auto counter_for_current_decoder_layer = m_cache_counter[decoder_layer_idx];

        // Make sure that there is at least one block that can be completely evicted
        OPENVINO_ASSERT((num_tracked_tokens + m_eviction_config.get_start_size()) > get_max_cache_size_after_eviction(),
                        "KV cache must be filled before scores for evictable blocks can be computed");

        size_t num_evictable_blocks = get_num_evictable_blocks(decoder_layer_idx);

        std::vector<double> block_scores(num_evictable_blocks);
        for (size_t i = 0; i < num_evictable_blocks; ++i) {
            double normalized_accumulated_attn_score_for_block = 0.0;
            for (size_t j = 0; j < m_block_size; ++j) {
                size_t token_offset = m_block_size * i + j;
                if (m_eviction_config.aggregation_mode == AggregationMode::NORM_SUM) {
                    normalized_accumulated_attn_score_for_block +=
                            accumulated_scores_for_current_decoder_layer[token_offset] /
                            counter_for_current_decoder_layer[token_offset];
                } else {
                    normalized_accumulated_attn_score_for_block += accumulated_scores_for_current_decoder_layer[token_offset];
                }
            }
            block_scores[i] = normalized_accumulated_attn_score_for_block;
        }

        return block_scores;
    }

    std::vector<std::size_t>
    CacheEvictionAlgorithm::get_indices_of_blocks_to_evict(
            const std::vector<double> &scores_for_each_evictable_block, size_t num_blocks_to_evict) const {
        // Returned indices are offsets of blocks to evict, taken from the beginning of the "intermediate", evictable
        // part of the logical KV cache. Indices are sorted in the ascending order.
        auto current_num_evictable_blocks = scores_for_each_evictable_block.size();
        OPENVINO_ASSERT(current_num_evictable_blocks >= num_blocks_to_evict);

        std::vector<std::pair<double, std::size_t>> evictable_block_score_and_index_pairs;
        evictable_block_score_and_index_pairs.reserve(current_num_evictable_blocks);
        for (std::size_t i = 0; i < current_num_evictable_blocks; ++i) {
            evictable_block_score_and_index_pairs.emplace_back(scores_for_each_evictable_block[i], i);
        }

        std::nth_element(evictable_block_score_and_index_pairs.begin(),
                         evictable_block_score_and_index_pairs.begin() + num_blocks_to_evict,
                         evictable_block_score_and_index_pairs.end(),
                         [](const auto &lhs, const auto &rhs) {
                             if (lhs.first < rhs.first) return true;
                             if (lhs.first == rhs.first && lhs.second < rhs.second) return true;
                             return false;
                         });

        evictable_block_score_and_index_pairs.resize(num_blocks_to_evict);

        std::vector<std::size_t> evicted_block_indices;
        evicted_block_indices.reserve(num_blocks_to_evict);
        for (const auto &pair: evictable_block_score_and_index_pairs) {
            evicted_block_indices.push_back(pair.second);
        }

        std::sort(evicted_block_indices.begin(), evicted_block_indices.end());
        return evicted_block_indices;
    }

    void CacheEvictionAlgorithm::remove_scores_of_evicted_blocks(const std::vector<std::size_t> &evicted_block_indices,
                                                                 size_t decoder_layer_idx) {
        if (evicted_block_indices.empty()) {
            return;
        }

        const auto &accumulated_scores_for_current_decoder_layer = m_scores[decoder_layer_idx];
        const auto &counter_for_current_decoder_layer = m_cache_counter[decoder_layer_idx];

        if (m_eviction_config.aggregation_mode == AggregationMode::NORM_SUM) {
            OPENVINO_ASSERT(
                    accumulated_scores_for_current_decoder_layer.size() == counter_for_current_decoder_layer.size());
        }

        auto old_size = accumulated_scores_for_current_decoder_layer.size();
        auto new_size =
                accumulated_scores_for_current_decoder_layer.size() - evicted_block_indices.size() * m_block_size;

        std::vector<double> new_scores;
        new_scores.reserve(new_size);

        std::vector<size_t> new_counter;

        if (m_eviction_config.aggregation_mode == AggregationMode::NORM_SUM) {
            new_counter.reserve(new_size);
        }

        for (size_t token_idx = 0, evicted_block_idx = 0; token_idx < old_size;) {
            if (evicted_block_idx < evicted_block_indices.size() &&
                token_idx == evicted_block_indices[evicted_block_idx] * m_block_size) {
                ++evicted_block_idx;
                token_idx += m_block_size;
                continue;
            }
            new_scores.push_back(accumulated_scores_for_current_decoder_layer[token_idx]);
            if (m_eviction_config.aggregation_mode == AggregationMode::NORM_SUM) {
                new_counter.push_back(counter_for_current_decoder_layer[token_idx]);
            }
            ++token_idx;
        }

        m_scores[decoder_layer_idx] = new_scores;
        m_cache_counter[decoder_layer_idx] = new_counter;
    }

    CacheRotationCalculator::CacheRotationCalculator(size_t block_size,
                                                     size_t max_context_length_in_blocks,
                                                     size_t kv_head_size,
                                                     double rope_theta)
        : m_block_size(block_size),
          m_head_size(kv_head_size) {
        // Frequencies follow the original recipe from RoFormer:
        // https://arxiv.org/pdf/2104.09864v5
        //
        // However, the way the rotation coefficients are ultimately applied in Llama and related models from
        // huggingface is very different from the RoFormer - the embedding-dimension coefficients are not treated as
        // consecutive x-y coordinate pairs, but are rather divided into contiguous x-like and y-like halves - see
        // `rotate_half` function in HF transformers. It can be shown that this form still preserves the relative
        // positioning property from the RoFormer article.
        OPENVINO_ASSERT(rope_theta > 0, "rope_theta must be positive");
        size_t num_freqs = kv_head_size / 2;
        m_rope_sin_lut.resize(max_context_length_in_blocks);
        m_rope_cos_lut.resize(max_context_length_in_blocks);

        for (size_t i = 0; i < max_context_length_in_blocks; i++) {
            m_rope_sin_lut[i].reserve(num_freqs);
            m_rope_cos_lut[i].reserve(num_freqs);
            for (size_t j = 0; j < num_freqs; j++) {
                double exponent = -static_cast<double>(2 * j) / kv_head_size;
                double base_angle = std::pow(rope_theta, exponent);
                m_rope_sin_lut[i].push_back(
                    -std::sin(i * block_size * base_angle));  // minus since we will be rotating by an inverse angle
                m_rope_cos_lut[i].push_back(std::cos(i * block_size * base_angle));
            }
        }
    }

    const std::vector<std::vector<float>>& CacheRotationCalculator::get_sin_lut() const {
        return m_rope_sin_lut;
    }

    const std::vector<std::vector<float>>& CacheRotationCalculator::get_cos_lut() const {
        return m_rope_cos_lut;
    }

    std::vector<CacheRotationCalculator::BlockRotationData> CacheRotationCalculator::get_rotation_data(
        const std::set<size_t>& evicted_block_logical_indices,
        size_t num_logical_blocks_before_eviction,
        bool deltas_only) {


        std::vector<BlockRotationData> retval;
        if (evicted_block_logical_indices.empty()) {
            return retval;
        }

        for (auto idx : evicted_block_logical_indices) {
            OPENVINO_ASSERT(idx < num_logical_blocks_before_eviction);
        }

        // num_logical_blocks_before_eviction > evicted_block_logical_indices.size() is automatically guaranteed by the
        // set property and the previous assertion
        retval.reserve(num_logical_blocks_before_eviction - evicted_block_logical_indices.size());

        ptrdiff_t current_rotation_delta_in_blocks = 0;
        std::vector<size_t> logical_block_space(num_logical_blocks_before_eviction);
        std::iota(logical_block_space.begin(), logical_block_space.end(), 0);

        for (size_t logical_block_idx : logical_block_space) {
            if (evicted_block_logical_indices.find(logical_block_idx) != evicted_block_logical_indices.end()) {
                current_rotation_delta_in_blocks += 1;
            } else {
                if (current_rotation_delta_in_blocks != 0) {
                    BlockRotationData block_rotation_data;
                    block_rotation_data.logical_block_idx = logical_block_idx - current_rotation_delta_in_blocks;

                    // rotation delta is in tokens, but LUT is in blocks right now since we evict per-block
                    // delta recomputation to a valid LUT index is done at a later stage
                    block_rotation_data.rotation_delta = current_rotation_delta_in_blocks * m_block_size;
                    OPENVINO_ASSERT(block_rotation_data.rotation_delta / m_block_size <= m_rope_cos_lut.size(), "rotation delta larger than LUT size");

                    if (!deltas_only) {
                        block_rotation_data.cosines.reserve(m_block_size);
                        block_rotation_data.sines.reserve(m_block_size);
                        for (size_t i = 0; i < m_block_size; i++) {
                            block_rotation_data.cosines.push_back(
                                m_rope_cos_lut[current_rotation_delta_in_blocks]);
                            block_rotation_data.sines.push_back(
                                m_rope_sin_lut[current_rotation_delta_in_blocks]);
                        }
                    }

                    retval.push_back(block_rotation_data);
                }
            }
        }

        return retval;
    }
}
