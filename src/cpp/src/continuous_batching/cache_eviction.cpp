// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache_eviction.hpp"
#include <queue>

namespace ov::genai {

    void EvictionScoreManager::remove_scores(const std::vector<size_t>& evicted_block_indices, size_t decoder_layer_idx) {
        if (evicted_block_indices.empty()) {
            return;
        }
        const auto &accumulated_scores_for_current_decoder_layer = m_scores[decoder_layer_idx];
        const auto &counter_for_current_decoder_layer = m_cache_counter[decoder_layer_idx];

        if (m_aggregation_mode == AggregationMode::NORM_SUM) {
            OPENVINO_ASSERT(
                    accumulated_scores_for_current_decoder_layer.size() == counter_for_current_decoder_layer.size());
        }

        auto old_size = accumulated_scores_for_current_decoder_layer.size();
        auto new_size =
                accumulated_scores_for_current_decoder_layer.size() - evicted_block_indices.size() * m_block_size;

        std::vector<double> new_scores;
        new_scores.reserve(new_size);

        std::vector<size_t> new_counter;

        if (m_aggregation_mode == AggregationMode::NORM_SUM) {
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
            if (m_aggregation_mode == AggregationMode::NORM_SUM) {
                new_counter.push_back(counter_for_current_decoder_layer[token_idx]);
            }
            ++token_idx;
        }

        m_scores[decoder_layer_idx] = new_scores;
        m_cache_counter[decoder_layer_idx] = new_counter;
    }

    template<class T>
    void _max_pool(std::vector<double>& dst, const T* src_data, size_t size, size_t max_pool_window_size) {
        OPENVINO_ASSERT(size == dst.size());
        for (size_t idx = 0; idx < size; idx++) {
            size_t effective_window_size = max_pool_window_size;
            size_t elements_left = size - idx;
            if (elements_left < effective_window_size) {
                effective_window_size = elements_left;
            }
            auto max_val = src_data[idx];
            for (size_t window_idx = 1; window_idx < effective_window_size; window_idx++) {
                auto val = src_data[idx + window_idx];
                max_val = std::max(val, max_val);
            }
            dst[idx] = max_val;
        }
    }

    void EvictionScoreManager::register_new_token_scores(
            const AttentionScoresForEachDecoderLayer &attention_scores_for_all_decoder_layers,
            const std::set<size_t>& skipped_logical_block_ids, size_t num_snapkv_scores) {

        if (m_num_registered_snapkv_aggregated_scores < m_snapkv_window_size) {
            OPENVINO_ASSERT(num_snapkv_scores + m_num_registered_snapkv_aggregated_scores <= m_snapkv_window_size, "Total number of aggregated SnapKV scores during prefill phase may not be larger than the configured SnapKV window size");
            m_num_registered_snapkv_aggregated_scores += num_snapkv_scores;
        }

        // FIXME (vshampor): currently in terms of counters we do not discern between the cases when the last chunk has been prefill-only
        // or last-prefill-chunk-plus-one-generation_token
        for (size_t decoder_layer_idx = 0; decoder_layer_idx < m_num_decoder_layers; decoder_layer_idx++) {

            const auto &attention_scores = attention_scores_for_all_decoder_layers[decoder_layer_idx];
            // "Start" tokens are never evicted, won't track scores for these
            // "Recent" tokens are also not evicted just yet, but need to accumulate their scores since they may
            // ultimately move into the "intermediate" eviction region of cache
            // Taking the [1, start_size:seq_len] span of the attention scores:
            auto attn_shape = attention_scores.get_shape();
            size_t scores_size_in_tokens = attn_shape[0];
            if (scores_size_in_tokens <= m_ignore_first_n_blocks * m_block_size) {
                return;
            }

            size_t num_skipped_blocks_in_ignore_area = 0;
            for (size_t i = 0; i < m_ignore_first_n_blocks; i++) {
                if (skipped_logical_block_ids.find(i) != skipped_logical_block_ids.end()) {
                    num_skipped_blocks_in_ignore_area++;
                }
            }

            OPENVINO_ASSERT(num_skipped_blocks_in_ignore_area <= m_ignore_first_n_blocks);

            size_t start_token_offset_in_scores = (m_ignore_first_n_blocks - num_skipped_blocks_in_ignore_area) * m_block_size;

            auto hh_score = ov::Tensor(
                    attention_scores,
                    ov::Coordinate{start_token_offset_in_scores},
                    ov::Coordinate{scores_size_in_tokens}
            );

            std::vector<double> processed_hh_scores(hh_score.get_size());

            if (m_aggregation_mode == AggregationMode::ADAPTIVE_RKV) {
                _max_pool(processed_hh_scores, hh_score.data<float>(), hh_score.get_size(), 1);
            } else {
                _max_pool(processed_hh_scores, hh_score.data<float>(), hh_score.get_size(), m_max_pool_window_size);
            }

            auto& accumulated_scores_for_current_decoder_layer = m_scores[decoder_layer_idx];

            if (accumulated_scores_for_current_decoder_layer.empty()) {
                _accumulate_initial_scores(processed_hh_scores, decoder_layer_idx, num_snapkv_scores, skipped_logical_block_ids);
            } else {
                _accumulate_with_existing_scores(processed_hh_scores, decoder_layer_idx, num_snapkv_scores, skipped_logical_block_ids);
            }
        }
    }


    void EvictionScoreManager::_accumulate_initial_scores(const std::vector<double>& max_pooled_hh_scores, size_t decoder_layer_idx, size_t num_snapkv_scores, const std::set<size_t>& skipped_logical_block_ids) {
        if (m_snapkv_window_size != 0 && num_snapkv_scores == 0) {
            // SnapKV window not yet reached, no meaningful scores to accumulate
            return;
        }


        OPENVINO_ASSERT(m_previous_scores_queues[decoder_layer_idx].empty());
        m_previous_scores_queues[decoder_layer_idx].emplace_back(max_pooled_hh_scores, skipped_logical_block_ids);
        auto& accumulated_scores_for_current_decoder_layer = m_scores[decoder_layer_idx];
        _initialize_score_with_skips(accumulated_scores_for_current_decoder_layer, max_pooled_hh_scores, skipped_logical_block_ids);

        if (m_aggregation_mode == AggregationMode::NORM_SUM) {
            std::size_t new_scores_size = max_pooled_hh_scores.size();
            std::vector<std::size_t> counter(new_scores_size);
            if (m_snapkv_window_size == 0) {
                // Will simulate that the tokens comprising the sequence were added one-by-one
                // from the standpoint of the occurrence tracker
                std::generate(counter.begin(), counter.begin() + new_scores_size,
                              [&new_scores_size] { return new_scores_size--; });
            }
            else {
                OPENVINO_ASSERT(num_snapkv_scores > 0);
                OPENVINO_ASSERT(new_scores_size >= num_snapkv_scores);
                std::fill(counter.begin(), counter.end() - num_snapkv_scores, num_snapkv_scores);
                std::iota(counter.rbegin(), counter.rbegin() + num_snapkv_scores, 1);
            }
            m_cache_counter[decoder_layer_idx] = counter;
        }
    }

    void EvictionScoreManager::_accumulate_with_existing_scores(const std::vector<double>& max_pooled_hh_scores, size_t decoder_layer_idx, size_t num_snapkv_scores, const std::set<size_t>& skipped_logical_block_ids) {
        if (m_aggregation_mode == AggregationMode::ADAPTIVE_RKV) {
            if (m_previous_scores_queues[decoder_layer_idx].size() >= m_adaptive_rkv_window_size) {
                m_previous_scores_queues[decoder_layer_idx].pop_front();
            }
            m_previous_scores_queues[decoder_layer_idx].emplace_back(max_pooled_hh_scores, skipped_logical_block_ids);
        }

        if (m_aggregation_mode == AggregationMode::ADAPTIVE_RKV) {
            OPENVINO_ASSERT(!m_previous_scores_queues[decoder_layer_idx].empty());
            auto start_it = m_previous_scores_queues[decoder_layer_idx].begin();
            auto& dst = m_scores[decoder_layer_idx];
            _initialize_score_with_skips(dst, start_it->score, start_it->skips);
            for (auto it = start_it + 1; it != m_previous_scores_queues[decoder_layer_idx].end(); it++) {
                _accumulate_layer_scores_to(decoder_layer_idx, it->score, it->skips, dst);
            }
            // mean
            size_t queue_size = m_previous_scores_queues[decoder_layer_idx].size();
            for (size_t i = 0; i < dst.size(); i++) {
                dst[i] /= queue_size;
            }
            auto score_copy = dst;
            _max_pool(dst, score_copy.data(), score_copy.size(), m_max_pool_window_size);
        } else {
            auto& accumulated_scores_for_current_decoder_layer = m_scores[decoder_layer_idx];
            size_t old_size_in_tokens = accumulated_scores_for_current_decoder_layer.size();
            _accumulate_layer_scores_to(decoder_layer_idx, max_pooled_hh_scores, skipped_logical_block_ids, accumulated_scores_for_current_decoder_layer);
            size_t new_size_in_tokens = accumulated_scores_for_current_decoder_layer.size();

            if (m_aggregation_mode == AggregationMode::NORM_SUM) {
                _adjust_norm_sum_counters(decoder_layer_idx, old_size_in_tokens, new_size_in_tokens);
            }
        }
    }

    void EvictionScoreManager::_accumulate_layer_scores_to(size_t decoder_layer_idx, const std::vector<double>& src, const std::set<size_t>& skipped_logical_block_ids, std::vector<double>& dst) {
        std::set<size_t> skip_set_adjusted;

        for (size_t skipped_block_id : skipped_logical_block_ids) {
            if (skipped_block_id >= m_ignore_first_n_blocks) {
                skip_set_adjusted.insert(skipped_block_id - m_ignore_first_n_blocks);
            } // else do not include this block in the adjusted skip set since it is in the start area already
        }
        size_t old_size_in_tokens = dst.size();
        size_t new_size_in_tokens = src.size() + m_block_size * skipped_logical_block_ids.size();

        OPENVINO_ASSERT(new_size_in_tokens >= old_size_in_tokens);
        dst.resize(new_size_in_tokens);
        add_with_skips(dst, src, skip_set_adjusted);
    }

    void EvictionScoreManager::_adjust_norm_sum_counters(size_t decoder_layer_idx, size_t old_size_in_tokens, size_t new_size_in_tokens) {
        OPENVINO_ASSERT(new_size_in_tokens >= old_size_in_tokens);
        size_t num_new_tokens = new_size_in_tokens - old_size_in_tokens;
        auto &counter_for_current_decoder_layer = m_cache_counter[decoder_layer_idx];
        counter_for_current_decoder_layer.resize(new_size_in_tokens);
        if (m_snapkv_window_size == 0 || m_num_registered_snapkv_aggregated_scores == m_snapkv_window_size) {
            // Increment occurrence counts of all currently tracked cache blocks
            for (auto it = counter_for_current_decoder_layer.begin();
                 it != counter_for_current_decoder_layer.end(); it++) {
                *it += num_new_tokens;
            }
            // Add occurrence counts for new tokens like above
            for (size_t i = 0; i < num_new_tokens; i++) {
                auto idx = old_size_in_tokens + i;
                counter_for_current_decoder_layer[idx] = num_new_tokens - i;
            }
        }
        else {
            OPENVINO_ASSERT(new_size_in_tokens >= m_num_registered_snapkv_aggregated_scores);
            std::fill(counter_for_current_decoder_layer.begin(), counter_for_current_decoder_layer.end() - m_num_registered_snapkv_aggregated_scores, m_num_registered_snapkv_aggregated_scores);
            std::iota(counter_for_current_decoder_layer.rbegin(), counter_for_current_decoder_layer.rbegin() + m_num_registered_snapkv_aggregated_scores, 1);
        }
    }

    void EvictionScoreManager::_initialize_score_with_skips(std::vector<double>& dst, const std::vector<double>& src, const std::set<size_t> skipped_logical_block_ids) {
        // New sequence to track
        if (skipped_logical_block_ids.empty()) {
            dst = src;
        }
        else {
            dst.clear();
            dst.resize(src.size() + m_block_size * skipped_logical_block_ids.size(), 0.0);
            size_t src_idx = 0;
            for (size_t dst_idx = 0; dst_idx < dst.size(); dst_idx++) {
                size_t curr_logical_block_idx = dst_idx / m_block_size;
                if (skipped_logical_block_ids.find(curr_logical_block_idx) != skipped_logical_block_ids.end()) {
                    dst_idx = curr_logical_block_idx * m_block_size + m_block_size - 1;
                    continue;
                }
                dst[dst_idx] = src[src_idx];
                src_idx++;
            }
            OPENVINO_ASSERT(src_idx == src.size());
        }
    }

    size_t EvictionScoreManager::get_current_scores_length_in_tokens(size_t layer_idx) const {
        return m_scores[layer_idx].size();
    }

    const std::vector<std::vector<double>>& EvictionScoreManager::get_scores() const {
        return m_scores;
    }

    const std::vector<std::vector<size_t>>& EvictionScoreManager::get_counters() const {
        return m_cache_counter;
    }

    void EvictionScoreManager::add_with_skips(std::vector<double>& dst, const std::vector<double>& src, const std::set<size_t>& skipped_logical_block_ids) const {
            OPENVINO_ASSERT(skipped_logical_block_ids.size() * m_block_size + src.size() == dst.size());
            size_t src_idx = 0;
            for (size_t dst_idx = 0; dst_idx < dst.size(); dst_idx++) {
                size_t curr_logical_block_idx = dst_idx / m_block_size;
                if (skipped_logical_block_ids.find(curr_logical_block_idx) != skipped_logical_block_ids.end()) {
                    dst_idx = m_block_size * (curr_logical_block_idx + 1) - 1;
                    continue;
                }
                dst[dst_idx] += src[src_idx];
                src_idx++;
            }
            OPENVINO_ASSERT(src_idx == src.size());
    }

    CacheEvictionAlgorithm::CacheEvictionAlgorithm(const CacheEvictionConfig &eviction_config, size_t block_size,
                                                   size_t num_decoder_layers, size_t max_pool_window_size) :
            m_eviction_config(eviction_config), m_block_size(block_size), m_num_decoder_layers(num_decoder_layers),
            m_score_manager(block_size, num_decoder_layers, max_pool_window_size, eviction_config.aggregation_mode, eviction_config.get_start_size() / block_size, eviction_config.snapkv_window_size, eviction_config.adaptive_rkv_config.window_size), m_kvcrush_algo(eviction_config.kvcrush_config, block_size)
    {
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

        const auto& scores = m_score_manager.get_scores();
        for (size_t decoder_layer_idx = 0; decoder_layer_idx < scores.size(); decoder_layer_idx++) {
            const auto &accumulated_scores_for_current_decoder_layer = scores[decoder_layer_idx];
            auto scores_length = accumulated_scores_for_current_decoder_layer.size();
            if (scores_length + m_eviction_config.get_start_size() <= get_max_cache_size_after_eviction()) {
                // KV cache is not yet filled, keep all currently occupied blocks
                continue;
            }


            auto scores_for_all_evictable_blocks = get_scores_for_all_evictable_blocks(decoder_layer_idx);
            std::vector<size_t> evicted_block_indices;
            if (m_eviction_config.aggregation_mode == AggregationMode::ADAPTIVE_RKV) {
                OPENVINO_ASSERT(!m_last_token_similarity.empty(), "Token similarity must be registered before each eviction in the Adaptive R-KV scenario");
                size_t num_evictable_blocks = get_num_evictable_blocks(decoder_layer_idx);
                size_t num_similarity_tokens_registered = m_last_token_similarity[0].size();
                OPENVINO_ASSERT(num_similarity_tokens_registered / m_block_size == num_evictable_blocks, "Similarity score size mismatch - registered ", num_similarity_tokens_registered / m_block_size, " blocks worth of similarity scores, but have ", num_evictable_blocks, " evictable blocks");
                auto similarity_set_and_num_blocks_kept = get_adaptive_rkv_similarity_set(num_evictable_blocks, scores_for_all_evictable_blocks);

                const auto& similarity_set = similarity_set_and_num_blocks_kept.first;
                size_t num_blocks_kept = similarity_set_and_num_blocks_kept.second;

                size_t num_evictable_blocks_to_keep_after_eviction = m_eviction_config.get_evictable_size() / m_block_size;
                OPENVINO_ASSERT(num_blocks_kept <= num_evictable_blocks_to_keep_after_eviction);
                size_t num_blocks_left_to_fill =  num_evictable_blocks_to_keep_after_eviction - num_blocks_kept;
                auto diverse_set = get_adaptive_rkv_diverse_blocks(num_blocks_left_to_fill, similarity_set, m_last_token_similarity[decoder_layer_idx]);

                for (size_t potentially_evicted_idx : similarity_set) {
                    if (diverse_set.find(potentially_evicted_idx) == diverse_set.end()) {
                        evicted_block_indices.push_back(potentially_evicted_idx);
                    }
                }

            } else {
                // Only the blocks in the "intermediate" part of the logical KV cache will be considered for eviction
                size_t num_blocks_to_evict = get_num_blocks_to_evict(decoder_layer_idx);
                evicted_block_indices = get_indices_of_blocks_to_evict(scores_for_all_evictable_blocks, num_blocks_to_evict);
                // KVCrush: start
                bool should_apply_kvcrush = (m_eviction_config.kvcrush_config.budget > 0) &&
                                            (evicted_block_indices.size() >= m_eviction_config.kvcrush_config.budget);
                if (should_apply_kvcrush) {
                    size_t num_tokens_in_evictable_blocks = scores_for_all_evictable_blocks.size() * m_block_size;

                    auto kvcrush_retained_block_indices = m_kvcrush_algo.get_indices_of_blocks_to_retain_using_kvcrush(
                        num_tokens_in_evictable_blocks,
                        evicted_block_indices,
                        m_score_manager.get_scores()[decoder_layer_idx]);

                    // Remove the indices in kvcrush_retained_block_indices from evicted_block_indices
                    if (!kvcrush_retained_block_indices.empty()) {
                        // Convert both vectors to sets for efficient operations
                        std::unordered_set<std::size_t> retained_set(kvcrush_retained_block_indices.begin(),
                                                                     kvcrush_retained_block_indices.end());

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
                // KVCrush: end
            }

            m_num_evicted_tokens += evicted_block_indices.size() * m_block_size;

            // No longer need to track the overall "heavy-hitter" attention scores for freshly evicted blocks
            remove_scores_of_evicted_blocks(evicted_block_indices, decoder_layer_idx);

            // Adjust indices to account for start area
            for (auto &idx: evicted_block_indices) idx += get_num_blocks(m_eviction_config.get_start_size());
            for (auto &idx: evicted_block_indices) retval[decoder_layer_idx].insert(idx);
        }

        m_last_token_similarity.clear();
        return retval;
    }

    std::pair<std::set<size_t>, size_t> CacheEvictionAlgorithm::get_adaptive_rkv_similarity_set(size_t max_num_blocks_kept, const std::vector<double>& evictable_area_block_scores) {
        struct ScoreAndBlockIdx {
            double score;
            size_t block_idx;
            bool operator<(const ScoreAndBlockIdx& rhs) const { return score < rhs.score; }
        };
        std::priority_queue<ScoreAndBlockIdx> score_block_queue;
        double total_sum = 0.0;
        for (size_t i = 0; i < evictable_area_block_scores.size(); i++) {
            total_sum += evictable_area_block_scores[i];
            score_block_queue.push({evictable_area_block_scores[i], i});
        }

        double expected_sum = total_sum * m_eviction_config.adaptive_rkv_config.attention_mass;
        std::set<size_t> retval;

        double sum = 0.0;
        size_t num_blocks_kept = 0;
        while (sum < expected_sum && !score_block_queue.empty() && num_blocks_kept <= max_num_blocks_kept) {
            // Blocks with most attention mass are kept
            auto score_and_idx = score_block_queue.top();
            sum += score_and_idx.score;
            score_block_queue.pop();
            num_blocks_kept += 1;
        }

        // The rest will be further filtered according to their cosine similarity separately
        while (!score_block_queue.empty()) {
            auto score_and_idx = score_block_queue.top();
            retval.insert(score_and_idx.block_idx);
            score_block_queue.pop();
        }
        return {retval, num_blocks_kept};
    }

    std::set<size_t> CacheEvictionAlgorithm::get_adaptive_rkv_diverse_blocks(size_t num_blocks_left_to_fill, const std::set<size_t>& similarity_set, const std::vector<double>& token_similarity) {
        OPENVINO_ASSERT(num_blocks_left_to_fill <= similarity_set.size());
        struct ScoreAndBlockIdx {
            double score;
            size_t block_idx;
            bool operator<(const ScoreAndBlockIdx& rhs) const { return score < rhs.score; } // sic!
        };
        std::priority_queue<ScoreAndBlockIdx> score_block_queue;
        OPENVINO_ASSERT(token_similarity.size() % m_block_size == 0);
        for (size_t block_idx : similarity_set) {
            OPENVINO_ASSERT(block_idx * m_block_size <= token_similarity.size());
            double block_diversity_score = 0.0;
            for (size_t tok_idx = 0; tok_idx < m_block_size; tok_idx++) {
                block_diversity_score -= token_similarity[block_idx * m_block_size + tok_idx];
            }
            score_block_queue.push({block_diversity_score, block_idx});
            block_diversity_score = 0.0;
        }

        std::set<size_t> retval;

        while (retval.size() < num_blocks_left_to_fill && !score_block_queue.empty()) {
            auto score_and_idx = score_block_queue.top();
            retval.insert(score_and_idx.block_idx);
            score_block_queue.pop();
        }
        return retval;
    }

    CacheEvictionAlgorithm::CacheEvictionRange CacheEvictionAlgorithm::get_evictable_block_range() const {
        return get_evictable_block_range(0);
    }

    CacheEvictionAlgorithm::CacheEvictionRange CacheEvictionAlgorithm::get_evictable_block_range(size_t layer_idx) const {
        std::size_t current_sequence_length = m_eviction_config.get_start_size() + m_score_manager.get_current_scores_length_in_tokens(layer_idx);
        if (current_sequence_length <= get_max_cache_size_after_eviction()) {
            return CacheEvictionRange::invalid(); // purposely invalid range since no eviction can take place yet
        }
        std::size_t start = m_eviction_config.get_start_size() / m_block_size;
        std::size_t end = current_sequence_length / m_block_size - (m_eviction_config.get_recent_size() / m_block_size);
        return {start, end};
    }

    void CacheEvictionAlgorithm::register_new_token_scores(
            const AttentionScoresForEachDecoderLayer &attention_scores_for_all_decoder_layers, size_t num_snapkv_scores_aggregated) {
        register_new_token_scores(attention_scores_for_all_decoder_layers, {}, num_snapkv_scores_aggregated);
    }

    void CacheEvictionAlgorithm::register_new_token_scores(
            const AttentionScoresForEachDecoderLayer &attention_scores_for_all_decoder_layers,
            const std::set<size_t>& skipped_logical_block_ids,
            size_t num_snapkv_scores_aggregated) {
        m_score_manager.register_new_token_scores(attention_scores_for_all_decoder_layers, skipped_logical_block_ids, num_snapkv_scores_aggregated);
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
        const auto& accumulated_scores_for_current_decoder_layer = m_score_manager.get_scores()[decoder_layer_idx];
        auto num_tracked_tokens = accumulated_scores_for_current_decoder_layer.size();
        const auto& counter_for_current_decoder_layer = m_score_manager.get_counters()[decoder_layer_idx];

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
        m_score_manager.remove_scores(evicted_block_indices, decoder_layer_idx);
    }

    void CacheEvictionAlgorithm::register_token_similarity(const TokenSimilarityForEachDecoderLayer& token_similarity_for_all_decoder_layers) {
        OPENVINO_ASSERT(m_last_token_similarity.empty(), "CacheEvictionAlgorithm already has token similarity, must evict before new similarity is registered");
        OPENVINO_ASSERT(token_similarity_for_all_decoder_layers.size() == m_num_decoder_layers);
        m_last_token_similarity.resize(m_num_decoder_layers);
        for (size_t layer_idx = 0; layer_idx < m_num_decoder_layers; layer_idx++)
        {
            const auto& layer_similarity = token_similarity_for_all_decoder_layers[layer_idx];
            const float* data = layer_similarity.data<float>();
            size_t num_similarity_tokens = layer_similarity.get_size();
            OPENVINO_ASSERT(num_similarity_tokens % m_block_size == 0);
            m_last_token_similarity[layer_idx].resize(num_similarity_tokens);
            for (size_t tok_idx = 0; tok_idx < layer_similarity.get_size(); tok_idx++) {
                m_last_token_similarity[layer_idx][tok_idx] = data[tok_idx];
            }
        }
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

size_t SnapKVScoreAggregationCalculator::get_num_token_scores_to_aggregate(size_t prompt_len, size_t num_scheduled_tokens, size_t num_processed_tokens) {
    if (m_snapkv_window_size == 0) {
        // If SnapKV is disabled, aggregate all available scores in this chunk
        return num_scheduled_tokens;
    }
    size_t first_scored_token_position = m_snapkv_window_size > prompt_len ? 0 : prompt_len - m_snapkv_window_size;
    size_t num_scored_token_positions_in_this_chunk = 0;
    size_t num_processed_tokens_before_this_chunk = num_processed_tokens;
    size_t num_processed_tokens_after_this_chunk = num_processed_tokens_before_this_chunk + num_scheduled_tokens;
    if (num_processed_tokens_after_this_chunk > first_scored_token_position) {
        if (num_processed_tokens_before_this_chunk > first_scored_token_position) {
            num_scored_token_positions_in_this_chunk = num_scheduled_tokens;
        }
        else {
            num_scored_token_positions_in_this_chunk = num_processed_tokens_after_this_chunk - first_scored_token_position;
        }

    }
    return num_scored_token_positions_in_this_chunk;
}

}

