// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENVINOGENAI_CACHE_EVICTION_HPP
#define OPENVINOGENAI_CACHE_EVICTION_HPP


#include <vector>
#include <cstdlib>
#include <cmath>

#include "openvino/openvino.hpp"

enum class AggregationMode {
    SUM,
    NORM_SUM
};

struct CacheEvictionConfig {
    // number of tokens in kv_block
    std::size_t block_size = 16;

    // number of tokens in the begging of KV cache to keep
    std::size_t start_size = 32;

    // number of recent tokens to keep for KV cache
    std::size_t recent_size = 128;

    // number of tokens in the intermediate, evictable part of KV cache to keep
    std::size_t evictable_size = 512;

    // the mode used to compute the importance of each token
    AggregationMode aggregation_mode = AggregationMode::NORM_SUM;

    std::size_t get_total_cache_size() const {
        return start_size + evictable_size + recent_size;
    }

    std::size_t get_num_blocks(std::size_t num_tokens) const{
        return static_cast<std::size_t>(std::ceil(num_tokens / block_size));
    }
};

class CacheEvictionAlgorithm {
    CacheEvictionConfig m_eviction_config;
    ov::Tensor m_scores;
    ov::Tensor m_cache_counter;
    std::size_t m_num_evicted_tokens = 0;
public:
    explicit CacheEvictionAlgorithm(const CacheEvictionConfig& eviction_config)  :
            m_eviction_config(eviction_config) {}

    void register_new_token_scores(const ov::Tensor& attention_scores) {
        // Increment all tokens occurrence
        if (m_cache_counter) {
            auto dtype = m_cache_counter.get_element_type();
            auto tensor_data = m_cache_counter.data<std::size_t>();
            for (size_t i = 0; i < m_cache_counter.get_size(); ++i) {
                tensor_data[i] += 1;
            }
        }

        // Skip frozen start tokens in cache -> [1, start_size:seq_len], 1 -> cross-head
        auto attn_shape = attention_scores.get_shape();
        auto hh_score = ov::Tensor(
                attention_scores,
                ov::Coordinate{0, m_eviction_config.start_size},
                ov::Coordinate{attn_shape[0], attn_shape[1]}
        );
        hh_score.set_shape(ov::Shape{attn_shape[1] - m_eviction_config.start_size});

        if (!m_scores) {
            m_scores = hh_score;
            if (m_eviction_config.aggregation_mode == AggregationMode::NORM_SUM) {
                std::size_t scores_size = hh_score.get_size();
                std::vector<std::size_t> counter(scores_size);
                // should be from (total_size - start_size) to 1
                std::generate(counter.begin(), counter.end(), [&scores_size]{ return scores_size--;});
                m_cache_counter = ov::Tensor(ov::element::u64, ov::Shape{scores_size}, counter.data());
            }
        } else {
            auto m_scores_data = m_scores.data<float>();
            auto hh_scores_data = hh_score.data<float>();
            for (size_t i = 0; i < m_scores.get_size(); ++i) {
                hh_scores_data[i] += m_scores_data[i];
            }
        }
        m_scores = hh_score;

        size_t num_new_tokens = hh_score.get_size() - m_scores.get_size();
        if (m_eviction_config.aggregation_mode == AggregationMode::NORM_SUM) {
            std::vector<std::size_t> new_tokens_counter(hh_score.get_size());
            for (size_t i = 0; i < m_cache_counter.get_size(); ++i) {
                new_tokens_counter[i] = m_cache_counter.data<std::size_t>()[i];
            }
            for (size_t i = m_cache_counter.get_size(); i < new_tokens_counter.size(); ++i) {
                new_tokens_counter[i] = num_new_tokens - (i - m_cache_counter.get_size());
            }
            m_cache_counter = ov::Tensor(ov::element::u64, ov::Shape{hh_score.get_size()}, new_tokens_counter.data());
        }
    }

    std::vector<float> get_scores_for_all_evictable_blocks() {
        std::size_t num_tokens_in_last_block = m_cache_counter.get_size() % m_eviction_config.block_size;
        // m_cache_counter does not consider the non-evictable "start" tokens, so its size is comprised of the
        // intermediate "evictable" part and the "recent" part
        std::size_t num_evictable_tokens = m_cache_counter.get_size() - m_eviction_config.recent_size;

        // if the last block is not completely filled,
        // then we must increase the number of intermediate tokens so that they are divisible by the block size
        // TODO (vshampor): why?
        if (num_tokens_in_last_block != 0) {
            num_evictable_tokens += m_eviction_config.block_size - num_tokens_in_last_block;
        }
        auto num_evictable_blocks = static_cast<std::size_t>(num_evictable_tokens / m_eviction_config.block_size);

        std::vector<float> block_scores(num_evictable_blocks);
        auto m_scores_data = m_scores.data<float>();
        if (m_eviction_config.aggregation_mode == AggregationMode::NORM_SUM) {
            auto m_cache_counter_data = m_cache_counter.data<std::size_t>();
            std::vector<float> normalized_attn_scores(m_scores.get_size());
            for (size_t i = 0; i < m_cache_counter.get_size(); ++i) {
                normalized_attn_scores[i] = m_scores_data[i] / static_cast<float>(m_cache_counter_data[i]);
            }

            for (size_t i = 0; i < num_evictable_blocks; ++i) {
                auto start = normalized_attn_scores.begin() + i * m_eviction_config.block_size;
                auto end = start + m_eviction_config.block_size;
                auto sum = std::accumulate(start, end, 0);
                block_scores.push_back(sum);
            }
        } else {
            for (size_t i = 0; i < num_evictable_blocks; ++i) {
                auto start = m_scores_data + i * m_eviction_config.block_size;
                auto end = start + m_eviction_config.block_size;
                auto sum = std::accumulate(start, end, 0);
                block_scores.push_back(sum);
            }
        }
        return block_scores;
    }

    std::vector<std::size_t> get_indices_of_blocks_to_evict(const std::vector<float>& scores_for_each_evictable_block) {
        // Returned indices are offsets of blocks to evict, taken from the beginning of the "intermediate", evictable
        // part of the logical KV cache. Indices are sorted in the ascending order.
        auto num_evictable_blocks_to_keep = m_eviction_config.get_num_blocks(m_eviction_config.evictable_size);
        auto current_num_evictable_blocks = scores_for_each_evictable_block.size();
        OPENVINO_ASSERT(current_num_evictable_blocks >= num_evictable_blocks_to_keep);
        if (current_num_evictable_blocks == num_evictable_blocks_to_keep) {
            return {}; // nothing to evict yet
        }

        auto num_blocks_to_evict = current_num_evictable_blocks - num_evictable_blocks_to_keep;

        std::vector<std::pair<float, std::size_t>> evictable_block_score_and_index_pairs;
        evictable_block_score_and_index_pairs.reserve(current_num_evictable_blocks);
        for (std::size_t i = 0; i < current_num_evictable_blocks; ++i) {
            evictable_block_score_and_index_pairs.emplace_back(scores_for_each_evictable_block[i], i);
        }

        std::nth_element(evictable_block_score_and_index_pairs.begin(),
                         evictable_block_score_and_index_pairs.begin() + num_blocks_to_evict,
                         evictable_block_score_and_index_pairs.end(),
                         [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

        evictable_block_score_and_index_pairs.resize(num_blocks_to_evict);

        std::vector<std::size_t> evicted_block_indices(num_blocks_to_evict);
        for (const auto& pair : evictable_block_score_and_index_pairs) {
            evicted_block_indices.push_back(pair.second);
        }

        std::sort(evicted_block_indices.begin(), evicted_block_indices.end());
        return evicted_block_indices;
    }

    void remove_scores_of_evicted_blocks(const std::vector<std::size_t>& evicted_block_indices) {
        if (evicted_block_indices.empty()) {
            return;
        }

        auto m_scores_data = m_scores.data<float>();
        auto m_cache_counter_data = m_cache_counter.data<std::size_t>();
        auto new_size = m_scores.get_size() - evicted_block_indices.size() * m_eviction_config.block_size;

        std::vector<float> scores;
        scores.reserve(new_size);

        std::vector<std::size_t> counter;
        counter.reserve(new_size);

        for (size_t i = 0, it = 0; i < m_scores.get_size();) {
            if (it < evicted_block_indices.size() && i == evicted_block_indices[it] * m_eviction_config.block_size) {
                ++it;
                i += m_eviction_config.block_size;
                continue;
            }
            scores.push_back(m_scores_data[i]);
            counter.push_back(m_cache_counter_data[i]);
            ++i;
        }

        m_scores = ov::Tensor(m_scores.get_element_type(), ov::Shape{new_size}, scores.data());
        m_cache_counter = ov::Tensor(m_cache_counter.get_element_type(), ov::Shape{new_size}, counter.data());
    }

    std::vector<std::size_t> get_remaining_block_indices(const std::vector<std::size_t>& evicted_block_indices) {
        // evicted_block_indices must be sorted in the ascending order, and correspond to offsets of the blocks to evict starting
        // from the first block in the "intermediate", evictable part of the logical KV cache associated with current
        // sequence or sequence group
        std::size_t num_blocks_in_cache = m_eviction_config.get_num_blocks(m_eviction_config.get_total_cache_size());
        auto num_start_blocks = m_eviction_config.get_num_blocks(m_eviction_config.start_size);
        auto num_evictable_blocks = m_eviction_config.get_num_blocks(m_eviction_config.evictable_size);
        auto num_recent_blocks = m_eviction_config.get_num_blocks(m_eviction_config.recent_size);
        std::vector<std::size_t> remaining_block_indices;
        remaining_block_indices.reserve(num_blocks_in_cache);

        for (size_t block_id = 0; block_id < num_start_blocks; ++block_id) {
            remaining_block_indices.push_back(block_id);
        }
        std::size_t evictable_block_id = 0;
        for (size_t block_id = 0; block_id < num_evictable_blocks; ++block_id) {
            if (block_id == evicted_block_indices[evictable_block_id]) {
                evictable_block_id += 1;
            } else {
                remaining_block_indices.push_back(block_id + num_start_blocks);
            }
        }
        for (size_t block_id = 0; block_id < num_recent_blocks; ++block_id) {
            remaining_block_indices.push_back(block_id + num_start_blocks + num_evictable_blocks);
        }
        return remaining_block_indices;
    }

    // TODO: code below is working with ov::Tensor instead of std::vector<ov::Tensor>
    std::vector<std::size_t> apply(const ov::Tensor& attention_scores) {
        // Returns the indices of logical KV cache blocks to be *kept* (the rest is to be discarded).
        // The kept indices are determined using `attention_scores`, which is expected to be the
        // attention head scores that are already reduced by the batch and head dimensions, i.e. the shape of
        // `attention_scores` must be [num_new_tokens, current_seq_len], where `num_new_tokens` is the dimension
        // corresponding to the number of freshly generated tokens since the last cache eviction has taken place,
        // and the `current_seq_len` is the dimension corresponding to the current sequence length at this stage
        // in the generation process, i.e. the dimension over which the attention scores over individual previous
        // tokens was being computed.

        register_new_token_scores(attention_scores);

        auto current_sequence_length = attention_scores.get_shape()[1];
        if (current_sequence_length <= m_eviction_config.get_total_cache_size()) {
            // KV cache is not yet filled, keep all currently occupied blocks
            std::size_t num_blocks = m_eviction_config.get_num_blocks(current_sequence_length);
            std::vector<std::size_t> remaining_block_indices(num_blocks);
            std::iota(remaining_block_indices.begin(), remaining_block_indices.end(), 0);
            return remaining_block_indices;
        }

        // Only the blocks in the "intermediate" part of the logical KV cache will be considered for eviction
        auto scores_for_all_evictable_blocks = get_scores_for_all_evictable_blocks();
        auto evicted_block_indices = get_indices_of_blocks_to_evict(scores_for_all_evictable_blocks);
        m_num_evicted_tokens += evicted_block_indices.size() * m_eviction_config.block_size;

        // No longer need to track the overall "heavy-hitter" attention scores for freshly evicted blocks
        remove_scores_of_evicted_blocks(evicted_block_indices);

        auto remaining_block_indices = get_remaining_block_indices(evicted_block_indices);
        return remaining_block_indices;
    }
};

#endif //OPENVINOGENAI_CACHE_EVICTION_HPP
