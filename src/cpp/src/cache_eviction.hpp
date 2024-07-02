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

    // number of tokens in the intermediate part of KV cache to keep
    std::size_t hh_size = 512;

    // the mode used to compute the importance of each token
    AggregationMode aggregation_mode = AggregationMode::NORM_SUM;

    std::size_t get_total_cache_size() {
        return start_size + hh_size + recent_size;
    }

    std::size_t get_num_blocks(std::size_t num_tokens) {
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

    void update_scores(const ov::Tensor& attention_scores) {
        // Increment all tokens occurrence
        if (m_cache_counter) {
            auto dtype = m_cache_counter.get_element_type();
            auto tensor_data = m_cache_counter.data<std::size_t>();
            for (size_t i = 0; i < m_cache_counter.get_size(); ++i) {
                tensor_data[i] += 1;
            }
        }

        // I assumed that we get partially accumulated scores by batch and heads: attn_score.sum(0, 1) -> [num_new_tokens, seq_len]

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

    std::vector<float> group_inter_scores_into_blocks() {
        // Calculate scores only for intermediate (evicting) part
        std::size_t num_tokens_in_last_block = m_cache_counter.get_size() % m_eviction_config.block_size;
        // find intermediate tokens
        std::size_t num_intermediate_tokens = m_cache_counter.get_size() - m_eviction_config.recent_size;
        // if the last block is not completely filled,
        // then we must increase the number of intermediate tokens so that they are divided by the block size
        if (num_tokens_in_last_block != 0) {
            num_intermediate_tokens += m_eviction_config.block_size - num_tokens_in_last_block;
        }
        std::size_t num_intermediate_blocks = static_cast<std::size_t>(num_intermediate_tokens / m_eviction_config.block_size);

        std::vector<float> block_scores(num_intermediate_blocks);
        auto m_scores_data = m_scores.data<float>();
        if (m_eviction_config.aggregation_mode == AggregationMode::NORM_SUM) {
            auto m_cache_counter_data = m_cache_counter.data<std::size_t>();
            std::vector<float> normalized_attn_scores(m_scores.get_size());
            for (size_t i = 0; i < m_cache_counter.get_size(); ++i) {
                normalized_attn_scores[i] = m_scores_data[i] / static_cast<float>(m_cache_counter_data[i]);
            }

            for (size_t i = 0; i < num_intermediate_blocks; ++i) {
                auto start = normalized_attn_scores.begin() + i * m_eviction_config.block_size;
                auto end = start + m_eviction_config.block_size;
                auto sum = std::accumulate(start, end, 0);
                block_scores.push_back(sum);
            }
        } else {
            for (size_t i = 0; i < num_intermediate_blocks; ++i) {
                auto start = m_scores_data + i * m_eviction_config.block_size;
                auto end = start + m_eviction_config.block_size;
                auto sum = std::accumulate(start, end, 0);
                block_scores.push_back(sum);
            }
        }
        return block_scores;
    }

    std::vector<std::size_t> find_blocks_to_evict(const std::vector<float>& block_scores) {
        // block_scores - scores for intermediate (evicting) blocks, doesn't contain scores for start blocks and recent_blocks
        auto hh_blocks = m_eviction_config.get_num_blocks(m_eviction_config.hh_size);
        auto total_blocks = block_scores.size();
        auto num_evicted_blocks = total_blocks - hh_blocks;

        std::vector<std::pair<float, std::size_t>> value_index_pairs;
        value_index_pairs.reserve(total_blocks);
        for (std::size_t i = 0; i < total_blocks; ++i) {
            value_index_pairs.emplace_back(block_scores[i], i);
        }

        std::nth_element(value_index_pairs.begin(), value_index_pairs.begin() + num_evicted_blocks, value_index_pairs.end(),
                         [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

        value_index_pairs.resize(num_evicted_blocks);

        std::vector<std::size_t> evicted_block_ids(num_evicted_blocks);
        for (const auto& pair : value_index_pairs) {
            evicted_block_ids.push_back(pair.second);
        }

        std::sort(evicted_block_ids.begin(), evicted_block_ids.end());
        return evicted_block_ids;
    }

    void remove_evicted_scores(const std::vector<std::size_t>& evict_block_indices) {
        if (evict_block_indices.empty())
            return;

        auto m_scores_data = m_scores.data<float>();
        auto m_cache_counter_data = m_cache_counter.data<std::size_t>();
        auto new_size = m_scores.get_size() - evict_block_indices.size() * m_eviction_config.block_size;

        std::vector<float> scores;
        scores.reserve(new_size);
        std::vector<std::size_t> counter;
        counter.reserve(new_size);
        for (size_t i = 0, it = 0; i < m_scores.get_size();) {
            if (it < evict_block_indices.size() && i == evict_block_indices[it] * m_eviction_config.block_size) {
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

    std::vector<std::size_t> get_remaining_block_indices(const std::vector<std::size_t>& evict_block_indices) {
        std::size_t num_blocks = m_eviction_config.get_num_blocks(m_eviction_config.get_total_cache_size());
        auto start_blocks = m_eviction_config.get_num_blocks(m_eviction_config.start_size);
        auto hh_blocks = m_eviction_config.get_num_blocks(m_eviction_config.hh_size);
        auto recent_blocks = m_eviction_config.get_num_blocks(m_eviction_config.recent_size);
        std::vector<std::size_t> remaining_block_indices;
        remaining_block_indices.reserve(num_blocks);

        for (size_t block_id = 0; block_id < start_blocks; ++block_id) {
            remaining_block_indices.push_back(block_id);
        }
        std::size_t evict_block_id = 0;
        for (size_t block_id = 0; block_id < hh_blocks; ++block_id) {
            if (block_id == evict_block_indices[evict_block_id]) {
                evict_block_id += 1;
            } else {
                remaining_block_indices.push_back(block_id + start_blocks);
            }
        }
        for (size_t block_id = 0; block_id < recent_blocks; ++block_id) {
            remaining_block_indices.push_back(block_id + start_blocks + hh_blocks);
        }
        return remaining_block_indices;
    }

    // TODO: code below is working with ov::Tensor instead of std::vector<ov::Tensor>
    std::vector<std::size_t> apply(const ov::Tensor& attention_scores) {
        update_scores(attention_scores);

        // attention_scores should be accumulated by batch and heads: [new_tokens, seq_len]
        auto kv_cache_size = attention_scores.get_shape()[1];
        if (kv_cache_size <= m_eviction_config.get_total_cache_size()) {
            std::size_t num_blocks = m_eviction_config.get_num_blocks(kv_cache_size);
            std::vector<std::size_t> remaining_block_indices(num_blocks);
            std::iota(remaining_block_indices.begin(), remaining_block_indices.end(), 0);
            return remaining_block_indices; // return all blocks
        }

        // Scores only for intermediate blocks
        auto scores_per_block = group_inter_scores_into_blocks();
        auto evict_block_indices = find_blocks_to_evict(scores_per_block);
        m_num_evicted_tokens += evict_block_indices.size() * m_eviction_config.block_size;

        // update m_scores and m_cache_counter
        remove_evicted_scores(evict_block_indices);

        auto remaining_block_indices = get_remaining_block_indices(evict_block_indices);
        return remaining_block_indices;
    }
};

#endif //OPENVINOGENAI_CACHE_EVICTION_HPP
