// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENVINOGENAI_CACHE_EVICTION_HPP
#define OPENVINOGENAI_CACHE_EVICTION_HPP


#include <vector>
#include <cstdlib>
#include <cmath>

#include "openvino/openvino.hpp"
#include "attention_output.hpp"

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
public:
    explicit CacheEvictionAlgorithm(const CacheEvictionConfig& eviction_config, size_t num_decoder_layers)  :
            m_eviction_config(eviction_config), m_num_decoder_layers(num_decoder_layers), m_cache_counter(num_decoder_layers) {}
    // TODO: code below is working with ov::Tensor instead of std::vector<ov::Tensor>
    std::vector<std::vector<std::size_t>> get_logical_block_indices_to_keep(const AttentionScoresForEachDecoderLayer& attention_scores_for_all_decoder_layers) {
        // Returns the indices of logical KV cache blocks to be *kept* (the rest is to be discarded) for each decoder layer in order.
        // The kept indices are determined using `attention_scores`, which is expected to be the
        // attention head scores that are already reduced by the batch and head dimensions, i.e. the shape of
        // `attention_scores` must be [num_new_tokens, current_seq_len], where `num_new_tokens` is the dimension
        // corresponding to the number of freshly generated tokens since the last cache eviction has taken place,
        // and the `current_seq_len` is the dimension corresponding to the current sequence length at this stage
        // in the generation process, i.e. the dimension over which the attention scores over individual previous
        // tokens was being computed.

        register_new_token_scores(attention_scores_for_all_decoder_layers);

        std::vector<std::vector<size_t>> retval(m_num_decoder_layers);

        for (size_t decoder_layer_idx = 0; decoder_layer_idx < attention_scores_for_all_decoder_layers.size(); decoder_layer_idx++) {
            auto attention_scores = attention_scores_for_all_decoder_layers[decoder_layer_idx];
            auto current_sequence_length = attention_scores.get_shape()[1];
            if (current_sequence_length <= m_eviction_config.get_total_cache_size()) {
                // KV cache is not yet filled, keep all currently occupied blocks
                std::size_t num_blocks = m_eviction_config.get_num_blocks(current_sequence_length);
                std::vector<std::size_t> remaining_block_indices(num_blocks);
                std::iota(remaining_block_indices.begin(), remaining_block_indices.end(), 0);
                retval.push_back(remaining_block_indices);
                break;
            }

            // Only the blocks in the "intermediate" part of the logical KV cache will be considered for eviction
            auto scores_for_all_evictable_blocks = get_scores_for_all_evictable_blocks(decoder_layer_idx);
            auto evicted_block_indices = get_indices_of_blocks_to_evict(scores_for_all_evictable_blocks);
            m_num_evicted_tokens += evicted_block_indices.size() * m_eviction_config.block_size;

            // No longer need to track the overall "heavy-hitter" attention scores for freshly evicted blocks
            remove_scores_of_evicted_blocks(evicted_block_indices, decoder_layer_idx);

            auto remaining_block_indices = get_remaining_block_indices(evicted_block_indices);
            retval.push_back(remaining_block_indices);
        }
        return retval;
    }
private:
    void register_new_token_scores(const AttentionScoresForEachDecoderLayer& attention_scores_for_all_decoder_layers) {
        for (size_t decoder_layer_idx = 0; decoder_layer_idx < m_cache_counter.size(); decoder_layer_idx++) {

            auto attention_scores = attention_scores_for_all_decoder_layers[decoder_layer_idx];
            // "Start" tokens are never evicted, won't track scores for these
            // "Recent" tokens are also not evicted just yet, but need to accumulate their scores since they may
            // ultimately move into the "intermediate" eviction region of cache
            // Taking the [1, start_size:seq_len] span of the attention scores:
            auto attn_shape = attention_scores.get_shape();
            size_t kv_cache_size_in_tokens = attn_shape[0];
            auto hh_score = ov::Tensor(
                    attention_scores,
                    ov::Coordinate{m_eviction_config.start_size},
                    ov::Coordinate{kv_cache_size_in_tokens}
            );

            auto accumulated_scores_for_current_decoder_layer = m_scores[decoder_layer_idx];

            if (accumulated_scores_for_current_decoder_layer.empty()) {
                accumulated_scores_for_current_decoder_layer = std::vector<double>(hh_score.get_size());
                for (size_t idx = 0; idx < accumulated_scores_for_current_decoder_layer.size(); idx++) {
                    accumulated_scores_for_current_decoder_layer[idx] = hh_score.data<float>()[idx];
                }
                if (m_eviction_config.aggregation_mode == AggregationMode::NORM_SUM) {
                    // New sequence to track - will simulate that the tokens comprising the sequence were added one-by-one
                    // from the standpoint of the occurence tracker
                    std::size_t new_scores_size = hh_score.get_size();
                    std::vector<std::size_t> counter(m_eviction_config.get_total_cache_size());
                    std::generate(counter.begin(), counter.begin() + new_scores_size, [&new_scores_size]{ return new_scores_size--;});
                    m_cache_counter[decoder_layer_idx] = counter;
                }
            } else {
                size_t old_size_in_tokens = accumulated_scores_for_current_decoder_layer.size();
                size_t num_new_tokens = hh_score.get_size() - accumulated_scores_for_current_decoder_layer.size();
                if (m_eviction_config.aggregation_mode == AggregationMode::NORM_SUM) {
                    // Increment occurence counts of all currently tracked cache blocks
                    auto counter_for_current_decoder_layer = m_cache_counter[decoder_layer_idx];
                    for (auto it = counter_for_current_decoder_layer.begin(); it != counter_for_current_decoder_layer.end(); it++) {
                       *it += 1;
                    }
                    // Add occurence counts for new tokens like above
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

    std::vector<double> get_scores_for_all_evictable_blocks(size_t decoder_layer_idx) {
        auto accumulated_scores_for_current_decoder_layer = m_scores[decoder_layer_idx];
        auto counter_for_current_decoder_layer = m_cache_counter[decoder_layer_idx];

        // Make sure that there is at least one block that can be completely evicted
        size_t minimal_cache_size_in_tokens_after_start_area_for_eviction = m_eviction_config.recent_size + m_eviction_config.block_size;
        OPENVINO_ASSERT(counter_for_current_decoder_layer.size() > minimal_cache_size_in_tokens_after_start_area_for_eviction, "KV cache must be filled before scores for evictable blocks can be computed");

        // counter_for_current_decoder_layer does not consider the non-evictable "start" tokens, so its size is comprised of the
        // intermediate "evictable" part and the "recent" part
        std::size_t num_evictable_tokens = counter_for_current_decoder_layer.size() - m_eviction_config.recent_size;

        // Eviction is block-wise, so num_evictable_tokens is effectively always a multiple of block_size
        // Assuming that the start-intermediate-recent sizes of eviction regions are aligned by block_size
        auto num_evictable_blocks = static_cast<std::size_t>(num_evictable_tokens / m_eviction_config.block_size);

        std::vector<double> block_scores(num_evictable_blocks);
        for (size_t i = 0; i < num_evictable_blocks; ++i) {
            double normalized_accumulated_attn_score_for_block = 0.0;
            for (size_t j = 0; j < m_eviction_config.block_size; ++j) {
                size_t token_offset = m_eviction_config.block_size * i + j;
                if (m_eviction_config.aggregation_mode == AggregationMode::NORM_SUM) {
                    normalized_accumulated_attn_score_for_block += accumulated_scores_for_current_decoder_layer[token_offset] / counter_for_current_decoder_layer[token_offset];
                }
                else {
                    normalized_accumulated_attn_score_for_block += accumulated_scores_for_current_decoder_layer[token_offset];
                }
            }
            block_scores.push_back(normalized_accumulated_attn_score_for_block);
        }
        return block_scores;
    }

    std::vector<std::size_t> get_indices_of_blocks_to_evict(const std::vector<double>& scores_for_each_evictable_block) {
        // Returned indices are offsets of blocks to evict, taken from the beginning of the "intermediate", evictable
        // part of the logical KV cache. Indices are sorted in the ascending order.
        auto num_evictable_blocks_to_keep = m_eviction_config.get_num_blocks(m_eviction_config.evictable_size);
        auto current_num_evictable_blocks = scores_for_each_evictable_block.size();
        OPENVINO_ASSERT(current_num_evictable_blocks >= num_evictable_blocks_to_keep);
        if (current_num_evictable_blocks == num_evictable_blocks_to_keep) {
            return {}; // nothing to evict yet
        }

        auto num_blocks_to_evict = current_num_evictable_blocks - num_evictable_blocks_to_keep;

        std::vector<std::pair<double, std::size_t>> evictable_block_score_and_index_pairs;
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

    void remove_scores_of_evicted_blocks(const std::vector<std::size_t>& evicted_block_indices, size_t decoder_layer_idx) {
        if (evicted_block_indices.empty()) {
            return;
        }

        auto accumulated_scores_for_current_decoder_layer = m_scores[decoder_layer_idx];
        auto counter_for_current_decoder_layer = m_cache_counter[decoder_layer_idx];
        OPENVINO_ASSERT(accumulated_scores_for_current_decoder_layer.size() == counter_for_current_decoder_layer.size());
        auto old_size = accumulated_scores_for_current_decoder_layer.size();

        auto new_size = accumulated_scores_for_current_decoder_layer.size() - evicted_block_indices.size() * m_eviction_config.block_size;

        std::vector<double> new_scores;
        new_scores.reserve(new_size);

        std::vector<size_t> new_counter;
        new_counter.reserve(new_size);

        for (size_t token_idx = 0, evicted_block_idx = 0; token_idx < old_size; token_idx++) {
            if (evicted_block_idx < evicted_block_indices.size() && token_idx == evicted_block_indices[evicted_block_idx] * m_eviction_config.block_size) {
                ++evicted_block_idx;
                token_idx += m_eviction_config.block_size;
                continue;
            }
            new_scores.push_back(accumulated_scores_for_current_decoder_layer[token_idx]);
            new_counter.push_back(counter_for_current_decoder_layer[token_idx]);
            ++token_idx;
        }

        m_scores[decoder_layer_idx] = new_scores;
        m_cache_counter[decoder_layer_idx] = new_counter;
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

    CacheEvictionConfig m_eviction_config;
    std::vector<std::vector<double>> m_scores;
    std::size_t m_num_evicted_tokens = 0;
    using LogicalBlockIdx = size_t;
    std::size_t m_num_decoder_layers;
    std::vector<std::vector<size_t>> m_cache_counter;
};

#endif //OPENVINOGENAI_CACHE_EVICTION_HPP
