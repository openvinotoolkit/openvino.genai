

#pragma once

#include <vector>

#include <openvino/runtime/tensor.hpp>

#include "model_config.hpp"

class CacheManager {
    std::vector<ov::Tensor> m_key_cache;
    std::vector<ov::Tensor> m_value_cache;

public:
    CacheManager() {
        // Allocate KV caches
        const ov::Shape k_cache_shape{NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE / X, BLOCK_SIZE, X};
        const ov::Shape v_cache_shape{NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE};

        for (size_t decoder_layer_id = 0; decoder_layer_id < NUM_DECODER_LAYERS; ++decoder_layer_id) {
            m_key_cache[decoder_layer_id] = ov::Tensor(kv_cache_precision, k_cache_shape);
            m_value_cache[decoder_layer_id] = ov::Tensor(kv_cache_precision, v_cache_shape);
        }
    }

    size_t get_num_layers() const {
        return m_key_cache.size();
    }

    ov::Tensor get_key_cache(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_key_cache.size());
        return m_key_cache[decoder_layer_id];
    }

    ov::Tensor get_value_cache(size_t decoder_layer_id) const {
        OPENVINO_ASSERT(decoder_layer_id < m_value_cache.size());
        return m_value_cache[decoder_layer_id];
    }

    void copy_blocks(const std::map<size_t, size_t>& block_copy_map) {
        for (const auto & blocks_pair : block_copy_map) {
            size_t src_block_id = blocks_pair.first, dst_block_id = blocks_pair.second;

            ov::Coordinate k_src_start_roi = { src_block_id, NUM_KV_HEADS, HEAD_SIZE / X, BLOCK_SIZE, X };
            ov::Coordinate k_src_end_roi = { src_block_id + 1, NUM_KV_HEADS, HEAD_SIZE / X, BLOCK_SIZE, X };
            ov::Coordinate k_dst_start_roi = { dst_block_id, NUM_KV_HEADS, HEAD_SIZE / X, BLOCK_SIZE, X };
            ov::Coordinate k_dst_end_roi = { dst_block_id + 1, NUM_KV_HEADS, HEAD_SIZE / X, BLOCK_SIZE, X };
            
            ov::Coordinate v_src_start_roi = { src_block_id, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE };
            ov::Coordinate v_src_end_roi = { src_block_id + 1, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE };
            ov::Coordinate v_dst_start_roi = { dst_block_id, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE };
            ov::Coordinate v_dst_end_roi = { dst_block_id + 1, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE };

            for (size_t decoder_layer_id = 0; decoder_layer_id < NUM_DECODER_LAYERS; ++decoder_layer_id) {
                ov::Tensor k_src_cache_roi(m_key_cache[decoder_layer_id], k_src_start_roi, k_src_end_roi);
                ov::Tensor k_dst_cache_roi(m_key_cache[decoder_layer_id], k_dst_start_roi, k_dst_end_roi);

                ov::Tensor v_src_cache_roi(m_value_cache[decoder_layer_id], v_src_start_roi, v_src_end_roi);
                ov::Tensor v_dst_cache_roi(m_value_cache[decoder_layer_id], v_dst_start_roi, v_dst_end_roi);

                k_src_cache_roi.copy_to(k_dst_cache_roi);
                v_src_cache_roi.copy_to(v_dst_cache_roi);
            }
        }
    }
};
