

#pragma once

#include <vector>

#include <openvino/runtime/tensor.hpp>

#include "model_config.hpp"

class CacheManager {
    std::vector<ov::Tensor> m_key_cache;
    std::vector<ov::Tensor> m_value_cache;

public:
    CacheManager() {
        m_key_cache.reserve(NUM_DECODER_LAYERS);
        m_value_cache.reserve(NUM_DECODER_LAYERS);

        // Allocate KV caches
        const ov::Shape k_cache_shape{NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE};
        const ov::Shape v_cache_shape{NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE};

        for (size_t decoder_layer_id = 0; decoder_layer_id < NUM_DECODER_LAYERS; ++decoder_layer_id) {
            ov::Tensor key_cache(kv_cache_precision, k_cache_shape);
            ov::Tensor value_cache(kv_cache_precision, v_cache_shape);

            m_key_cache.emplace_back(key_cache);
            m_value_cache.emplace_back(value_cache);
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

    void copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map) {
        for (const auto & blocks_pair : block_copy_map) {
            size_t src_block_id = blocks_pair.first;
            const std::list<size_t>& dst_block_ids = blocks_pair.second;

            ov::Coordinate src_start_roi = { src_block_id, 0, 0, 0 };
            ov::Coordinate src_end_roi = { src_block_id + 1, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE };

            for (size_t dst_block_id : dst_block_ids) {
                ov::Coordinate dst_start_roi = { dst_block_id, 0, 0, 0 };
                ov::Coordinate dst_end_roi = { dst_block_id + 1, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE };

                for (size_t decoder_layer_id = 0; decoder_layer_id < NUM_DECODER_LAYERS; ++decoder_layer_id) {
                    ov::Tensor k_src_cache_roi(m_key_cache[decoder_layer_id], src_start_roi, src_end_roi);
                    ov::Tensor k_dst_cache_roi(m_key_cache[decoder_layer_id], dst_start_roi, dst_end_roi);

                    ov::Tensor v_src_cache_roi(m_value_cache[decoder_layer_id], src_start_roi, src_end_roi);
                    ov::Tensor v_dst_cache_roi(m_value_cache[decoder_layer_id], dst_start_roi, dst_end_roi);

                    k_src_cache_roi.copy_to(k_dst_cache_roi);
                    v_src_cache_roi.copy_to(v_dst_cache_roi);
                }
            }
        }
    }
};
