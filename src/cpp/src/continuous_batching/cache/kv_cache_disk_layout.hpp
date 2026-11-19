// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <limits>
#include <vector>

#include "openvino/core/except.hpp"

namespace ov::genai {

class KVCacheDiskLayout {
public:
    struct Segment {
        size_t offset;
        size_t size;
    };

    KVCacheDiskLayout(const std::vector<size_t>& key_block_sizes,
                      const std::vector<size_t>& value_block_sizes)
        : m_key_segments(key_block_sizes.size()),
          m_value_segments(value_block_sizes.size()),
          m_slot_size(0) {
        OPENVINO_ASSERT(key_block_sizes.size() == value_block_sizes.size(),
                        "Key and value cache layer counts must match");
        for (size_t layer = 0; layer < key_block_sizes.size(); ++layer) {
            m_key_segments[layer] = {m_slot_size, key_block_sizes[layer]};
            m_slot_size += key_block_sizes[layer];
            m_value_segments[layer] = {m_slot_size, value_block_sizes[layer]};
            m_slot_size += value_block_sizes[layer];
        }
    }

    size_t get_slot_size() const {
        return m_slot_size;
    }

    size_t get_num_layers() const {
        return m_key_segments.size();
    }

    Segment get_key_segment(size_t layer) const {
        return m_key_segments.at(layer);
    }

    Segment get_value_segment(size_t layer) const {
        return m_value_segments.at(layer);
    }

    size_t get_slot_offset(size_t slot_id) const {
        OPENVINO_ASSERT(m_slot_size == 0 || slot_id <= (std::numeric_limits<size_t>::max() / m_slot_size),
                        "Disk slot offset overflows size_t");
        return slot_id * m_slot_size;
    }

private:
    std::vector<Segment> m_key_segments;
    std::vector<Segment> m_value_segments;
    size_t m_slot_size;
};

}  // namespace ov::genai
