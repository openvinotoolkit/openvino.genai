// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai::kv_chunk_table {

// The global KV chunk table: PagedAttentionExtension input `kv_chunk_table`, an i32 tensor of 2 * total_slots elements
// shared by all layers. It must stay identical to the GPU plugin's decoder
// (src/plugins/intel_gpu/include/intel_gpu/runtime/kv_chunk_table.hpp); see split_kv_cache_design-3.md.
// Every u64 slot is stored as two i32 elements, low half first:
//   slots [0, 16)                    header (magic, L, C_max, live chunks, granule shift, live granules, D_max, ...)
//   slots [16, 16 + ceil(D_max/2))   directory: D_max u32 entries, dir[k] = (chunk_id << 20) | granule_in_chunk
//   then, per layer l, 2 * C_max slots: K chunk pointers [0, C_max) followed by V chunk pointers [0, C_max)
constexpr uint64_t kMagic = 0x50414348554E4B33ull;  // "PACHUNK3"
constexpr uint32_t kGranuleInChunkBits = 20;
constexpr size_t kMaxChunkId = (size_t{1} << 12) - 1;
constexpr size_t kMaxGranulesPerChunk = size_t{1} << kGranuleInChunkBits;
constexpr size_t kHeaderSlots = 16;

enum HeaderSlot : size_t {
    MAGIC = 0,
    NUM_LAYERS = 1,
    MAX_CHUNKS = 2,
    LIVE_CHUNKS = 3,
    GRANULE_SHIFT = 4,
    LIVE_GRANULES = 5,
    MAX_GRANULES = 6,
    GENERATION = 8,
    FLAGS = 9,
};

inline size_t bases_offset(size_t max_granules) {
    return kHeaderSlots + (max_granules + 1) / 2;
}

inline size_t total_slots(size_t num_layers, size_t max_chunks, size_t max_granules) {
    return bases_offset(max_granules) + num_layers * 2 * max_chunks;
}

// Owns the in-place updates of the table tensor. The tensor is the inference request's own USM host input tensor
// (obtained with get_tensor() and sized once): it is bound once and only ever updated in place, so growth never rebinds
// anything and a plain host tensor (copied to the device on every inference) is avoided.
class GlobalChunkTable {
public:
    GlobalChunkTable() = default;

    // `table` must have room for 2 * total_slots(...) i32 elements.
    GlobalChunkTable(ov::Tensor table, size_t num_layers, size_t max_chunks, size_t max_granules, uint32_t granule_shift)
        : m_table(std::move(table)),
          m_num_layers(num_layers),
          m_max_chunks(max_chunks),
          m_max_granules(max_granules),
          m_granule_shift(granule_shift) {
        OPENVINO_ASSERT(max_chunks >= 1 && max_chunks <= kMaxChunkId + 1, "KV chunk table: invalid max chunk count ", max_chunks);
        OPENVINO_ASSERT(m_table.get_element_type() == ov::element::i32 &&
                            m_table.get_size() == 2 * total_slots(num_layers, max_chunks, max_granules),
                        "KV chunk table tensor has an unexpected type or size");
        clear();
        // Static fields never change after this point.
        set_slot(NUM_LAYERS, m_num_layers);
        set_slot(MAX_CHUNKS, m_max_chunks);
        set_slot(GRANULE_SHIFT, m_granule_shift);
        set_slot(MAX_GRANULES, m_max_granules);
        set_slot(MAGIC, kMagic);
    }

    static size_t tensor_elements(size_t num_layers, size_t max_chunks, size_t max_granules) {
        return 2 * total_slots(num_layers, max_chunks, max_granules);
    }

    size_t live_chunks() const {
        return get_slot(LIVE_CHUNKS);
    }

    size_t live_granules() const {
        return get_slot(LIVE_GRANULES);
    }

    // Appends one chunk of `chunk_blocks` blocks (a non-zero multiple of the granule size); `key_ptrs`/`value_ptrs`
    // hold the chunk base pointer of every layer. Existing entries are not touched.
    void append_chunk(size_t chunk_blocks, const std::vector<uint64_t>& key_ptrs, const std::vector<uint64_t>& value_ptrs) {
        const size_t granule_blocks = size_t{1} << m_granule_shift;
        const size_t chunk_id = live_chunks();
        const size_t first_granule = live_granules();
        const size_t granules = chunk_blocks / granule_blocks;
        OPENVINO_ASSERT(key_ptrs.size() == m_num_layers && value_ptrs.size() == m_num_layers,
                        "KV chunk table: expected one chunk pointer per layer");
        OPENVINO_ASSERT(chunk_blocks != 0 && chunk_blocks % granule_blocks == 0 && granules <= kMaxGranulesPerChunk,
                        "KV cache chunk size must be a non-zero multiple of the granule size and fit the directory");
        OPENVINO_ASSERT(chunk_id < m_max_chunks,
                        "KV cache needs more than ", m_max_chunks, " chunks; the chunk table capacity is exhausted");
        OPENVINO_ASSERT(first_granule + granules <= m_max_granules,
                        "KV cache needs more than ", m_max_granules, " granules of ", granule_blocks,
                        " blocks; the chunk table directory capacity is exhausted");
        for (size_t granule = 0; granule < granules; ++granule) {
            const uint32_t entry = static_cast<uint32_t>((chunk_id << kGranuleInChunkBits) | granule);
            std::memcpy(directory_ptr() + (first_granule + granule) * sizeof(uint32_t), &entry, sizeof(entry));
        }
        for (size_t layer = 0; layer < m_num_layers; ++layer) {
            const size_t base = bases_offset(m_max_granules) + layer * 2 * m_max_chunks;
            set_slot(base + chunk_id, key_ptrs[layer]);
            set_slot(base + m_max_chunks + chunk_id, value_ptrs[layer]);
        }
        set_slot(LIVE_GRANULES, first_granule + granules);
        set_slot(LIVE_CHUNKS, chunk_id + 1);
        set_slot(GENERATION, get_slot(GENERATION) + 1);
    }

    // Drops all chunks; the directory and pointers of dropped chunks are zeroed so a stale read is easy to spot.
    void clear() {
        if (!m_table) {
            return;
        }
        std::memset(m_table.data(), 0, m_table.get_byte_size());
        // Keep the static header valid even between clear() and the next append.
        if (m_num_layers != 0) {
            set_slot(NUM_LAYERS, m_num_layers);
            set_slot(MAX_CHUNKS, m_max_chunks);
            set_slot(GRANULE_SHIFT, m_granule_shift);
            set_slot(MAX_GRANULES, m_max_granules);
            set_slot(MAGIC, kMagic);
        }
        set_slot(GENERATION, ++m_generation);
    }

private:
    uint8_t* directory_ptr() {
        return static_cast<uint8_t*>(m_table.data()) + kHeaderSlots * sizeof(uint64_t);
    }

    uint64_t get_slot(size_t slot) const {
        uint64_t value = 0;
        std::memcpy(&value, static_cast<const uint8_t*>(const_cast<ov::Tensor&>(m_table).data()) + slot * sizeof(uint64_t), sizeof(value));
        return value;
    }

    void set_slot(size_t slot, uint64_t value) {
        std::memcpy(static_cast<uint8_t*>(m_table.data()) + slot * sizeof(uint64_t), &value, sizeof(value));
    }

    ov::Tensor m_table;
    size_t m_num_layers = 0;
    size_t m_max_chunks = 0;
    size_t m_max_granules = 0;
    uint32_t m_granule_shift = 0;
    uint64_t m_generation = 0;
};

}  // namespace ov::genai::kv_chunk_table
