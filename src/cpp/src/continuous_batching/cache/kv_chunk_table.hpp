// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "openvino/core/except.hpp"

namespace ov::genai::kv_chunk_table {

// Layout of the i64 tensor bound to a PagedAttention key_cache.N / value_cache.N port in chunk-table mode. It must stay
// identical to the GPU plugin's decoder (src/plugins/intel_gpu/include/intel_gpu/runtime/kv_chunk_table.hpp).
// Granule = 2^granule_shift blocks; every chunk holds a whole number of granules and block ids are dense across chunks
// in allocation order.
//   word 0                    : granule_shift (bits 0..7) | num_chunks (bits 8..31) | generation (bits 32..63)
//   word 1                    : num_granules
//   words [2, 2 + num_chunks) : chunk base pointers (this layer, K or V)
//   then, num_granules u32    : (chunk_id << 20) | granule_in_chunk, two entries per word, low half first
constexpr uint32_t kGranuleInChunkBits = 20;
constexpr size_t kMaxChunks = size_t{1} << 12;
constexpr size_t kMaxGranulesPerChunk = size_t{1} << kGranuleInChunkBits;
constexpr size_t kHeaderWords = 2;

inline std::vector<int64_t> build(uint32_t granule_shift,
                                  uint32_t generation,
                                  const std::vector<int64_t>& chunk_base_ptrs,
                                  const std::vector<size_t>& chunk_blocks) {
    OPENVINO_ASSERT(chunk_base_ptrs.size() == chunk_blocks.size(), "Chunk pointer and size lists must match");
    OPENVINO_ASSERT(chunk_base_ptrs.size() <= kMaxChunks, "KV cache needs more than ", kMaxChunks,
                    " chunks; increase kv_cache_chunk_size_blocks");
    const size_t granule_blocks = size_t{1} << granule_shift;
    std::vector<uint32_t> directory;
    for (size_t chunk = 0; chunk < chunk_blocks.size(); ++chunk) {
        OPENVINO_ASSERT(chunk_blocks[chunk] != 0 && chunk_blocks[chunk] % granule_blocks == 0 &&
                            chunk_blocks[chunk] / granule_blocks <= kMaxGranulesPerChunk,
                        "KV cache chunk size must be a non-zero multiple of the granule size and fit the directory");
        for (size_t granule = 0; granule < chunk_blocks[chunk] / granule_blocks; ++granule) {
            directory.push_back(static_cast<uint32_t>((chunk << kGranuleInChunkBits) | granule));
        }
    }
    const size_t num_chunks = chunk_base_ptrs.size();
    std::vector<int64_t> table(kHeaderWords + num_chunks + (directory.size() + 1) / 2, 0);
    table[0] = static_cast<int64_t>(static_cast<uint64_t>(granule_shift) | (static_cast<uint64_t>(num_chunks) << 8) |
                                    (static_cast<uint64_t>(generation) << 32));
    table[1] = static_cast<int64_t>(directory.size());
    std::copy(chunk_base_ptrs.begin(), chunk_base_ptrs.end(), table.begin() + kHeaderWords);
    std::memcpy(table.data() + kHeaderWords + num_chunks, directory.data(), directory.size() * sizeof(uint32_t));
    return table;
}

}  // namespace ov::genai::kv_chunk_table
