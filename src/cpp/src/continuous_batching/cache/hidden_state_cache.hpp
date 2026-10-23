// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

/**
 * @brief Intermediate hidden states of full KV blocks, keyed by the prefix hash at the block end.
 *
 * The KV prefix cache does not recompute a restored prefix, so its hidden states would be missing.
 * The pipeline restores KV only up to the prefix covered here and replays these rows for it. The
 * prefix hash identifies all tokens before the block end, so an entry stays valid whichever physical
 * block holds the KV, and the two caches can evict independently.
 */
class HiddenStateCache {
public:
    static constexpr size_t kDefaultBudgetBytes = 256 * 1024 * 1024;

    explicit HiddenStateCache(size_t budget_bytes = kDefaultBudgetBytes) : m_budget_bytes(budget_bytes) {}

    /// @param rows One tensor per token of the block, in token order. The cache stores copies.
    void put(uint64_t hash, const std::vector<ov::Tensor>& rows) {
        size_t bytes = 0;
        for (const ov::Tensor& row : rows) {
            bytes += row.get_byte_size();
        }
        std::lock_guard<std::mutex> lock{m_mutex};
        const auto it = m_entries.find(hash);
        if (it != m_entries.end()) {
            m_lru.splice(m_lru.begin(), m_lru, it->second.lru_it);
            return;
        }
        if (bytes > m_budget_bytes) {
            return;
        }
        std::vector<ov::Tensor> copies;
        copies.reserve(rows.size());
        for (const ov::Tensor& row : rows) {
            copies.push_back(copy_row(row));
        }
        m_lru.push_front(hash);
        m_entries.emplace(hash, Entry{std::move(copies), bytes, m_lru.begin()});
        m_used_bytes += bytes;
        while (m_used_bytes > m_budget_bytes) {
            const auto victim = m_entries.find(m_lru.back());
            m_used_bytes -= victim->second.bytes;
            m_entries.erase(victim);
            m_lru.pop_back();
        }
    }

    /// @return Copies of the rows of the longest leading run of block_hashes that are all cached, in token order.
    std::vector<ov::Tensor> get_prefix(const std::vector<uint64_t>& block_hashes) {
        std::vector<ov::Tensor> rows;
        std::lock_guard<std::mutex> lock{m_mutex};
        for (const uint64_t hash : block_hashes) {
            const auto it = m_entries.find(hash);
            if (it == m_entries.end()) {
                break;
            }
            m_lru.splice(m_lru.begin(), m_lru, it->second.lru_it);
            for (const ov::Tensor& row : it->second.rows) {
                rows.push_back(copy_row(row));
            }
        }
        return rows;
    }

    size_t used_bytes() const {
        std::lock_guard<std::mutex> lock{m_mutex};
        return m_used_bytes;
    }

private:
    // Rows reach callers through public results, so a write there must not change what later requests replay.
    static ov::Tensor copy_row(const ov::Tensor& row) {
        ov::Tensor copy(row.get_element_type(), row.get_shape());
        row.copy_to(copy);
        return copy;
    }

    struct Entry {
        std::vector<ov::Tensor> rows;
        size_t bytes;
        std::list<uint64_t>::iterator lru_it;
    };

    const size_t m_budget_bytes;
    size_t m_used_bytes = 0;
    // Front is the most recently used hash.
    std::list<uint64_t> m_lru;
    std::unordered_map<uint64_t, Entry> m_entries;
    mutable std::mutex m_mutex;
};

}  // namespace ov::genai
