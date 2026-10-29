// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

/**
 * @brief Keeps the intermediate hidden states of prompt prefixes, so that prefix caching works for Omni models.
 *
 * The Omni talker reads one intermediate hidden state per token position. When the KV prefix cache restores a
 * prefix, the model does not run on those tokens, so it does not produce their hidden states. This cache stores
 * them instead.
 *
 * An entry covers one full KV block: one hidden state tensor for each token of the block, in token order. The
 * key is the prefix hash at the end of the block, the same hash the KV prefix cache uses. This hash depends on
 * all tokens before the block end, so an entry stays valid whichever physical KV block holds the KV.
 *
 * On restore, the pipeline restores KV only for the leading blocks that have an entry here, and copies their
 * hidden states into the sequence. The two caches evict independently: a block that lost its entry here is
 * computed again.
 */
class HiddenStateCache {
public:
    /// @param capacity_tokens Returns how many tokens the cache may hold. The pipeline passes the KV cache token
    /// capacity, which grows with dynamic cache allocation.
    explicit HiddenStateCache(std::function<size_t()> capacity_tokens)
        : m_capacity_tokens(std::move(capacity_tokens)) {}

    /// @param hidden_states One tensor per token of the block, in token order. The cache stores copies.
    void put(uint64_t hash, const std::vector<ov::Tensor>& hidden_states) {
        std::lock_guard<std::mutex> lock{m_mutex};
        const auto it = m_entries.find(hash);
        if (it != m_entries.end()) {
            m_lru.splice(m_lru.begin(), m_lru, it->second.lru_it);
            return;
        }
        const size_t capacity = m_capacity_tokens();
        if (hidden_states.size() > capacity) {
            return;
        }
        std::vector<ov::Tensor> copies;
        copies.reserve(hidden_states.size());
        for (const ov::Tensor& hidden_state : hidden_states) {
            copies.push_back(copy_tensor(hidden_state));
        }
        m_lru.push_front(hash);
        m_entries.emplace(hash, Entry{std::move(copies), m_lru.begin()});
        m_used_tokens += hidden_states.size();
        while (m_used_tokens > capacity) {
            const auto victim = m_entries.find(m_lru.back());
            m_used_tokens -= victim->second.hidden_states.size();
            m_entries.erase(victim);
            m_lru.pop_back();
        }
    }

    /// @return Copies of the hidden states of the longest leading run of block_hashes that are all cached, in
    /// token order.
    std::vector<ov::Tensor> get_prefix(const std::vector<uint64_t>& block_hashes) {
        std::vector<ov::Tensor> hidden_states;
        std::lock_guard<std::mutex> lock{m_mutex};
        for (const uint64_t hash : block_hashes) {
            const auto it = m_entries.find(hash);
            if (it == m_entries.end()) {
                break;
            }
            m_lru.splice(m_lru.begin(), m_lru, it->second.lru_it);
            for (const ov::Tensor& hidden_state : it->second.hidden_states) {
                hidden_states.push_back(copy_tensor(hidden_state));
            }
        }
        return hidden_states;
    }

    size_t used_tokens() const {
        std::lock_guard<std::mutex> lock{m_mutex};
        return m_used_tokens;
    }

private:
    // Hidden states reach callers through public results, so a write there must not change what later
    // requests replay.
    static ov::Tensor copy_tensor(const ov::Tensor& tensor) {
        ov::Tensor copy(tensor.get_element_type(), tensor.get_shape());
        tensor.copy_to(copy);
        return copy;
    }

    struct Entry {
        std::vector<ov::Tensor> hidden_states;
        std::list<uint64_t>::iterator lru_it;
    };

    const std::function<size_t()> m_capacity_tokens;
    size_t m_used_tokens = 0;
    // Front is the most recently used hash.
    std::list<uint64_t> m_lru;
    std::unordered_map<uint64_t, Entry> m_entries;
    mutable std::mutex m_mutex;
};

}  // namespace ov::genai
