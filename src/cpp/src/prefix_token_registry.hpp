// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ov::genai {

/**
 * @brief Remembers the tokens behind each prefix hash that a Sequence computed over blocks of a wanted size.
 *
 * A store that keeps the linear-attention checkpoints next to the KV blocks wants to know which tokens a checkpoint
 * covers, but by the time a checkpoint reaches the store its sequence is gone and only the hash is left. While a
 * consumer holds a Lease for a block size, Sequence::get_hash() records the whole prefix [0, content_length) of every
 * hash it computes with that block size, keyed by the hash. With no Lease nothing is copied.
 *
 * A lookup that misses (the entry was evicted, or the hash did not come from a Sequence of token ids) means the
 * tokens are not known.
 */
class PrefixTokenRegistry {
public:
    using TokenIds = std::shared_ptr<const std::vector<std::int64_t>>;

    static PrefixTokenRegistry& instance() {
        static PrefixTokenRegistry registry;
        return registry;
    }

    /// @return Whether a Lease for @p block_size is alive, i.e. whether Sequence has to record the tokens.
    static bool wants(std::size_t block_size) noexcept {
        State& state = shared_state();
        if (state.num_leases.load(std::memory_order_relaxed) == 0) {
            return false;
        }
        std::lock_guard<std::mutex> lock(state.mutex);
        return std::find(state.block_sizes.begin(), state.block_sizes.end(), block_size) != state.block_sizes.end();
    }

    /// Keeps token recording on for one block size while it is alive.
    class Lease {
    public:
        explicit Lease(std::size_t block_size) : m_block_size(block_size) {
            State& state = shared_state();
            std::lock_guard<std::mutex> lock(state.mutex);
            state.block_sizes.push_back(block_size);
            state.num_leases.fetch_add(1, std::memory_order_relaxed);
        }

        ~Lease() {
            State& state = shared_state();
            std::lock_guard<std::mutex> lock(state.mutex);
            const auto it = std::find(state.block_sizes.begin(), state.block_sizes.end(), m_block_size);
            if (it != state.block_sizes.end()) {
                state.block_sizes.erase(it);
            }
            state.num_leases.fetch_sub(1, std::memory_order_relaxed);
        }

        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

    private:
        std::size_t m_block_size;
    };

    void record(std::size_t hash, TokenIds token_ids) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_current.size() >= kGenerationSize && m_current.find(hash) == m_current.end()) {
            m_previous = std::move(m_current);
            m_current.clear();
        }
        m_current[hash] = std::move(token_ids);
    }

    /// @return The tokens recorded for @p hash, or null.
    TokenIds find(std::size_t hash) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (const auto it = m_current.find(hash); it != m_current.end()) {
            return it->second;
        }
        if (const auto it = m_previous.find(hash); it != m_previous.end()) {
            return it->second;
        }
        return nullptr;
    }

private:
    // An entry is a whole prefix, so the generations are small: when the current one is full it becomes the previous
    // one and the older previous one is dropped.
    static constexpr std::size_t kGenerationSize = 64;

    struct State {
        std::atomic<std::size_t> num_leases{0};
        std::mutex mutex;
        std::vector<std::size_t> block_sizes;
    };

    static State& shared_state() {
        static State state;
        return state;
    }

    mutable std::mutex m_mutex;
    std::unordered_map<std::size_t, TokenIds> m_current;
    std::unordered_map<std::size_t, TokenIds> m_previous;
};

}  // namespace ov::genai
