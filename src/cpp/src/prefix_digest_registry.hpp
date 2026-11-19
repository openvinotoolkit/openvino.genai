// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "content_hash.hpp"

namespace ov::genai {

/**
 * @brief Remembers the 128-bit digest behind each prefix hash that a Sequence computed.
 *
 * The prefix cache itself keys blocks by a size_t hash. A persistent store that outlives the process wants a wider,
 * reproducible identity, so while a digest-keyed consumer holds a Lease, Sequence::get_hash() also computes the
 * digest chain and records it here, keyed by the size_t hash it returned. With no Lease nothing is computed or
 * recorded, so the prefix cache is unaffected.
 *
 * A lookup that misses (the entry was evicted, or the hash did not come from a Sequence) means no digest is known.
 */
class PrefixDigestRegistry {
public:
    static PrefixDigestRegistry& instance();

    /// @return Whether a Lease is alive, i.e. whether Sequence has to record digests.
    static bool enabled() noexcept;

    /// Keeps digest recording on while it is alive.
    class Lease {
    public:
        Lease();
        ~Lease();
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
    };

    void record(std::size_t hash, const ContentHash& digest);

    std::optional<ContentHash> find(std::size_t hash) const;

    /// One lock for the whole list; entry i answers for hashes[i].
    std::vector<std::optional<ContentHash>> find_many(const std::vector<std::size_t>& hashes) const;

private:
    // Two generations bound the memory: when the current one is full it becomes the previous one and the older
    // previous one is dropped, so an entry survives at least one generation's worth of newer recordings.
    static constexpr std::size_t kGenerationSize = std::size_t{1} << 17;

    mutable std::mutex m_mutex;
    std::unordered_map<std::size_t, ContentHash> m_current;
    std::unordered_map<std::size_t, ContentHash> m_previous;
};

}  // namespace ov::genai
