// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "prefix_digest_registry.hpp"

namespace ov::genai {

namespace {

std::atomic<std::size_t> g_num_leases{0};

}  // namespace

PrefixDigestRegistry& PrefixDigestRegistry::instance() {
    static PrefixDigestRegistry registry;
    return registry;
}

bool PrefixDigestRegistry::enabled() noexcept {
    return g_num_leases.load(std::memory_order_relaxed) > 0;
}

PrefixDigestRegistry::Lease::Lease() {
    g_num_leases.fetch_add(1, std::memory_order_relaxed);
}

PrefixDigestRegistry::Lease::~Lease() {
    g_num_leases.fetch_sub(1, std::memory_order_relaxed);
}

void PrefixDigestRegistry::record(std::size_t hash, const ContentHash& digest) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_current.size() >= kGenerationSize && m_current.find(hash) == m_current.end()) {
        m_previous = std::move(m_current);
        m_current.clear();
    }
    m_current[hash] = digest;
}

std::optional<ContentHash> PrefixDigestRegistry::find(std::size_t hash) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (const auto it = m_current.find(hash); it != m_current.end()) {
        return it->second;
    }
    if (const auto it = m_previous.find(hash); it != m_previous.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::vector<std::optional<ContentHash>> PrefixDigestRegistry::find_many(const std::vector<std::size_t>& hashes) const {
    std::vector<std::optional<ContentHash>> digests;
    digests.reserve(hashes.size());
    std::lock_guard<std::mutex> lock(m_mutex);
    for (const std::size_t hash : hashes) {
        if (const auto it = m_current.find(hash); it != m_current.end()) {
            digests.push_back(it->second);
        } else if (const auto previous = m_previous.find(hash); previous != m_previous.end()) {
            digests.push_back(previous->second);
        } else {
            digests.push_back(std::nullopt);
        }
    }
    return digests;
}

}  // namespace ov::genai
