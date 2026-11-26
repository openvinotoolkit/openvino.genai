// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"

#include "kv_cache_store.hpp"

namespace ov::genai {

/**
 * @brief Composes several key-addressed stores into one, fastest first (for example DRAM, then SSD).
 *
 * The tiers see the same writes, so each of them evicts by its own policy and the smaller, faster ones simply hold
 * fewer of the blocks. A read takes a block from the first tier that has it and copies it into the faster tiers that
 * lost it, so the next restore of the prefix is served from the fast tier. A probe reports a block when any tier has it.
 * All tiers must use the same block size.
 */
class TieredKVCacheStore final : public IKVCacheStore {
public:
    explicit TieredKVCacheStore(std::vector<std::unique_ptr<IKVCacheStore>> tiers) : m_tiers(std::move(tiers)) {
        OPENVINO_ASSERT(!m_tiers.empty(), "A tiered KV cache store needs at least one tier");
        for (const auto& tier : m_tiers) {
            OPENVINO_ASSERT(tier->block_bytes() == m_tiers.front()->block_bytes(),
                            "All tiers of a tiered KV cache store must use the same block size");
        }
    }

    std::vector<bool> has_blocks(const std::vector<PrefixKey>& chain) const override {
        std::vector<bool> present(chain.size(), false);
        for (const auto& tier : m_tiers) {
            const std::vector<bool> in_tier = tier->has_blocks(chain);
            for (size_t i = 0; i < chain.size(); ++i) {
                present[i] = present[i] || in_tier[i];
            }
        }
        return present;
    }

    std::vector<bool> read_blocks(const std::vector<PrefixKey>& keys,
                                  std::vector<std::vector<uint8_t>>& data) const override {
        std::vector<bool> found(keys.size(), false);
        data.assign(keys.size(), {});
        for (size_t tier_index = 0; tier_index < m_tiers.size(); ++tier_index) {
            std::vector<PrefixKey> missing_keys;
            std::vector<size_t> missing_positions;
            for (size_t i = 0; i < keys.size(); ++i) {
                if (!found[i]) {
                    missing_keys.push_back(keys[i]);
                    missing_positions.push_back(i);
                }
            }
            if (missing_keys.empty()) {
                break;
            }
            std::vector<std::vector<uint8_t>> tier_data;
            const std::vector<bool> in_tier = m_tiers[tier_index]->read_blocks(missing_keys, tier_data);
            for (size_t i = 0; i < missing_keys.size(); ++i) {
                if (in_tier[i]) {
                    found[missing_positions[i]] = true;
                    data[missing_positions[i]] = std::move(tier_data[i]);
                }
            }
            promote(tier_index, keys, found, data, missing_positions, in_tier);
        }
        return found;
    }

    std::vector<bool> write_blocks(const std::vector<PrefixKey>& keys,
                                   const std::vector<const std::vector<uint8_t>*>& data) override {
        std::vector<bool> written(keys.size(), false);
        for (auto& tier : m_tiers) {
            const std::vector<bool> in_tier = tier->write_blocks(keys, data);
            for (size_t i = 0; i < keys.size(); ++i) {
                written[i] = written[i] || in_tier[i];
            }
        }
        return written;
    }

    void invalidate(const PrefixKey& key) override {
        for (auto& tier : m_tiers) {
            tier->invalidate(key);
        }
    }

    size_t block_bytes() const override {
        return m_tiers.front()->block_bytes();
    }

    /// Every tier sees every write, so the largest tier bounds how many blocks the whole holds.
    size_t capacity_blocks_hint() const override {
        size_t largest = 0;
        for (const auto& tier : m_tiers) {
            largest = std::max(largest, tier->capacity_blocks_hint());
        }
        return largest;
    }

    bool is_persistent() const override {
        return std::any_of(m_tiers.begin(), m_tiers.end(), [](const auto& tier) {
            return tier->is_persistent();
        });
    }

    bool wants_chain_context() const override {
        return std::any_of(m_tiers.begin(), m_tiers.end(), [](const auto& tier) {
            return tier->wants_chain_context();
        });
    }

    void flush() override {
        for (auto& tier : m_tiers) {
            tier->flush();
        }
    }

    /// Every tier gets the report; the caller reports the metrics unless some tier does.
    bool report_restore_stats(const RestoreStats& stats) override {
        bool taken = false;
        for (auto& tier : m_tiers) {
            taken = tier->report_restore_stats(stats) || taken;
        }
        return taken;
    }

    bool wants_prefix_tokens() const override {
        return std::any_of(m_tiers.begin(), m_tiers.end(), [](const auto& tier) {
            return tier->wants_prefix_tokens();
        });
    }

    void announce_prefix_tokens(const std::vector<PrefixTokens>& prefixes) override {
        for (auto& tier : m_tiers) {
            if (tier->wants_prefix_tokens()) {
                tier->announce_prefix_tokens(prefixes);
            }
        }
    }

private:
    /// Copies the blocks tier @p source_tier just served into the faster tiers.
    void promote(size_t source_tier,
                 const std::vector<PrefixKey>& keys,
                 const std::vector<bool>& found,
                 const std::vector<std::vector<uint8_t>>& data,
                 const std::vector<size_t>& positions,
                 const std::vector<bool>& served) const {
        if (source_tier == 0) {
            return;
        }
        std::vector<PrefixKey> promoted_keys;
        std::vector<const std::vector<uint8_t>*> promoted_data;
        for (size_t i = 0; i < positions.size(); ++i) {
            if (served[i] && found[positions[i]]) {
                promoted_keys.push_back(keys[positions[i]]);
                promoted_data.push_back(&data[positions[i]]);
            }
        }
        if (promoted_keys.empty()) {
            return;
        }
        for (size_t faster = 0; faster < source_tier; ++faster) {
            m_tiers[faster]->write_blocks(promoted_keys, promoted_data);
        }
    }

    std::vector<std::unique_ptr<IKVCacheStore>> m_tiers;
};

}  // namespace ov::genai
