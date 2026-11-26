// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <deque>
#include <map>

#include "continuous_batching/cache/tiered_kv_cache_store.hpp"
#include "store_test_utils.hpp"

using namespace ov::genai;

namespace {

constexpr size_t block_size = 4;

/// An in-memory tier that keeps at most `capacity` blocks and drops the oldest by itself.
class FifoMemoryStore : public test::HashKeyedStore {
public:
    FifoMemoryStore(size_t capacity, bool persistent = false) : m_capacity(capacity), m_persistent(persistent) {}

    bool has_block(size_t hash) const override {
        return m_blocks.count(hash) != 0;
    }
    bool write_block(size_t hash, const std::vector<uint8_t>& data) override {
        if (m_blocks.count(hash) == 0) {
            m_order.push_back(hash);
        }
        m_blocks[hash] = data;
        while (m_blocks.size() > m_capacity) {
            m_blocks.erase(m_order.front());
            m_order.pop_front();
        }
        ++num_writes;
        return true;
    }
    bool read_block(size_t hash, std::vector<uint8_t>& data) const override {
        const auto it = m_blocks.find(hash);
        if (it == m_blocks.end()) {
            return false;
        }
        data = it->second;
        ++num_reads;
        return true;
    }
    void drop_block(size_t hash) override {
        m_blocks.erase(hash);
        m_order.erase(std::remove(m_order.begin(), m_order.end(), hash), m_order.end());
    }
    size_t get_block_size() const override {
        return block_size;
    }
    size_t get_capacity_blocks() const override {
        return m_capacity;
    }
    bool is_plugin_backed() const override {
        return m_persistent;
    }
    bool report_restore_stats(const RestoreStats&) override {
        ++num_reports;
        return m_reports;
    }

    mutable size_t num_reads = 0;
    size_t num_writes = 0;
    size_t num_reports = 0;
    bool m_reports = false;

private:
    size_t m_capacity;
    bool m_persistent;
    std::map<size_t, std::vector<uint8_t>> m_blocks;
    std::deque<size_t> m_order;
};

struct Tiers {
    FifoMemoryStore* fast;
    FifoMemoryStore* slow;
    std::unique_ptr<TieredKVCacheStore> store;
};

Tiers make_two_tiers(size_t fast_capacity, size_t slow_capacity) {
    auto fast = std::make_unique<FifoMemoryStore>(fast_capacity);
    auto slow = std::make_unique<FifoMemoryStore>(slow_capacity, /*persistent=*/true);
    Tiers tiers{fast.get(), slow.get(), nullptr};
    std::vector<std::unique_ptr<IKVCacheStore>> list;
    list.push_back(std::move(fast));
    list.push_back(std::move(slow));
    tiers.store = std::make_unique<TieredKVCacheStore>(std::move(list));
    return tiers;
}

PrefixKey key(size_t hash) {
    return PrefixKey{hash, std::nullopt};
}

std::vector<uint8_t> payload(uint8_t value) {
    return std::vector<uint8_t>(block_size, value);
}

}  // namespace

TEST(TestTieredKVCacheStore, WritesEveryBlockToEveryTier) {
    auto tiers = make_two_tiers(4, 4);
    const auto a = payload(1);
    const auto b = payload(2);
    EXPECT_EQ(tiers.store->write_blocks({key(1), key(2)}, {&a, &b}), (std::vector<bool>{true, true}));
    EXPECT_TRUE(tiers.fast->has_block(1));
    EXPECT_TRUE(tiers.slow->has_block(2));
    EXPECT_EQ(tiers.store->has_blocks({key(1), key(2), key(3)}), (std::vector<bool>{true, true, false}));
}

TEST(TestTieredKVCacheStore, ReadsFromTheFastTierFirst) {
    auto tiers = make_two_tiers(4, 4);
    const auto a = payload(1);
    tiers.store->write_blocks({key(1)}, {&a});

    std::vector<std::vector<uint8_t>> data;
    EXPECT_EQ(tiers.store->read_blocks({key(1)}, data), std::vector<bool>({true}));
    EXPECT_EQ(data[0], a);
    EXPECT_EQ(tiers.fast->num_reads, 1u);
    EXPECT_EQ(tiers.slow->num_reads, 0u);
}

TEST(TestTieredKVCacheStore, ServesABlockTheFastTierLostFromTheSlowTierWithoutCopyingItBack) {
    auto tiers = make_two_tiers(/*fast=*/1, /*slow=*/4);
    const auto a = payload(1);
    const auto b = payload(2);
    tiers.store->write_blocks({key(1), key(2)}, {&a, &b});
    ASSERT_FALSE(tiers.fast->has_block(1));  // the fast tier kept only the newer block

    std::vector<std::vector<uint8_t>> data;
    EXPECT_EQ(tiers.store->read_blocks({key(1), key(2)}, data), (std::vector<bool>{true, true}));
    EXPECT_EQ(data[0], a);
    EXPECT_EQ(data[1], b);
    EXPECT_EQ(tiers.fast->num_reads, 1u);
    EXPECT_EQ(tiers.slow->num_reads, 1u);  // only the block the fast tier missed
    EXPECT_FALSE(tiers.fast->has_block(1));
}

TEST(TestTieredKVCacheStore, ReportsABlockWhenAnyTierHasIt) {
    auto tiers = make_two_tiers(/*fast=*/1, /*slow=*/1);
    const auto a = payload(1);
    const auto b = payload(2);
    tiers.store->write_blocks({key(1), key(2)}, {&a, &b});
    EXPECT_EQ(tiers.store->has_blocks({key(1), key(2)}), (std::vector<bool>{false, true}));

    std::vector<std::vector<uint8_t>> data;
    EXPECT_EQ(tiers.store->read_blocks({key(1), key(2)}, data), (std::vector<bool>{false, true}));
    EXPECT_TRUE(data[0].empty());
}

TEST(TestTieredKVCacheStore, InvalidatesTheBlockInEveryTier) {
    auto tiers = make_two_tiers(4, 4);
    const auto a = payload(1);
    tiers.store->write_blocks({key(1)}, {&a});
    tiers.store->invalidate(key(1));
    EXPECT_FALSE(tiers.fast->has_block(1));
    EXPECT_FALSE(tiers.slow->has_block(1));
}

TEST(TestTieredKVCacheStore, CombinesTheTraitsOfItsTiers) {
    auto tiers = make_two_tiers(2, 8);
    EXPECT_EQ(tiers.store->block_bytes(), block_size);
    EXPECT_EQ(tiers.store->capacity_blocks_hint(), 8u);
    EXPECT_TRUE(tiers.store->is_persistent());
    EXPECT_TRUE(tiers.store->wants_chain_context());
    EXPECT_FALSE(tiers.store->wants_prefix_tokens());
}

TEST(TestTieredKVCacheStore, TheCallerReportsRestoreStatsUnlessATierDoes) {
    auto tiers = make_two_tiers(4, 4);
    EXPECT_FALSE(tiers.store->report_restore_stats(RestoreStats{}));
    EXPECT_EQ(tiers.fast->num_reports, 1u);
    EXPECT_EQ(tiers.slow->num_reports, 1u);

    tiers.slow->m_reports = true;
    EXPECT_TRUE(tiers.store->report_restore_stats(RestoreStats{}));
}

TEST(TestTieredKVCacheStore, RejectsTiersOfDifferentBlockSizes) {
    class OtherSize : public FifoMemoryStore {
    public:
        OtherSize() : FifoMemoryStore(1) {}
        size_t get_block_size() const override {
            return 2 * block_size;
        }
    };
    std::vector<std::unique_ptr<IKVCacheStore>> list;
    list.push_back(std::make_unique<FifoMemoryStore>(1));
    list.push_back(std::make_unique<OtherSize>());
    EXPECT_THROW(TieredKVCacheStore(std::move(list)), ov::Exception);
}

TEST(TestTieredKVCacheStore, ThreeTiersAskTheDeeperTierOnlyForWhatTheFasterOnesMiss) {
    auto first = std::make_unique<FifoMemoryStore>(1);
    auto second = std::make_unique<FifoMemoryStore>(2);
    auto third = std::make_unique<FifoMemoryStore>(8);
    FifoMemoryStore* first_ptr = first.get();
    FifoMemoryStore* second_ptr = second.get();
    FifoMemoryStore* third_ptr = third.get();
    std::vector<std::unique_ptr<IKVCacheStore>> list;
    list.push_back(std::move(first));
    list.push_back(std::move(second));
    list.push_back(std::move(third));
    TieredKVCacheStore store(std::move(list));

    const auto a = payload(1);
    const auto b = payload(2);
    const auto c = payload(3);
    store.write_blocks({key(1), key(2), key(3)}, {&a, &b, &c});
    ASSERT_FALSE(first_ptr->has_block(2));
    ASSERT_FALSE(second_ptr->has_block(1));

    std::vector<std::vector<uint8_t>> data;
    EXPECT_EQ(store.read_blocks({key(1), key(2), key(3)}, data), (std::vector<bool>{true, true, true}));
    EXPECT_EQ(data[0], a);
    EXPECT_EQ(data[1], b);
    EXPECT_EQ(data[2], c);
    EXPECT_EQ(first_ptr->num_reads, 1u);   // block 3
    EXPECT_EQ(second_ptr->num_reads, 1u);  // block 2
    EXPECT_EQ(third_ptr->num_reads, 1u);   // block 1
}
