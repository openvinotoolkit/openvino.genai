// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "continuous_batching/cache/default_file_store.hpp"
#include "continuous_batching/cache/kv_cache_store.hpp"

namespace ov::genai::test {

/**
 * @brief An IKVCacheStore that tests drive with one block hash at a time.
 *
 * The production interface is addressed by PrefixKey and probes whole chains; tests mostly want to say "write block
 * 42", so this class turns the hash-addressed calls below into the key-addressed ones. A hash is a key without a digest.
 * Test doubles derive from it and override what they observe.
 */
class HashKeyedStore : public IKVCacheStore {
public:
    virtual bool has_block(size_t hash) const = 0;
    virtual bool write_block(size_t hash, const std::vector<uint8_t>& data) = 0;
    virtual bool read_block(size_t hash, std::vector<uint8_t>& data) const = 0;
    virtual void drop_block(size_t hash) = 0;
    virtual size_t get_block_size() const = 0;
    virtual size_t get_capacity_blocks() const = 0;
    virtual size_t get_num_free_blocks() const {
        return 0;
    }
    virtual bool is_plugin_backed() const {
        return false;
    }

    virtual std::vector<bool> has_blocks(const std::vector<size_t>& chain) const {
        std::vector<bool> present;
        for (const size_t hash : chain) {
            present.push_back(has_block(hash));
        }
        return present;
    }
    virtual std::vector<bool> write_blocks(const std::vector<size_t>& hashes,
                                           const std::vector<const std::vector<uint8_t>*>& data) {
        std::vector<bool> written;
        for (size_t i = 0; i < hashes.size(); ++i) {
            written.push_back(write_block(hashes[i], *data[i]));
        }
        return written;
    }
    virtual std::vector<bool> read_blocks(const std::vector<size_t>& hashes, std::vector<std::vector<uint8_t>>& data) const {
        std::vector<bool> found;
        data.assign(hashes.size(), {});
        for (size_t i = 0; i < hashes.size(); ++i) {
            found.push_back(read_block(hashes[i], data[i]));
        }
        return found;
    }

    std::vector<bool> has_blocks(const std::vector<PrefixKey>& chain) const override {
        return has_blocks(hashes_of(chain));
    }
    std::vector<bool> write_blocks(const std::vector<PrefixKey>& keys,
                                   const std::vector<const std::vector<uint8_t>*>& data) override {
        return write_blocks(hashes_of(keys), data);
    }
    std::vector<bool> read_blocks(const std::vector<PrefixKey>& keys, std::vector<std::vector<uint8_t>>& data) const override {
        return read_blocks(hashes_of(keys), data);
    }
    void invalidate(const PrefixKey& key) override {
        drop_block(key.hash);
    }
    size_t block_bytes() const override {
        return get_block_size();
    }
    size_t capacity_blocks_hint() const override {
        return get_capacity_blocks();
    }
    bool is_persistent() const override {
        return is_plugin_backed();
    }
    bool wants_chain_context() const override {
        return is_plugin_backed();
    }

protected:
    static std::vector<size_t> hashes_of(const std::vector<PrefixKey>& keys) {
        std::vector<size_t> hashes;
        for (const PrefixKey& key : keys) {
            hashes.push_back(key.hash);
        }
        return hashes;
    }
};

/// A HashKeyedStore over any store, for tests that name blocks by hash. The inner store has to take blocks without a digest.
class HashAdapter final : public HashKeyedStore {
public:
    explicit HashAdapter(std::unique_ptr<IKVCacheStore> inner) : m_inner(std::move(inner)) {}

    bool has_block(size_t hash) const override {
        return m_inner->has_blocks({PrefixKey{hash, std::nullopt}})[0];
    }
    bool write_block(size_t hash, const std::vector<uint8_t>& data) override {
        return m_inner->write_blocks({PrefixKey{hash, std::nullopt}}, {&data})[0];
    }
    bool read_block(size_t hash, std::vector<uint8_t>& data) const override {
        std::vector<std::vector<uint8_t>> blocks;
        if (!m_inner->read_blocks({PrefixKey{hash, std::nullopt}}, blocks)[0]) {
            return false;
        }
        data = std::move(blocks[0]);
        return true;
    }
    void drop_block(size_t hash) override {
        m_inner->invalidate(PrefixKey{hash, std::nullopt});
    }
    size_t get_block_size() const override {
        return m_inner->block_bytes();
    }
    size_t get_capacity_blocks() const override {
        return m_inner->capacity_blocks_hint();
    }
    size_t get_num_free_blocks() const override {
        const auto* files = dynamic_cast<const DefaultFileGroupStore*>(m_inner.get());
        return files != nullptr ? files->get_num_free_blocks() : 0;
    }
    void flush() override {
        m_inner->flush();
    }

private:
    std::unique_ptr<IKVCacheStore> m_inner;
};

}  // namespace ov::genai::test
