// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <list>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <sys/file.h>
#    include <unistd.h>
#endif

#include "content_hash.hpp"
#include "continuous_batching/cache/kv_cache_store.hpp"
#include "logger.hpp"
#include "openvino/core/except.hpp"

namespace ov::genai {

namespace default_file_store_detail {

constexpr uint32_t SLOT_MAGIC = 0x4B56564F;  // "OVVK"
constexpr uint32_t FORMAT_VERSION = 1;
constexpr size_t HEADER_BYTES = 64;
constexpr size_t STRIDE_ALIGNMENT = 4096;
constexpr const char* ROOT_DIRECTORY_NAME = "ov_kvcache_v1";

struct SlotHeader {
    uint64_t seq = 0;
    ContentHash key;
    ContentHash payload_hash;
    uint64_t payload_bytes = 0;
    uint32_t flags = 0;  ///< none defined yet; a reader of this format version ignores it
};

// Field offsets inside the 64 byte header; the bytes in between are zero.
constexpr size_t OFFSET_MAGIC = 0;
constexpr size_t OFFSET_VERSION = 4;
constexpr size_t OFFSET_SEQ = 8;
constexpr size_t OFFSET_KEY = 16;
constexpr size_t OFFSET_PAYLOAD_HASH = 32;
constexpr size_t OFFSET_PAYLOAD_BYTES = 48;
constexpr size_t OFFSET_FLAGS = 56;

inline void encode_header(const SlotHeader& header, uint8_t (&bytes)[HEADER_BYTES]) {
    std::memset(bytes, 0, HEADER_BYTES);
    const uint32_t magic = SLOT_MAGIC;
    const uint32_t version = FORMAT_VERSION;
    std::memcpy(bytes + OFFSET_MAGIC, &magic, sizeof(magic));
    std::memcpy(bytes + OFFSET_VERSION, &version, sizeof(version));
    std::memcpy(bytes + OFFSET_SEQ, &header.seq, sizeof(header.seq));
    std::memcpy(bytes + OFFSET_KEY, &header.key.lo, sizeof(uint64_t));
    std::memcpy(bytes + OFFSET_KEY + sizeof(uint64_t), &header.key.hi, sizeof(uint64_t));
    std::memcpy(bytes + OFFSET_PAYLOAD_HASH, &header.payload_hash.lo, sizeof(uint64_t));
    std::memcpy(bytes + OFFSET_PAYLOAD_HASH + sizeof(uint64_t), &header.payload_hash.hi, sizeof(uint64_t));
    std::memcpy(bytes + OFFSET_PAYLOAD_BYTES, &header.payload_bytes, sizeof(header.payload_bytes));
    std::memcpy(bytes + OFFSET_FLAGS, &header.flags, sizeof(header.flags));
}

/// @return false when the bytes are not a header of this format.
inline bool decode_header(const uint8_t (&bytes)[HEADER_BYTES], SlotHeader& header) {
    uint32_t magic = 0;
    uint32_t version = 0;
    std::memcpy(&magic, bytes + OFFSET_MAGIC, sizeof(magic));
    std::memcpy(&version, bytes + OFFSET_VERSION, sizeof(version));
    if (magic != SLOT_MAGIC || version != FORMAT_VERSION) {
        return false;
    }
    std::memcpy(&header.seq, bytes + OFFSET_SEQ, sizeof(header.seq));
    std::memcpy(&header.key.lo, bytes + OFFSET_KEY, sizeof(uint64_t));
    std::memcpy(&header.key.hi, bytes + OFFSET_KEY + sizeof(uint64_t), sizeof(uint64_t));
    std::memcpy(&header.payload_hash.lo, bytes + OFFSET_PAYLOAD_HASH, sizeof(uint64_t));
    std::memcpy(&header.payload_hash.hi, bytes + OFFSET_PAYLOAD_HASH + sizeof(uint64_t), sizeof(uint64_t));
    std::memcpy(&header.payload_bytes, bytes + OFFSET_PAYLOAD_BYTES, sizeof(header.payload_bytes));
    std::memcpy(&header.flags, bytes + OFFSET_FLAGS, sizeof(header.flags));
    return true;
}

/// Reserves @p size bytes of disk for the file, so that a full disk is found when the store opens and not by a failed write.
/// A file system without the call (e.g. tmpfs) is left sparse.
inline void reserve_file_space(const std::filesystem::path& path, size_t size) {
    std::error_code error;
    std::filesystem::resize_file(path, size, error);
    OPENVINO_ASSERT(!error, "Failed to reserve ", size, " bytes in '", path.string(), "': ", error.message());
#ifndef _WIN32
    const int fd = ::open(path.c_str(), O_RDWR);
    if (fd >= 0) {
        const int result = ::posix_fallocate(fd, 0, static_cast<off_t>(size));
        ::close(fd);
        OPENVINO_ASSERT(result == 0 || result == EOPNOTSUPP || result == EINVAL,
                        "Failed to reserve ", size, " bytes in '", path.string(), "': ", std::strerror(result));
    }
#endif
}

struct ContentHashHasher {
    size_t operator()(const ContentHash& hash) const noexcept {
        return static_cast<size_t>(hash.lo ^ (hash.hi * 0x9E3779B97F4A7C15ull));
    }
};

inline std::string to_hex(const ContentHash& hash) {
    char text[33];
    std::snprintf(text, sizeof(text), "%016llx%016llx", static_cast<unsigned long long>(hash.hi),
                  static_cast<unsigned long long>(hash.lo));
    return text;
}

/// Holds an exclusive lock on a file until destroyed; the lock is dropped by the OS when the process dies.
class ExclusiveFileLock {
public:
    explicit ExclusiveFileLock(const std::filesystem::path& path) {
#ifdef _WIN32
        m_handle = ::CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                                 FILE_ATTRIBUTE_NORMAL, nullptr);
        OPENVINO_ASSERT(m_handle != INVALID_HANDLE_VALUE,
                        "The KV cache directory '", path.parent_path().string(),
                        "' is in use by another pipeline or process (cannot lock '", path.string(), "')");
#else
        m_fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0600);
        OPENVINO_ASSERT(m_fd >= 0, "Failed to open the KV cache lock file '", path.string(), "'");
        if (::flock(m_fd, LOCK_EX | LOCK_NB) != 0) {
            ::close(m_fd);
            OPENVINO_THROW("The KV cache directory '", path.parent_path().string(),
                           "' is in use by another pipeline or process (cannot lock '", path.string(), "')");
        }
#endif
    }

    ~ExclusiveFileLock() {
#ifdef _WIN32
        ::CloseHandle(m_handle);
#else
        ::close(m_fd);
#endif
    }

    ExclusiveFileLock(const ExclusiveFileLock&) = delete;
    ExclusiveFileLock& operator=(const ExclusiveFileLock&) = delete;

private:
#ifdef _WIN32
    HANDLE m_handle = INVALID_HANDLE_VALUE;
#else
    int m_fd = -1;
#endif
};

/// The directory of one namespace, locked for as long as any group store uses it.
class StoreDirectory {
public:
    /**
     * @param root The user's `kv_cache_dir`, which has to exist.
     * @param name_space Namespace of the store. One that carries a model identity (it contains ':') names a directory
     * that is kept and reused by later runs; any other one gets a private directory that is deleted on destruction.
     */
    StoreDirectory(const std::filesystem::path& root, const std::string& name_space)
        : m_persistent(name_space.find(':') != std::string::npos) {
        OPENVINO_ASSERT(!root.empty() && std::filesystem::is_directory(root),
                        "KV cache offload kv_cache_dir '", root.string(), "' is not an existing directory");
        const std::filesystem::path parent = root / ROOT_DIRECTORY_NAME;
        std::error_code error;
        std::filesystem::create_directories(parent, error);
        OPENVINO_ASSERT(!error, "Failed to create '", parent.string(), "': ", error.message());
        m_path = parent / (m_persistent ? to_hex(compute_content_hash(name_space.data(), name_space.size()))
                                        : make_private_name());
        std::filesystem::create_directories(m_path, error);
        OPENVINO_ASSERT(!error, "Failed to create '", m_path.string(), "': ", error.message());
        m_lock = std::make_unique<ExclusiveFileLock>(m_path / "LOCK");
    }

    ~StoreDirectory() {
        m_lock.reset();
        if (!m_persistent) {
            std::error_code error;
            std::filesystem::remove_all(m_path, error);
        }
    }

    StoreDirectory(const StoreDirectory&) = delete;
    StoreDirectory& operator=(const StoreDirectory&) = delete;

    const std::filesystem::path& path() const {
        return m_path;
    }

    bool persistent() const {
        return m_persistent;
    }

private:
    static std::string make_private_name() {
        static std::atomic<uint64_t> counter{0};
        std::random_device random_device;
        return "run_" + std::to_string(counter.fetch_add(1)) + "_" + std::to_string(random_device()) + "_" +
               std::to_string(random_device());
    }

    bool m_persistent;
    std::filesystem::path m_path;
    std::unique_ptr<ExclusiveFileLock> m_lock;
};

}  // namespace default_file_store_detail

/**
 * @brief The in-tree store of one cache group: fixed-size slots in a file of the SSD tier, with an LRU of its own.
 *
 * A slot is `[header 64 B][payload]`, aligned to 4 KiB. The payload is written before the header, and a header names
 * the key and a hash of the payload, so a slot that a crash left half written fails the check on read and counts as
 * a miss. There is no index file: opening a store reads all headers and orders the slots by their sequence number.
 * The store evicts the least recently used block when it is full, so writing never fails for lack of capacity.
 *
 * A store of a namespace with a model identity is persistent: it is keyed by the 128-bit digest only, and a block
 * without a digest is not stored. Without an identity the store is private to this run and also takes a block by its size_t hash.
 */
class DefaultFileGroupStore final : public IKVCacheStore {
public:
    DefaultFileGroupStore(std::shared_ptr<default_file_store_detail::StoreDirectory> directory,
                          const std::string& file_stem,
                          size_t block_bytes,
                          size_t capacity_blocks,
                          bool reuse_existing)
        : m_directory(std::move(directory)),
          m_block_bytes(block_bytes),
          m_capacity_blocks(capacity_blocks),
          m_stride(align_up(default_file_store_detail::HEADER_BYTES + block_bytes, default_file_store_detail::STRIDE_ALIGNMENT)) {
        using namespace default_file_store_detail;
        OPENVINO_ASSERT(m_block_bytes > 0, "KV cache offload slot size must be greater than 0");
        OPENVINO_ASSERT(m_capacity_blocks > 0, "KV cache offload capacity is smaller than a single cache block of ", m_block_bytes, " bytes");

        const std::filesystem::path path = m_directory->path() / (file_stem + ".slots");
        const bool keep_content = reuse_existing && m_directory->persistent() && std::filesystem::exists(path);
        if (!keep_content) {
            std::ofstream(path, std::ios::binary | std::ios::trunc);
        }
        reserve_file_space(path, m_capacity_blocks * m_stride);
        m_file.open(path, std::ios::in | std::ios::out | std::ios::binary);
        OPENVINO_ASSERT(m_file.is_open(), "Failed to open '", path.string(), "'");

        std::vector<std::pair<uint64_t, size_t>> found;  // sequence number, slot
        std::unordered_map<ContentHash, std::pair<uint64_t, size_t>, ContentHashHasher> newest;
        if (keep_content) {
            for (size_t slot = 0; slot < m_capacity_blocks; ++slot) {
                SlotHeader header;
                if (!read_header(slot, header) || header.payload_bytes != m_block_bytes) {
                    continue;
                }
                const auto it = newest.find(header.key);
                if (it == newest.end()) {
                    newest.emplace(header.key, std::make_pair(header.seq, slot));
                } else if (header.seq > it->second.first) {
                    invalidate_slot(it->second.second);
                    it->second = {header.seq, slot};
                } else {
                    invalidate_slot(slot);
                }
                m_next_seq = std::max(m_next_seq, header.seq + 1);
            }
            for (const auto& [key, seq_and_slot] : newest) {
                found.push_back(seq_and_slot);
            }
        }
        std::sort(found.begin(), found.end());
        std::vector<bool> used(m_capacity_blocks, false);
        for (const auto& [seq, slot] : found) {
            used[slot] = true;
            SlotHeader header;
            read_header(slot, header);
            m_recency.push_back(header.key);
            m_index.emplace(header.key, Entry{slot, std::prev(m_recency.end())});
        }
        for (size_t slot = m_capacity_blocks; slot > 0; --slot) {
            if (!used[slot - 1]) {
                m_free_slots.push_back(slot - 1);
            }
        }
        GENAI_INFO("[KV_TRACE] DefaultFileStore '%s' persistent=%d capacity_blocks=%zu restored_blocks=%zu",
                   path.string().c_str(),
                   static_cast<int>(m_directory->persistent()),
                   m_capacity_blocks,
                   m_index.size());
    }

    std::vector<bool> has_blocks(const std::vector<PrefixKey>& chain) const override {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<bool> present;
        present.reserve(chain.size());
        for (const PrefixKey& key : chain) {
            const auto id = key_of(key);
            present.push_back(id.has_value() && m_index.find(*id) != m_index.end());
        }
        return present;
    }

    std::vector<bool> write_blocks(const std::vector<PrefixKey>& keys,
                                   const std::vector<const std::vector<uint8_t>*>& data) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<bool> written;
        written.reserve(keys.size());
        for (size_t i = 0; i < keys.size(); ++i) {
            const auto id = key_of(keys[i]);
            written.push_back(id.has_value() && data[i]->size() == m_block_bytes && write_locked(*id, *data[i]));
        }
        m_file.flush();
        return written;
    }

    std::vector<bool> read_blocks(const std::vector<PrefixKey>& keys,
                                  std::vector<std::vector<uint8_t>>& data) const override {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<bool> found;
        found.reserve(keys.size());
        data.assign(keys.size(), {});
        for (size_t i = 0; i < keys.size(); ++i) {
            const auto id = key_of(keys[i]);
            found.push_back(id.has_value() && read_locked(*id, data[i]));
        }
        return found;
    }

    void invalidate(const PrefixKey& key) override {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (const auto id = key_of(key)) {
            discard_locked(*id);
        }
    }

    size_t block_bytes() const override {
        return m_block_bytes;
    }

    size_t capacity_blocks_hint() const override {
        return m_capacity_blocks;
    }

    /// How many slots hold no block; informational.
    size_t get_num_free_blocks() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_free_slots.size();
    }

    void flush() override {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_file.flush();
    }

    bool is_persistent() const override {
        return m_directory->persistent();
    }

    bool wants_chain_context() const override {
        return false;
    }

private:
    using Recency = std::list<ContentHash>;  // least recently used first
    struct Entry {
        size_t slot;
        Recency::iterator position;
    };

    static size_t align_up(size_t value, size_t alignment) {
        return (value + alignment - 1) / alignment * alignment;
    }

    /// A persistent store addresses a block by its digest alone. A store private to this run may also take a block that
    /// has none, by its size_t hash, which is as good as any key inside one process.
    std::optional<ContentHash> key_of(const PrefixKey& key) const {
        if (key.digest) {
            return key.digest;
        }
        if (m_directory->persistent()) {
            return std::nullopt;
        }
        return ContentHash{static_cast<uint64_t>(key.hash), static_cast<uint64_t>(key.hash)};
    }

    uint64_t slot_offset(size_t slot) const {
        return static_cast<uint64_t>(slot) * m_stride;
    }

    bool read_header(size_t slot, default_file_store_detail::SlotHeader& header) const {
        uint8_t bytes[default_file_store_detail::HEADER_BYTES];
        m_file.clear();
        m_file.seekg(static_cast<std::streamoff>(slot_offset(slot)));
        m_file.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
        return m_file.good() && default_file_store_detail::decode_header(bytes, header);
    }

    void invalidate_slot(size_t slot) const {
        const uint8_t zeros[default_file_store_detail::HEADER_BYTES] = {};
        m_file.clear();
        m_file.seekp(static_cast<std::streamoff>(slot_offset(slot)));
        m_file.write(reinterpret_cast<const char*>(zeros), sizeof(zeros));
    }

    /// Frees the slot of @p id, if it has one.
    void discard_locked(const ContentHash& id) const {
        const auto it = m_index.find(id);
        if (it == m_index.end()) {
            return;
        }
        invalidate_slot(it->second.slot);
        m_free_slots.push_back(it->second.slot);
        m_recency.erase(it->second.position);
        m_index.erase(it);
    }

    bool write_locked(const ContentHash& id, const std::vector<uint8_t>& data) {
        using namespace default_file_store_detail;
        size_t slot = 0;
        const auto existing = m_index.find(id);
        if (existing != m_index.end()) {
            slot = existing->second.slot;
        } else if (!m_free_slots.empty()) {
            slot = m_free_slots.back();
            m_free_slots.pop_back();
        } else {
            const ContentHash victim = m_recency.front();
            slot = m_index.at(victim).slot;
            invalidate_slot(slot);
            m_recency.pop_front();
            m_index.erase(victim);
        }

        SlotHeader header;
        header.seq = m_next_seq++;
        header.key = id;
        header.payload_hash = compute_content_hash(data.data(), data.size());
        header.payload_bytes = data.size();
        uint8_t bytes[HEADER_BYTES];
        encode_header(header, bytes);

        m_file.clear();
        m_file.seekp(static_cast<std::streamoff>(slot_offset(slot) + HEADER_BYTES));
        m_file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        m_file.seekp(static_cast<std::streamoff>(slot_offset(slot)));
        m_file.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
        if (!m_file.good()) {
            GENAI_WARN("DefaultFileStore failed to write slot %zu", slot);
            m_file.clear();
            if (existing != m_index.end()) {
                m_recency.erase(existing->second.position);
                m_index.erase(existing);
            }
            m_free_slots.push_back(slot);
            return false;
        }

        if (existing != m_index.end()) {
            m_recency.splice(m_recency.end(), m_recency, existing->second.position);
        } else {
            m_recency.push_back(id);
            m_index.emplace(id, Entry{slot, std::prev(m_recency.end())});
        }
        return true;
    }

    bool read_locked(const ContentHash& id, std::vector<uint8_t>& data) const {
        using namespace default_file_store_detail;
        const auto it = m_index.find(id);
        if (it == m_index.end()) {
            return false;
        }
        const size_t slot = it->second.slot;
        SlotHeader header;
        bool valid = read_header(slot, header) && header.key == id && header.payload_bytes == m_block_bytes;
        if (valid) {
            data.resize(m_block_bytes);
            m_file.clear();
            m_file.seekg(static_cast<std::streamoff>(slot_offset(slot) + HEADER_BYTES));
            m_file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
            valid = m_file.good() && compute_content_hash(data.data(), data.size()) == header.payload_hash;
        }
        if (!valid) {
            GENAI_WARN("DefaultFileStore found slot %zu damaged, treating the block as missing", slot);
            m_file.clear();
            data.clear();
            discard_locked(id);
            return false;
        }
        m_recency.splice(m_recency.end(), m_recency, it->second.position);
        return true;
    }

    std::shared_ptr<default_file_store_detail::StoreDirectory> m_directory;
    size_t m_block_bytes;
    size_t m_capacity_blocks;
    size_t m_stride;
    mutable std::mutex m_mutex;
    mutable std::fstream m_file;
    mutable std::unordered_map<ContentHash, Entry, default_file_store_detail::ContentHashHasher> m_index;
    mutable Recency m_recency;
    mutable std::vector<size_t> m_free_slots;
    uint64_t m_next_seq = 1;
};

/// The group stores of one namespace directory; see DefaultFileGroupStore.
class DefaultFileStore final : public IGroupedKVCacheStore {
public:
    struct Group {
        std::string file_stem;
        size_t block_bytes;
        size_t capacity_blocks;
    };

    DefaultFileStore(const std::filesystem::path& kv_cache_dir,
                     const std::string& name_space,
                     const std::vector<Group>& groups,
                     bool reuse_existing) {
        auto directory = std::make_shared<default_file_store_detail::StoreDirectory>(kv_cache_dir, name_space);
        for (const Group& group : groups) {
            m_views.push_back(std::make_unique<DefaultFileGroupStore>(directory,
                                                                      group.file_stem,
                                                                      group.block_bytes,
                                                                      group.capacity_blocks,
                                                                      reuse_existing));
        }
    }

    size_t group_count() const override {
        return m_views.size();
    }

    std::unique_ptr<IKVCacheStore> group(size_t index) override {
        return std::move(m_views.at(index));
    }

private:
    std::vector<std::unique_ptr<DefaultFileGroupStore>> m_views;
};

}  // namespace ov::genai
