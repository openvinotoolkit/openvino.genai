// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching/cache/kv_cache_offload_manager.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <limits>
#include <random>

#ifdef _WIN32
#    include <fcntl.h>
#    include <io.h>
#    include <process.h>
#    include <share.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#else
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#    include <unistd.h>
#endif

#include "openvino/core/except.hpp"

namespace ov::genai {

namespace {

constexpr size_t MAX_FILE_NAME_ATTEMPTS = 16;

std::string make_offload_file_name() {
    static std::atomic<uint64_t> counter{0};
#ifdef _WIN32
    const auto pid = static_cast<unsigned long long>(_getpid());
#else
    const auto pid = static_cast<unsigned long long>(::getpid());
#endif
    std::random_device random_device;
    return "ov_genai_kv_offload_" + std::to_string(pid) + "_" +
           std::to_string(counter.fetch_add(1)) + "_" + std::to_string(random_device()) + ".bin";
}

int create_exclusive_file(const std::filesystem::path& path) {
#ifdef _WIN32
    int fd = -1;
    const errno_t error = _wsopen_s(&fd,
                                    path.wstring().c_str(),
                                    _O_RDWR | _O_CREAT | _O_EXCL | _O_BINARY | _O_RANDOM,
                                    _SH_DENYRW,
                                    _S_IREAD | _S_IWRITE);
    return error == 0 ? fd : -1;
#else
    return ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
#endif
}

void resize_file(int fd, size_t size) {
#ifdef _WIN32
    const int result = _chsize_s(fd, static_cast<__int64>(size));
    const bool failed = result != 0;
#else
    const bool failed = ::ftruncate(fd, static_cast<off_t>(size)) != 0;
#endif
    OPENVINO_ASSERT(!failed, "Failed to reserve ", size, " bytes for the KV cache offload file: ", std::strerror(errno));
}

void close_file(int fd) {
#ifdef _WIN32
    _close(fd);
#else
    ::close(fd);
#endif
}

}  // namespace

bool KVCacheOffloadManager::is_supported_device(const std::string& device) {
    // Device caches are staged through host tensors, so anything the KV cache manager can copy a block
    // out of works; NPU keeps its cache outside of this manager and is therefore excluded.
    return device.find("CPU") != std::string::npos || device.find("GPU") != std::string::npos;
}

KVCacheOffloadManager::KVCacheOffloadManager(const KVCacheDiskLayout& layout,
                                             const CacheOffloadConfig& config,
                                             const std::string& device)
    : m_layout(layout) {
    OPENVINO_ASSERT(is_supported_device(device),
                    "KV cache disk offload is implemented for CPU and GPU, but the inference device is '",
                    device,
                    "'");
    OPENVINO_ASSERT(config.use_page_cache,
                    "KV cache disk offload with direct I/O is not implemented yet, set use_page_cache to true");

    m_slot_size = m_layout.get_slot_size();
    OPENVINO_ASSERT(m_slot_size > 0, "KV cache offload slot size must be greater than 0");

    m_num_slots = config.capacity_bytes / m_slot_size;
    OPENVINO_ASSERT(m_num_slots > 0,
                    "KV cache offload capacity of ",
                    config.capacity_bytes,
                    " bytes is smaller than a single cache block of ",
                    m_slot_size,
                    " bytes");

    std::filesystem::path directory;
    if (config.path.empty()) {
        directory = std::filesystem::temp_directory_path();
    } else {
        directory = std::filesystem::path(config.path);
        OPENVINO_ASSERT(std::filesystem::is_directory(directory),
                        "KV cache offload path '",
                        config.path,
                        "' is not an existing directory");
    }

    for (size_t attempt = 0; attempt < MAX_FILE_NAME_ATTEMPTS && m_fd < 0; ++attempt) {
        const std::filesystem::path candidate = directory / make_offload_file_name();
        const int fd = create_exclusive_file(candidate);
        if (fd >= 0) {
            m_fd = fd;
            m_file_path = candidate;
        } else if (errno != EEXIST) {
            OPENVINO_THROW("Failed to create the KV cache offload file in '",
                           directory.string(),
                           "': ",
                           std::strerror(errno));
        }
    }
    OPENVINO_ASSERT(m_fd >= 0,
                    "Failed to create a unique KV cache offload file in '",
                    directory.string(),
                    "'");

    try {
        resize_file(m_fd, m_num_slots * m_slot_size);
    } catch (...) {
        close_and_remove();
        throw;
    }

    m_free_slots.reserve(m_num_slots);
    for (size_t slot_id = m_num_slots; slot_id > 0; --slot_id) {
        m_free_slots.push_back(slot_id - 1);
    }
}

KVCacheOffloadManager::~KVCacheOffloadManager() {
    close_and_remove();
}

void KVCacheOffloadManager::close_and_remove() noexcept {
    if (m_fd >= 0) {
        close_file(m_fd);
        m_fd = -1;
    }
    if (!m_file_path.empty()) {
        std::error_code error_code;
        std::filesystem::remove(m_file_path, error_code);
        m_file_path.clear();
    }
}

size_t KVCacheOffloadManager::get_num_free_slots() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_free_slots.size();
}

std::optional<size_t> KVCacheOffloadManager::acquire_slot() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_free_slots.empty()) {
        return std::nullopt;
    }
    const size_t slot_id = m_free_slots.back();
    m_free_slots.pop_back();
    return slot_id;
}

void KVCacheOffloadManager::release_slot(size_t slot_id) {
    std::lock_guard<std::mutex> lock(m_mutex);
    OPENVINO_ASSERT(slot_id < m_num_slots, "Invalid KV cache offload slot ", slot_id);
    OPENVINO_ASSERT(std::find(m_free_slots.begin(), m_free_slots.end(), slot_id) == m_free_slots.end(),
                    "KV cache offload slot ",
                    slot_id,
                    " is released twice");
    m_free_slots.push_back(slot_id);
}

void KVCacheOffloadManager::write_slot(size_t slot_id, const std::vector<uint8_t>& block_data) {
    OPENVINO_ASSERT(slot_id < m_num_slots, "Invalid KV cache offload slot ", slot_id);
    OPENVINO_ASSERT(block_data.size() == m_slot_size,
                    "Unexpected KV cache offload block size: got ",
                    block_data.size(),
                    ", expected ",
                    m_slot_size);

    std::lock_guard<std::mutex> lock(m_mutex);
    write_at(m_layout.get_slot_offset(slot_id), block_data.data(), m_slot_size);
}

void KVCacheOffloadManager::read_slot(size_t slot_id, std::vector<uint8_t>& block_data) const {
    OPENVINO_ASSERT(slot_id < m_num_slots, "Invalid KV cache offload slot ", slot_id);
    block_data.resize(m_slot_size);

    std::lock_guard<std::mutex> lock(m_mutex);
    read_at(m_layout.get_slot_offset(slot_id), block_data.data(), m_slot_size);
}

void KVCacheOffloadManager::write_at(size_t offset, const uint8_t* data, size_t size) const {
    size_t written = 0;
    while (written < size) {
        const size_t remaining = size - written;
#ifdef _WIN32
        OPENVINO_ASSERT(_lseeki64(m_fd, static_cast<__int64>(offset + written), SEEK_SET) >= 0,
                        "Failed to seek in the KV cache offload file: ",
                        std::strerror(errno));
        const auto chunk =
            static_cast<unsigned int>(std::min<size_t>(remaining, std::numeric_limits<int>::max()));
        const int result = _write(m_fd, data + written, chunk);
#else
        const ssize_t result =
            ::pwrite(m_fd, data + written, remaining, static_cast<off_t>(offset + written));
#endif
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            OPENVINO_THROW("Failed to write the KV cache offload file: ", std::strerror(errno));
        }
        OPENVINO_ASSERT(result > 0,
                        "Short write to the KV cache offload file: wrote ",
                        written,
                        " of ",
                        size,
                        " bytes");
        written += static_cast<size_t>(result);
    }
}

void KVCacheOffloadManager::read_at(size_t offset, uint8_t* data, size_t size) const {
    size_t read_bytes = 0;
    while (read_bytes < size) {
        const size_t remaining = size - read_bytes;
#ifdef _WIN32
        OPENVINO_ASSERT(_lseeki64(m_fd, static_cast<__int64>(offset + read_bytes), SEEK_SET) >= 0,
                        "Failed to seek in the KV cache offload file: ",
                        std::strerror(errno));
        const auto chunk =
            static_cast<unsigned int>(std::min<size_t>(remaining, std::numeric_limits<int>::max()));
        const int result = _read(m_fd, data + read_bytes, chunk);
#else
        const ssize_t result =
            ::pread(m_fd, data + read_bytes, remaining, static_cast<off_t>(offset + read_bytes));
#endif
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            OPENVINO_THROW("Failed to read the KV cache offload file: ", std::strerror(errno));
        }
        OPENVINO_ASSERT(result > 0,
                        "Short read from the KV cache offload file: read ",
                        read_bytes,
                        " of ",
                        size,
                        " bytes");
        read_bytes += static_cast<size_t>(result);
    }
}

}  // namespace ov::genai
