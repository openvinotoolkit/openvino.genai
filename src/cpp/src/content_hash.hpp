// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace ov::genai {

/// Stable, cross-process / cross-platform 128-bit content digest.
///
/// Unlike `std::hash<T>`, whose output is implementation-defined and may change
/// across standard library versions or compilers, `ContentHash` is a documented,
/// versioned algorithm (see `CONTENT_HASH_ALGORITHM_VERSION`) that two different
/// processes or builds are guaranteed to reproduce identically for the same
/// input bytes. This is required wherever a content hash may outlive the
/// process that computed it (e.g. a persisted prefix-cache block key).
struct ContentHash final {
    uint64_t lo = 0;
    uint64_t hi = 0;

    // Hand-written comparisons: this header must stay buildable as C++17, where
    // defaulted comparison operators are not available.
    friend bool operator==(const ContentHash& a, const ContentHash& b) noexcept {
        return a.lo == b.lo && a.hi == b.hi;
    }
    friend bool operator!=(const ContentHash& a, const ContentHash& b) noexcept {
        return !(a == b);
    }
};

/// Bumped whenever the underlying hashing algorithm changes. Callers that persist
/// a `ContentHash` must also persist this version alongside it.
constexpr uint32_t CONTENT_HASH_ALGORITHM_VERSION = 1;  // MurmurHash3_x64_128

/// Computes the stable 128-bit content digest of `[data, data + size)`.
ContentHash compute_content_hash(const void* data, size_t size, uint64_t seed = 0);

}  // namespace ov::genai
