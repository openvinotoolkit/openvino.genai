// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "content_hash.hpp"

#include <cstring>

namespace ov::genai {

namespace {

// MurmurHash3_x64_128, public domain (Austin Appleby, 2011). Chosen over
// std::hash because its output is a documented algorithm rather than an
// implementation-defined one, so it is reproducible across processes, compilers
// and standard library versions - the property a persisted content hash needs.
inline uint64_t rotl64(uint64_t x, int8_t r) {
    return (x << r) | (x >> (64 - r));
}

inline uint64_t fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

void murmur_hash3_x64_128(const void* key, size_t len, uint64_t seed, uint64_t out[2]) {
    const uint8_t* data = reinterpret_cast<const uint8_t*>(key);
    const size_t num_blocks = len / 16;

    uint64_t h1 = seed;
    uint64_t h2 = seed;

    constexpr uint64_t c1 = 0x87c37b91114253d5ULL;
    constexpr uint64_t c2 = 0x4cf5ad432745937fULL;

    const uint8_t* blocks = data;
    for (size_t i = 0; i < num_blocks; ++i) {
        uint64_t k1, k2;
        std::memcpy(&k1, blocks + i * 16, sizeof(k1));
        std::memcpy(&k2, blocks + i * 16 + 8, sizeof(k2));

        k1 *= c1;
        k1 = rotl64(k1, 31);
        k1 *= c2;
        h1 ^= k1;

        h1 = rotl64(h1, 27);
        h1 += h2;
        h1 = h1 * 5 + 0x52dce729;

        k2 *= c2;
        k2 = rotl64(k2, 33);
        k2 *= c1;
        h2 ^= k2;

        h2 = rotl64(h2, 31);
        h2 += h1;
        h2 = h2 * 5 + 0x38495ab5;
    }

    const uint8_t* tail = data + num_blocks * 16;
    uint64_t k1 = 0;
    uint64_t k2 = 0;
    switch (len & 15) {
        case 15: k2 ^= static_cast<uint64_t>(tail[14]) << 48; [[fallthrough]];
        case 14: k2 ^= static_cast<uint64_t>(tail[13]) << 40; [[fallthrough]];
        case 13: k2 ^= static_cast<uint64_t>(tail[12]) << 32; [[fallthrough]];
        case 12: k2 ^= static_cast<uint64_t>(tail[11]) << 24; [[fallthrough]];
        case 11: k2 ^= static_cast<uint64_t>(tail[10]) << 16; [[fallthrough]];
        case 10: k2 ^= static_cast<uint64_t>(tail[9]) << 8; [[fallthrough]];
        case 9:
            k2 ^= static_cast<uint64_t>(tail[8]);
            k2 *= c2;
            k2 = rotl64(k2, 33);
            k2 *= c1;
            h2 ^= k2;
            [[fallthrough]];
        case 8: k1 ^= static_cast<uint64_t>(tail[7]) << 56; [[fallthrough]];
        case 7: k1 ^= static_cast<uint64_t>(tail[6]) << 48; [[fallthrough]];
        case 6: k1 ^= static_cast<uint64_t>(tail[5]) << 40; [[fallthrough]];
        case 5: k1 ^= static_cast<uint64_t>(tail[4]) << 32; [[fallthrough]];
        case 4: k1 ^= static_cast<uint64_t>(tail[3]) << 24; [[fallthrough]];
        case 3: k1 ^= static_cast<uint64_t>(tail[2]) << 16; [[fallthrough]];
        case 2: k1 ^= static_cast<uint64_t>(tail[1]) << 8; [[fallthrough]];
        case 1:
            k1 ^= static_cast<uint64_t>(tail[0]);
            k1 *= c1;
            k1 = rotl64(k1, 31);
            k1 *= c2;
            h1 ^= k1;
            break;
        default:
            break;
    }

    h1 ^= len;
    h2 ^= len;

    h1 += h2;
    h2 += h1;
    h1 = fmix64(h1);
    h2 = fmix64(h2);
    h1 += h2;
    h2 += h1;

    out[0] = h1;
    out[1] = h2;
}

}  // namespace

ContentHash compute_content_hash(const void* data, size_t size, uint64_t seed) {
    uint64_t digest[2];
    murmur_hash3_x64_128(data, size, seed, digest);
    ContentHash result;
    result.lo = digest[0];
    result.hi = digest[1];
    return result;
}

}  // namespace ov::genai
