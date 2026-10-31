// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "content_hash.hpp"

using ov::genai::compute_content_hash;
using ov::genai::ContentHash;

TEST(ContentHashTest, SameInputProducesSameDigest) {
    const std::string data = "the quick brown fox jumps over the lazy dog";
    const auto first = compute_content_hash(data.data(), data.size());
    const auto second = compute_content_hash(data.data(), data.size());
    EXPECT_EQ(first, second);
}

TEST(ContentHashTest, DifferentInputProducesDifferentDigest) {
    const std::string a = "block of tokens A";
    const std::string b = "block of tokens B";
    EXPECT_NE(compute_content_hash(a.data(), a.size()), compute_content_hash(b.data(), b.size()));
}

TEST(ContentHashTest, EmptyInputIsWellDefined) {
    EXPECT_NO_THROW(compute_content_hash(nullptr, 0));
}

TEST(ContentHashTest, DifferentSeedProducesDifferentDigest) {
    const std::string data = "identical payload";
    EXPECT_NE(compute_content_hash(data.data(), data.size(), /*seed=*/0),
              compute_content_hash(data.data(), data.size(), /*seed=*/1));
}

TEST(ContentHashTest, DigestIsStableAcrossAllByteLengths) {
    // Exercise every tail-handling branch of the block/tail split (0..31 bytes).
    std::string data;
    for (size_t len = 0; len < 32; ++len) {
        data.push_back(static_cast<char>('a' + (len % 26)));
        const auto first = compute_content_hash(data.data(), data.size());
        const auto second = compute_content_hash(data.data(), data.size());
        EXPECT_EQ(first, second) << "length=" << len;
    }
}
