// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "continuous_batching/cache/abi_file_store_provider.hpp"
#include "continuous_batching/cache/abi_kv_cache_store.hpp"

using namespace ov::genai;

namespace {

constexpr size_t BLOCK_SIZE = 64;

std::unique_ptr<AbiKVCacheStore> open_store(size_t num_blocks) {
    const OvKVCacheStoreModuleInfo* module_info = get_file_store_provider_module_info();
    const OvKVCacheStoreProviderDescriptor* provider = module_info->get_provider(0);

    const std::string block_size_str = std::to_string(BLOCK_SIZE);
    const OvKVCacheProperty properties[] = {{"block_size_bytes", block_size_str.c_str()}};

    OvKVCacheStoreOpenOptions options{};
    options.directory = nullptr;  // system temp directory
    options.capacity_bytes = BLOCK_SIZE * num_blocks;
    options.properties = properties;
    options.property_count = 1;

    OvKVCacheStoreOpenResult result{};
    const OvKVCacheStatus status = provider->vtable->open(provider->self, nullptr, &options, &result);
    EXPECT_EQ(status, OV_KVCACHE_OK);
    EXPECT_EQ(result.status, OV_KVCACHE_OK);
    EXPECT_NE(result.vtable, nullptr);
    EXPECT_NE(result.store, nullptr);

    return std::make_unique<AbiKVCacheStore>(result.vtable, result.store, "kv", BLOCK_SIZE, num_blocks);
}

std::vector<uint8_t> make_block(uint8_t seed) {
    std::vector<uint8_t> data(BLOCK_SIZE);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(seed + i);
    }
    return data;
}

}  // namespace

TEST(TestAbiFileStoreProvider, ModuleHandshakeIsWellFormed) {
    const OvKVCacheStoreModuleInfo* module_info = get_file_store_provider_module_info();
    ASSERT_NE(module_info, nullptr);
    EXPECT_EQ(module_info->abi_major, static_cast<uint32_t>(OV_KVCACHE_STORE_ABI_MAJOR));
    EXPECT_LE(module_info->abi_minor, static_cast<uint32_t>(OV_KVCACHE_STORE_ABI_MINOR));
    ASSERT_EQ(module_info->provider_count, 1u);

    const OvKVCacheStoreProviderDescriptor* provider = module_info->get_provider(0);
    ASSERT_NE(provider, nullptr);
    EXPECT_STREQ(provider->name, "file");
    EXPECT_NE(provider->vtable, nullptr);
    EXPECT_EQ(provider->vtable->struct_bytes, sizeof(OvKVCacheStoreProviderVTable));
    EXPECT_EQ(module_info->get_provider(1), nullptr);
}

TEST(TestAbiFileStoreProvider, WriteThenReadRoundTripsBytesThroughTheAbi) {
    auto store = open_store(/*num_blocks=*/4);
    const auto expected = make_block(/*seed=*/7);

    EXPECT_FALSE(store->has_block(42));
    EXPECT_TRUE(store->write_block(42, expected));
    EXPECT_TRUE(store->has_block(42));

    std::vector<uint8_t> actual;
    ASSERT_TRUE(store->read_block(42, actual));
    EXPECT_EQ(actual, expected);
}

TEST(TestAbiFileStoreProvider, ReadReportsMissForUnknownHash) {
    auto store = open_store(/*num_blocks=*/4);
    std::vector<uint8_t> data;
    EXPECT_FALSE(store->read_block(1234, data));
}

TEST(TestAbiFileStoreProvider, DropBlockRemovesTheEntry) {
    auto store = open_store(/*num_blocks=*/4);
    ASSERT_TRUE(store->write_block(1, make_block(1)));
    ASSERT_TRUE(store->has_block(1));

    store->drop_block(1);

    EXPECT_FALSE(store->has_block(1));
    std::vector<uint8_t> data;
    EXPECT_FALSE(store->read_block(1, data));
}

TEST(TestAbiFileStoreProvider, WriteFailsWhenCapacityIsExhausted) {
    auto store = open_store(/*num_blocks=*/2);
    ASSERT_TRUE(store->write_block(1, make_block(1)));
    ASSERT_TRUE(store->write_block(2, make_block(2)));
    EXPECT_EQ(store->get_num_free_blocks(), 0u);

    EXPECT_FALSE(store->write_block(3, make_block(3)));
}

TEST(TestAbiFileStoreProvider, OverwritingAnExistingKeyReplacesItsContents) {
    auto store = open_store(/*num_blocks=*/4);
    ASSERT_TRUE(store->write_block(1, make_block(1)));
    ASSERT_TRUE(store->write_block(1, make_block(9)));

    std::vector<uint8_t> actual;
    ASSERT_TRUE(store->read_block(1, actual));
    EXPECT_EQ(actual, make_block(9));
}
