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

TEST(TestAbiFileStoreProvider, KeysDifferingOnlyInTheHighHalfStayDistinct) {
    const OvKVCacheStoreModuleInfo* module_info = get_file_store_provider_module_info();
    const OvKVCacheStoreProviderDescriptor* provider = module_info->get_provider(0);

    const std::string block_size_str = std::to_string(BLOCK_SIZE);
    const OvKVCacheProperty properties[] = {{"block_size_bytes", block_size_str.c_str()}};
    OvKVCacheStoreOpenOptions options{};
    options.capacity_bytes = BLOCK_SIZE * 4;
    options.properties = properties;
    options.property_count = 1;
    OvKVCacheStoreOpenResult opened{};
    ASSERT_EQ(provider->vtable->open(provider->self, nullptr, &options, &opened), OV_KVCACHE_OK);

    auto make_key = [](uint64_t hi) {
        OvKVCacheBlockKey key{};
        key.hash = OvKVCacheContentHash{/*lo=*/7, hi};
        return key;
    };
    const auto first_payload = make_block(1);
    const auto second_payload = make_block(2);
    OvKVCacheStoreWriteRequest writes[2]{};
    writes[0].key = make_key(100);
    writes[0].source = first_payload.data();
    writes[0].bytes = first_payload.size();
    writes[1].key = make_key(200);
    writes[1].source = second_payload.data();
    writes[1].bytes = second_payload.size();
    OvKVCacheStoreWriteResult write_results[2]{};
    ASSERT_EQ(opened.vtable->write_blocks(opened.store, writes, write_results, 2), OV_KVCACHE_OK);

    OvKVCacheStoreStats stats{};
    ASSERT_EQ(opened.vtable->stats(opened.store, &stats), OV_KVCACHE_OK);
    EXPECT_EQ(stats.blocks, 2u) << "two keys with the same low half must occupy two blocks";

    for (const auto& [hi, expected] : {std::pair<uint64_t, std::vector<uint8_t>>{100, first_payload},
                                       std::pair<uint64_t, std::vector<uint8_t>>{200, second_payload}}) {
        std::vector<uint8_t> actual(BLOCK_SIZE);
        OvKVCacheStoreReadRequest read{};
        read.key = make_key(hi);
        read.target.kind = OV_KVCACHE_TARGET_HOST;
        read.target.base = actual.data();
        read.target.bytes = actual.size();
        OvKVCacheStoreReadResult read_result{};
        ASSERT_EQ(opened.vtable->read_blocks(opened.store, &read, &read_result, 1), OV_KVCACHE_OK);
        EXPECT_EQ(actual, expected) << "hi=" << hi;
    }
    opened.vtable->destroy(opened.store);
}

TEST(TestAbiFileStoreProvider, SessionSnapshotNamespacesAreRejectedBecauseTheStoreIsEphemeral) {
    const OvKVCacheStoreModuleInfo* module_info = get_file_store_provider_module_info();
    const OvKVCacheStoreProviderDescriptor* provider = module_info->get_provider(0);

    const std::string block_size_str = std::to_string(BLOCK_SIZE);
    const OvKVCacheProperty properties[] = {{"block_size_bytes", block_size_str.c_str()}};
    OvKVCacheStoreOpenOptions options{};
    options.capacity_bytes = BLOCK_SIZE * 4;
    options.properties = properties;
    options.property_count = 1;

    const std::string snapshot_group = "session_snapshot_kv:some/path.bin";
    const OvKVCacheNamespace snapshot_ns{reinterpret_cast<const uint8_t*>(snapshot_group.data()), snapshot_group.size()};
    OvKVCacheStoreOpenResult rejected{};
    EXPECT_EQ(provider->vtable->open(provider->self, &snapshot_ns, &options, &rejected), OV_KVCACHE_ERR_UNSUPPORTED);
    EXPECT_EQ(rejected.store, nullptr);

    const std::string kv_group = "kv";
    const OvKVCacheNamespace kv_ns{reinterpret_cast<const uint8_t*>(kv_group.data()), kv_group.size()};
    OvKVCacheStoreOpenResult accepted{};
    ASSERT_EQ(provider->vtable->open(provider->self, &kv_ns, &options, &accepted), OV_KVCACHE_OK);
    accepted.vtable->destroy(accepted.store);
}

namespace {
std::vector<std::pair<OvKVCacheLogLevel, std::string>> g_captured_logs;
void capture_log(OvKVCacheLogLevel level, const char* message) {
    g_captured_logs.emplace_back(level, message);
}
}  // namespace

TEST(TestAbiFileStoreProvider, OpenAcceptsMatchingGroupInfoAndForwardsLogsToTheHostCallback) {
    const OvKVCacheStoreProviderDescriptor* provider = get_file_store_provider_module_info()->get_provider(0);
    const std::string block_size_str = std::to_string(BLOCK_SIZE);
    const OvKVCacheProperty properties[] = {{"block_size_bytes", block_size_str.c_str()}};
    const std::string group = "kv";
    const OvKVCacheGroupInfo groups[] = {{group.c_str(), static_cast<uint32_t>(group.size()), 32, BLOCK_SIZE}};
    const OvKVCacheNamespace ns{reinterpret_cast<const uint8_t*>(group.data()), group.size()};

    OvKVCacheStoreOpenOptions options{};
    options.capacity_bytes = BLOCK_SIZE * 4;
    options.properties = properties;
    options.property_count = 1;
    options.groups = groups;
    options.group_count = 1;
    options.witness_bytes_max = 16;
    options.log = &capture_log;

    g_captured_logs.clear();
    OvKVCacheStoreOpenResult result{};
    ASSERT_EQ(provider->vtable->open(provider->self, &ns, &options, &result), OV_KVCACHE_OK);
    result.vtable->destroy(result.store);
    ASSERT_FALSE(g_captured_logs.empty());
    EXPECT_EQ(g_captured_logs.front().first, static_cast<OvKVCacheLogLevel>(OV_KVCACHE_LOG_DEBUG));
}

TEST(TestAbiFileStoreProvider, OpenRejectsGroupInfoWhoseBlockBytesDisagreeWithTheStoreBlockSize) {
    const OvKVCacheStoreProviderDescriptor* provider = get_file_store_provider_module_info()->get_provider(0);
    const std::string block_size_str = std::to_string(BLOCK_SIZE);
    const OvKVCacheProperty properties[] = {{"block_size_bytes", block_size_str.c_str()}};
    const std::string group = "kv";
    const OvKVCacheGroupInfo groups[] = {{group.c_str(), static_cast<uint32_t>(group.size()), 32, BLOCK_SIZE + 1}};
    const OvKVCacheNamespace ns{reinterpret_cast<const uint8_t*>(group.data()), group.size()};

    OvKVCacheStoreOpenOptions options{};
    options.capacity_bytes = BLOCK_SIZE * 4;
    options.properties = properties;
    options.property_count = 1;
    options.groups = groups;
    options.group_count = 1;

    OvKVCacheStoreOpenResult result{};
    EXPECT_EQ(provider->vtable->open(provider->self, &ns, &options, &result), OV_KVCACHE_ERR_INVALID_ARG);
    EXPECT_EQ(result.store, nullptr);
}

TEST(TestAbiFileStoreProvider, MissLeavesTheCallersBufferUntouched) {
    auto store = open_store(/*num_blocks=*/4);
    std::vector<uint8_t> data = {1, 2, 3};
    EXPECT_FALSE(store->read_block(1234, data));
    EXPECT_EQ(data, (std::vector<uint8_t>{1, 2, 3}));
}

TEST(TestAbiFileStoreProvider, MalformedBlockSizeIsReportedAsInvalidArgInsteadOfThrowing) {
    const OvKVCacheStoreModuleInfo* module_info = get_file_store_provider_module_info();
    const OvKVCacheStoreProviderDescriptor* provider = module_info->get_provider(0);

    for (const char* bad_value : {"not-a-number", "99999999999999999999999999"}) {
        const OvKVCacheProperty properties[] = {{"block_size_bytes", bad_value}};
        OvKVCacheStoreOpenOptions options{};
        options.capacity_bytes = BLOCK_SIZE * 4;
        options.properties = properties;
        options.property_count = 1;

        OvKVCacheStoreOpenResult result{};
        OvKVCacheStatus status = OV_KVCACHE_OK;
        EXPECT_NO_THROW(status = provider->vtable->open(provider->self, nullptr, &options, &result)) << bad_value;
        EXPECT_EQ(status, OV_KVCACHE_ERR_INVALID_ARG) << bad_value;
        EXPECT_EQ(result.store, nullptr) << bad_value;
    }
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
