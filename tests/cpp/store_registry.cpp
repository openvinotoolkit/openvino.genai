// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "continuous_batching/cache/abi_kv_cache_store.hpp"
#include "continuous_batching/cache/default_file_store.hpp"
#include "continuous_batching/cache/kv_cache_disk_layout.hpp"
#include "continuous_batching/cache/store_registry.hpp"
#include "store_plugin_locator.hpp"

using namespace ov::genai;

namespace {

constexpr size_t BLOCK_SIZE = 32;
constexpr size_t NUM_BLOCKS = 4;

OffloadSettings make_config() {
    OffloadSettings config;
    config.kv_cache_dir = std::filesystem::temp_directory_path().string();
    config.capacity_bytes = BLOCK_SIZE * NUM_BLOCKS;
    return config;
}

KVCacheDiskLayout make_layout() {
    return KVCacheDiskLayout({BLOCK_SIZE}, {0});
}

std::vector<uint8_t> make_block(uint8_t seed) {
    std::vector<uint8_t> data(BLOCK_SIZE);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(seed + i);
    }
    return data;
}

// Tests that name a block by one number give it a digest made of that number, which every store takes.
PrefixKey hash_key(size_t hash) {
    return PrefixKey{hash, ContentHash{hash, hash}};
}

bool has_hash(const IKVCacheStore& store, size_t hash) {
    return store.has_blocks({hash_key(hash)})[0];
}

bool write_hash(IKVCacheStore& store, size_t hash, const std::vector<uint8_t>& data) {
    return store.write_blocks({hash_key(hash)}, {&data})[0];
}

bool read_hash(const IKVCacheStore& store, size_t hash, std::vector<uint8_t>& data) {
    std::vector<std::vector<uint8_t>> blocks;
    if (!store.read_blocks({hash_key(hash)}, blocks)[0]) {
        return false;
    }
    data = std::move(blocks[0]);
    return true;
}

void drop_hash(IKVCacheStore& store, size_t hash) {
    store.invalidate(hash_key(hash));
}

bool is_plugin(const IKVCacheStore& store) {
    return dynamic_cast<const AbiKVCacheStore*>(&store) != nullptr;
}

/// How many more blocks the store has room for without evicting; informational.
size_t free_blocks(const IKVCacheStore& store) {
    if (const auto* plugin = dynamic_cast<const AbiKVCacheStore*>(&store)) {
        return plugin->get_num_free_blocks();
    }
    return dynamic_cast<const DefaultFileGroupStore&>(store).get_num_free_blocks();
}

void expect_store_is_usable(IKVCacheStore& store) {
    EXPECT_EQ(store.block_bytes(), BLOCK_SIZE);
    EXPECT_EQ(store.capacity_blocks_hint(), NUM_BLOCKS);
    EXPECT_FALSE(has_hash(store, 1));

    const auto expected = make_block(/*seed=*/5);
    ASSERT_TRUE(write_hash(store, 1, expected));
    ASSERT_TRUE(has_hash(store, 1));

    std::vector<uint8_t> actual;
    ASSERT_TRUE(read_hash(store, 1, actual));
    EXPECT_EQ(actual, expected);
}

using ov::genai::test::find_sample_plugin_library;
using ov::genai::test::find_self_as_plugin_library;

}  // namespace

TEST(TestStoreRegistry, DefaultConfigUsesTheInTreeFileStoreDirectly) {
    auto store = open_kv_cache_store(make_layout(), make_config(), "CPU", "kv");
    ASSERT_NE(store, nullptr);
    expect_store_is_usable(*store);
}

TEST(TestStoreRegistry, NonexistentPluginLibraryThrows) {
    OffloadSettings config = make_config();
    config.store_library_path = "this_library_does_not_exist_anywhere.dll";

    EXPECT_THROW(open_kv_cache_store(make_layout(), config, "CPU", "kv"), ov::Exception);
}

#ifdef _WIN32
TEST(TestStoreRegistry, LibraryMissingTheEntrySymbolThrows) {
    OffloadSettings config = make_config();
    // A real, always-present system DLL that definitely does not export our ABI entry symbol.
    config.store_library_path = "kernel32.dll";

    EXPECT_THROW(open_kv_cache_store(make_layout(), config, "CPU", "kv"), ov::Exception);
}
#endif

TEST(TestStoreRegistry, LoadsAndUsesAGenuinelySeparateDynamicallyLoadedPlugin) {
    const std::optional<std::string> library_path = find_self_as_plugin_library();
    if (!library_path.has_value()) {
        GTEST_SKIP() << "Could not locate openvino_genai's own shared library next to the test binary";
    }

    OffloadSettings config = make_config();
    config.store_library_path = *library_path;
    config.store_provider_name = "file";

    auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
    ASSERT_NE(store, nullptr);
    ASSERT_TRUE(is_plugin(*store));
    expect_store_is_usable(*store);
}

TEST(TestStoreRegistry, LoadsAndUsesTheStandaloneSampleThirdPartyPlugin) {
    const std::optional<std::string> library_path = find_sample_plugin_library();
    if (!library_path.has_value()) {
        GTEST_SKIP() << "samples/c/kv_cache_store_plugin was not built (requires -DENABLE_SAMPLES=ON)";
    }

    OffloadSettings config = make_config();
    config.store_library_path = *library_path;
    config.store_provider_name = "sample-in-memory";

    auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
    ASSERT_NE(store, nullptr);
    ASSERT_TRUE(is_plugin(*store));
    expect_store_is_usable(*store);
}

// The sample plugin implements `on_restore_finish` and prints what the host tells it; a provider without it takes none.
TEST(TestStoreRegistry, SampleThirdPartyPluginTakesRestoreReportsAndTheFileProviderDoesNot) {
    const std::optional<std::string> library_path = find_sample_plugin_library();
    if (!library_path.has_value()) {
        GTEST_SKIP() << "samples/c/kv_cache_store_plugin was not built (requires -DENABLE_SAMPLES=ON)";
    }

    OffloadSettings config = make_config();
    config.store_library_path = *library_path;
    config.store_provider_name = "sample-in-memory";

    auto store = open_kv_cache_store(make_layout(), config, "CPU", "restore_stats_test");
    ASSERT_NE(store, nullptr);
    ASSERT_TRUE(is_plugin(*store));

    testing::internal::CaptureStderr();
    const bool taken = store->report_restore_stats(RestoreStats{16384, 1024, 4096, 5000});
    const std::string output = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(taken);
    EXPECT_NE(output.find("sample: restore prompt=16384 device_hit=1024 store_hit=4096 reused=5000"), std::string::npos)
        << output;

    const std::optional<std::string> self_path = find_self_as_plugin_library();
    if (!self_path.has_value()) {
        return;
    }
    OffloadSettings file_config = make_config();
    file_config.store_library_path = *self_path;
    file_config.store_provider_name = "file";
    auto file_store = open_kv_cache_store(make_layout(), file_config, "CPU", "kv");
    ASSERT_NE(file_store, nullptr);
    ASSERT_TRUE(is_plugin(*file_store));
    EXPECT_FALSE(file_store->report_restore_stats(RestoreStats{1, 0, 0, 0}));
}

// The ABI opens one store per namespace, so blocks of different groups must not be counted together.
TEST(TestStoreRegistry, SampleThirdPartyPluginKeepsGroupsInSeparateStores) {
    const std::optional<std::string> library_path = find_sample_plugin_library();
    if (!library_path.has_value()) {
        GTEST_SKIP() << "samples/c/kv_cache_store_plugin was not built (requires -DENABLE_SAMPLES=ON)";
    }

    OffloadSettings config = make_config();
    config.store_library_path = *library_path;
    config.store_provider_name = "sample-in-memory";

    auto first = open_kv_cache_store(make_layout(), config, "CPU", "namespace_test_first");
    auto second = open_kv_cache_store(make_layout(), config, "CPU", "namespace_test_second");
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    ASSERT_TRUE(is_plugin(*first));
    ASSERT_TRUE(is_plugin(*second));

    ASSERT_TRUE(write_hash(*first, 7, make_block(/*seed=*/1)));
    EXPECT_EQ(free_blocks(*first), NUM_BLOCKS - 1);
    EXPECT_EQ(free_blocks(*second), NUM_BLOCKS);
}

namespace {

// A KV group of one-block slots and a linear-attention group of slots twice that size, as a hybrid model has.
std::vector<StoreGroupSpec> make_group_specs(const std::string& tag, uint64_t kv_capacity_bytes, uint64_t la_capacity_bytes) {
    return {StoreGroupSpec{"kv", "kv_" + tag, make_layout(), 4, kv_capacity_bytes},
            StoreGroupSpec{"la", "la_" + tag, KVCacheDiskLayout({2 * BLOCK_SIZE}, {0}), 16, la_capacity_bytes}};
}

OffloadSettings make_plugin_config(const std::string& library_path, const std::string& provider_name) {
    OffloadSettings config = make_config();
    config.store_library_path = library_path;
    config.store_provider_name = provider_name;
    return config;
}

std::vector<uint8_t> make_la_block(uint8_t seed) {
    std::vector<uint8_t> data(2 * BLOCK_SIZE);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(seed + i);
    }
    return data;
}

}  // namespace

TEST(TestStoreRegistry, GroupedStoreOfTheDefaultBackendHasAnIndependentStorePerGroup) {
    const auto specs = make_group_specs("default", NUM_BLOCKS * BLOCK_SIZE, NUM_BLOCKS * 2 * BLOCK_SIZE);
    auto stores = open_grouped_kv_cache_store(specs, make_config(), "CPU", "kv+la");
    ASSERT_NE(stores, nullptr);
    ASSERT_EQ(stores->group_count(), 2u);
    auto kv = stores->group(0);
    auto la = stores->group(1);

    EXPECT_EQ(kv->block_bytes(), BLOCK_SIZE);
    EXPECT_EQ(la->block_bytes(), 2 * BLOCK_SIZE);
    ASSERT_TRUE(write_hash(*kv, 1, make_block(/*seed=*/3)));
    EXPECT_FALSE(has_hash(*la, 1));
    EXPECT_FALSE(kv->wants_prefix_tokens());
}

// A provider without OV_KVCACHE_CAP_MULTI_GROUP is opened once per group, under the namespace of the group.
TEST(TestStoreRegistry, GroupedStoreOpensAPluginWithoutTheMultiGroupCapabilityOncePerGroup) {
    const std::optional<std::string> library_path = find_sample_plugin_library();
    if (!library_path.has_value()) {
        GTEST_SKIP() << "samples/c/kv_cache_store_plugin was not built (requires -DENABLE_SAMPLES=ON)";
    }

    const auto specs = make_group_specs("per_group", NUM_BLOCKS * BLOCK_SIZE, NUM_BLOCKS * 2 * BLOCK_SIZE);
    auto stores = open_grouped_kv_cache_store(specs, make_plugin_config(*library_path, "sample-in-memory"), "CPU", "kv+la:per_group");
    ASSERT_EQ(stores->group_count(), 2u);
    auto kv = stores->group(0);
    auto la = stores->group(1);
    ASSERT_TRUE(is_plugin(*kv));
    ASSERT_TRUE(is_plugin(*la));

    ASSERT_TRUE(write_hash(*kv, 1, make_block(/*seed=*/3)));
    EXPECT_EQ(free_blocks(*kv), NUM_BLOCKS - 1);
    EXPECT_EQ(free_blocks(*la), NUM_BLOCKS);
    EXPECT_FALSE(kv->wants_prefix_tokens());
}

// All groups share one instance and its byte budget, so a block of one group can push out a block of the other.
TEST(TestStoreRegistry, GroupedStoreOpensAMultiGroupPluginOnceForAllGroups) {
    const std::optional<std::string> library_path = find_sample_plugin_library();
    if (!library_path.has_value()) {
        GTEST_SKIP() << "samples/c/kv_cache_store_plugin was not built (requires -DENABLE_SAMPLES=ON)";
    }

    // Room for two KV blocks and one linear-attention block, the budget of both groups together.
    const auto specs = make_group_specs("multi_group", 2 * BLOCK_SIZE, 2 * BLOCK_SIZE);
    auto stores = open_grouped_kv_cache_store(specs,
                                              make_plugin_config(*library_path, "sample-self-evict-memory"),
                                              "CPU",
                                              "kv+la:multi_group");
    ASSERT_EQ(stores->group_count(), 2u);
    auto kv = stores->group(0);
    auto la = stores->group(1);
    ASSERT_TRUE(is_plugin(*kv));
    ASSERT_TRUE(is_plugin(*la));
    EXPECT_TRUE(kv->wants_prefix_tokens());
    EXPECT_TRUE(la->wants_prefix_tokens());

    ASSERT_TRUE(write_hash(*la, 9, make_la_block(/*seed=*/1)));
    ASSERT_TRUE(write_hash(*kv, 1, make_block(/*seed=*/2)));
    ASSERT_TRUE(write_hash(*kv, 2, make_block(/*seed=*/3)));
    EXPECT_TRUE(has_hash(*la, 9));

    // The instance is full: this block evicts the oldest one, which is the linear-attention block of the other group.
    ASSERT_TRUE(write_hash(*kv, 3, make_block(/*seed=*/4)));
    EXPECT_FALSE(has_hash(*la, 9));
    EXPECT_TRUE(has_hash(*kv, 3));

    // The group is part of the key: the same hash in the other group is another block.
    ASSERT_TRUE(write_hash(*la, 1, make_la_block(/*seed=*/5)));
    std::vector<uint8_t> kv_data;
    ASSERT_TRUE(read_hash(*kv, 3, kv_data));
    EXPECT_EQ(kv_data, make_block(/*seed=*/4));
}

// The persistent provider keeps a block in a file under OffloadSettings::kv_cache_dir, which is what lets another process
// read it back; the cross-process round trip itself is checked by verify_chunk_store.py (write/read phases).
TEST(TestStoreRegistry, SamplePersistentProviderKeepsBlocksAsFilesOnDisk) {
    const std::optional<std::string> library_path = find_sample_plugin_library();
    if (!library_path.has_value()) {
        GTEST_SKIP() << "samples/c/kv_cache_store_plugin was not built (requires -DENABLE_SAMPLES=ON)";
    }

    const auto directory = std::filesystem::temp_directory_path() / "openvino_genai_persistent_store_test";
    std::filesystem::remove_all(directory);
    OffloadSettings config = make_config();
    config.kv_cache_dir = directory.string();
    config.store_library_path = *library_path;
    config.store_provider_name = "sample-persistent-file";

    {
        auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        ASSERT_NE(store, nullptr);
        ASSERT_TRUE(is_plugin(*store));
        expect_store_is_usable(*store);

        size_t block_files = 0;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
            block_files += entry.is_regular_file() && entry.path().extension() == ".blk" ? 1 : 0;
        }
        EXPECT_EQ(block_files, 1u);
    }

    // Without a directory the provider refuses to open, and that is an error rather than a silent fallback.
    config.kv_cache_dir.clear();
    EXPECT_THROW(open_kv_cache_store(make_layout(), config, "CPU", "kv"), ov::Exception);
    std::filesystem::remove_all(directory);
}

namespace {

size_t count_block_files(const std::filesystem::path& directory) {
    size_t block_files = 0;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
        block_files += entry.is_regular_file() && entry.path().extension() == ".blk" ? 1 : 0;
    }
    return block_files;
}

}  // namespace

TEST(TestStoreRegistry, SamplePersistentProviderKeepsOrPurgesBlocksAccordingToReuseSsdCache) {
    const std::optional<std::string> library_path = find_sample_plugin_library();
    if (!library_path.has_value()) {
        GTEST_SKIP() << "samples/c/kv_cache_store_plugin was not built (requires -DENABLE_SAMPLES=ON)";
    }

    const auto directory = std::filesystem::temp_directory_path() / "openvino_genai_reuse_ssd_cache_test";
    std::filesystem::remove_all(directory);
    OffloadSettings config = make_config();
    config.kv_cache_dir = directory.string();
    config.store_library_path = *library_path;
    config.store_provider_name = "sample-persistent-file";

    {
        auto first = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        ASSERT_NE(first, nullptr);
        ASSERT_TRUE(is_plugin(*first));
        ASSERT_TRUE(write_hash(*first, 1, make_block(/*seed=*/5)));
    }
    ASSERT_EQ(count_block_files(directory), 1u);

    ASSERT_TRUE(config.reuse_ssd_cache);
    {
        auto reused = open_kv_cache_store(make_layout(), config, "CPU", "kv");
        ASSERT_NE(reused, nullptr);
        ASSERT_TRUE(is_plugin(*reused));
        EXPECT_TRUE(has_hash(*reused, 1));
    }

    config.reuse_ssd_cache = false;
    auto purged = open_kv_cache_store(make_layout(), config, "CPU", "kv");
    ASSERT_NE(purged, nullptr);
    ASSERT_TRUE(is_plugin(*purged));
    EXPECT_FALSE(has_hash(*purged, 1));
    EXPECT_EQ(free_blocks(*purged), NUM_BLOCKS);
    EXPECT_EQ(count_block_files(directory), 0u);
    std::filesystem::remove_all(directory);
}

// A purge is asked of the plugin right after the open, and a plugin that cannot do it fails the open.
TEST(TestStoreRegistry, PluginIsPurgedWhenTheCacheOfAnEarlierRunIsNotReused) {
    const std::optional<std::string> library_path = find_sample_plugin_library();
    if (!library_path.has_value()) {
        GTEST_SKIP() << "samples/c/kv_cache_store_plugin was not built (requires -DENABLE_SAMPLES=ON)";
    }

    OffloadSettings config = make_plugin_config(*library_path, "sample-in-memory");
    auto first = open_kv_cache_store(make_layout(), config, "CPU", "purge_test");
    ASSERT_TRUE(write_hash(*first, 1, make_block(/*seed=*/1)));

    auto reused = open_kv_cache_store(make_layout(), config, "CPU", "purge_test");
    EXPECT_TRUE(has_hash(*reused, 1));

    config.reuse_ssd_cache = false;
    auto purged = open_kv_cache_store(make_layout(), config, "CPU", "purge_test");
    EXPECT_FALSE(has_hash(*purged, 1));
}

TEST(TestStoreRegistry, StorePropertiesReachThePluginUnchanged) {
    const std::optional<std::string> library_path = find_sample_plugin_library();
    if (!library_path.has_value()) {
        GTEST_SKIP() << "samples/c/kv_cache_store_plugin was not built (requires -DENABLE_SAMPLES=ON)";
    }

    // Names the host used to set itself are the user's to use now.
    OffloadSettings config = make_plugin_config(*library_path, "sample-in-memory");
    config.store_properties = {{"device", "x"}, {"reuse_ssd_cache", "false"}, {"vendor_key", "1"}};
    EXPECT_NO_THROW(open_kv_cache_store(make_layout(), config, "CPU", "properties_test"));
}

namespace {

PrefixKey digest_key(uint64_t number) {
    return PrefixKey{static_cast<size_t>(number), ContentHash{number, number * 7 + 1}};
}

std::filesystem::path fresh_directory(const std::string& name) {
    const auto directory = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    return directory;
}

OffloadSettings make_default_store_config(const std::filesystem::path& directory) {
    OffloadSettings config = make_config();
    config.kv_cache_dir = directory.string();
    return config;
}

bool write_one(IKVCacheStore& store, const PrefixKey& key, const std::vector<uint8_t>& data) {
    return store.write_blocks({key}, {&data})[0];
}

bool has_one(const IKVCacheStore& store, const PrefixKey& key) {
    return store.has_blocks({key})[0];
}

std::filesystem::path find_slots_file(const std::filesystem::path& directory) {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
        if (entry.path().extension() == ".slots") {
            return entry.path();
        }
    }
    return {};
}

}  // namespace

// A namespace with a model identity (it contains ':') is kept in kv_cache_dir for the next run.
TEST(TestStoreRegistry, DefaultStoreKeepsBlocksOfAModelAcrossReopening) {
    const auto directory = fresh_directory("openvino_genai_default_store_reopen");
    const OffloadSettings config = make_default_store_config(directory);
    {
        auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv:model_a");
        ASSERT_TRUE(store->is_persistent());
        ASSERT_TRUE(write_one(*store, digest_key(1), make_block(1)));
        ASSERT_TRUE(write_one(*store, digest_key(2), make_block(2)));
    }

    auto reopened = open_kv_cache_store(make_layout(), config, "CPU", "kv:model_a");
    EXPECT_TRUE(has_one(*reopened, digest_key(1)));
    EXPECT_EQ(free_blocks(*reopened), NUM_BLOCKS - 2);
    std::vector<std::vector<uint8_t>> data;
    ASSERT_TRUE(reopened->read_blocks({digest_key(2)}, data)[0]);
    EXPECT_EQ(data[0], make_block(2));

    auto other_model = open_kv_cache_store(make_layout(), config, "CPU", "kv:model_b");
    EXPECT_FALSE(has_one(*other_model, digest_key(1)));
    reopened.reset();
    other_model.reset();
    std::filesystem::remove_all(directory);
}

TEST(TestStoreRegistry, DefaultStoreDropsTheCacheOfAnEarlierRunWhenNotReused) {
    const auto directory = fresh_directory("openvino_genai_default_store_purge");
    OffloadSettings config = make_default_store_config(directory);
    {
        auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv:model_a");
        ASSERT_TRUE(write_one(*store, digest_key(1), make_block(1)));
    }

    config.reuse_ssd_cache = false;
    auto purged = open_kv_cache_store(make_layout(), config, "CPU", "kv:model_a");
    EXPECT_FALSE(has_one(*purged, digest_key(1)));
    EXPECT_EQ(free_blocks(*purged), NUM_BLOCKS);
    purged.reset();
    std::filesystem::remove_all(directory);
}

TEST(TestStoreRegistry, DefaultStoreEvictsTheLeastRecentlyUsedBlock) {
    const auto directory = fresh_directory("openvino_genai_default_store_lru");
    auto store = open_kv_cache_store(make_layout(), make_default_store_config(directory), "CPU", "kv:model_a");
    for (uint64_t number = 1; number <= NUM_BLOCKS; ++number) {
        ASSERT_TRUE(write_one(*store, digest_key(number), make_block(static_cast<uint8_t>(number))));
    }

    std::vector<std::vector<uint8_t>> data;
    ASSERT_TRUE(store->read_blocks({digest_key(1)}, data)[0]);  // a hit makes block 1 the most recent
    ASSERT_TRUE(write_one(*store, digest_key(5), make_block(5)));      // never fails for lack of capacity

    EXPECT_TRUE(has_one(*store, digest_key(1)));
    EXPECT_FALSE(has_one(*store, digest_key(2)));
    EXPECT_TRUE(has_one(*store, digest_key(5)));
    store.reset();
    std::filesystem::remove_all(directory);
}

TEST(TestStoreRegistry, DefaultStoreRestoresTheEvictionOrderAfterReopening) {
    const auto directory = fresh_directory("openvino_genai_default_store_order");
    const OffloadSettings config = make_default_store_config(directory);
    {
        auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv:model_a");
        for (uint64_t number = 1; number <= NUM_BLOCKS; ++number) {
            ASSERT_TRUE(write_one(*store, digest_key(number), make_block(static_cast<uint8_t>(number))));
        }
    }

    auto reopened = open_kv_cache_store(make_layout(), config, "CPU", "kv:model_a");
    ASSERT_TRUE(write_one(*reopened, digest_key(5), make_block(5)));
    EXPECT_FALSE(has_one(*reopened, digest_key(1)));  // written first, so least recent
    EXPECT_TRUE(has_one(*reopened, digest_key(2)));
    reopened.reset();
    std::filesystem::remove_all(directory);
}

TEST(TestStoreRegistry, DefaultStoreTreatsADamagedBlockAsMissing) {
    const auto directory = fresh_directory("openvino_genai_default_store_damage");
    const OffloadSettings config = make_default_store_config(directory);
    {
        auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv:model_a");
        ASSERT_TRUE(write_one(*store, digest_key(1), make_block(1)));
    }
    {
        const auto slots = find_slots_file(directory);
        ASSERT_FALSE(slots.empty());
        std::fstream file(slots, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(64 + 3);  // inside the payload of the first slot, behind its 64 byte header
        const char flipped = 0x5A;
        file.write(&flipped, 1);
    }

    auto reopened = open_kv_cache_store(make_layout(), config, "CPU", "kv:model_a");
    ASSERT_TRUE(has_one(*reopened, digest_key(1)));
    std::vector<std::vector<uint8_t>> data;
    EXPECT_FALSE(reopened->read_blocks({digest_key(1)}, data)[0]);
    EXPECT_FALSE(has_one(*reopened, digest_key(1)));
    EXPECT_EQ(free_blocks(*reopened), NUM_BLOCKS);
    reopened.reset();
    std::filesystem::remove_all(directory);
}

TEST(TestStoreRegistry, DefaultStoreRefusesToOpenADirectoryInUse) {
    const auto directory = fresh_directory("openvino_genai_default_store_lock");
    const OffloadSettings config = make_default_store_config(directory);
    auto first = open_kv_cache_store(make_layout(), config, "CPU", "kv:model_a");
    EXPECT_THROW(open_kv_cache_store(make_layout(), config, "CPU", "kv:model_a"), ov::Exception);
    EXPECT_NO_THROW(open_kv_cache_store(make_layout(), config, "CPU", "kv:model_b"));
    first.reset();
    EXPECT_NO_THROW(open_kv_cache_store(make_layout(), config, "CPU", "kv:model_a"));
    std::filesystem::remove_all(directory);
}

TEST(TestStoreRegistry, PersistentDefaultStoreSkipsABlockWithoutADigest) {
    const auto directory = fresh_directory("openvino_genai_default_store_no_digest");
    auto store = open_kv_cache_store(make_layout(), make_default_store_config(directory), "CPU", "kv:model_a");
    EXPECT_FALSE(write_one(*store, PrefixKey{5, std::nullopt}, make_block(5)));
    EXPECT_FALSE(has_one(*store, PrefixKey{5, std::nullopt}));
    store.reset();
    std::filesystem::remove_all(directory);
}

TEST(TestStoreRegistry, DefaultStoreWithoutAModelIdentityLeavesNothingBehind) {
    const auto directory = fresh_directory("openvino_genai_default_store_private");
    {
        auto store = open_kv_cache_store(make_layout(), make_default_store_config(directory), "CPU", "kv");
        EXPECT_FALSE(store->is_persistent());
        ASSERT_TRUE(write_one(*store, PrefixKey{5, std::nullopt}, make_block(5)));
    }
    EXPECT_TRUE(find_slots_file(directory).empty());
    std::filesystem::remove_all(directory);
}

TEST(TestStoreRegistry, DefaultStoreRequiresAnExistingKvCacheDirectory) {
    OffloadSettings config = make_config();
    config.kv_cache_dir.clear();
    EXPECT_THROW(open_kv_cache_store(make_layout(), config, "CPU", "kv:model_a"), ov::Exception);
    config.kv_cache_dir = (std::filesystem::temp_directory_path() / "openvino_genai_default_store_missing").string();
    EXPECT_THROW(open_kv_cache_store(make_layout(), config, "CPU", "kv:model_a"), ov::Exception);
}