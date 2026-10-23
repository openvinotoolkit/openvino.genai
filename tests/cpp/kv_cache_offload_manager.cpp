// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <filesystem>
#include <numeric>

#include "continuous_batching/cache/kv_cache_offload_manager.hpp"

using namespace ov::genai;

namespace {

constexpr size_t KEY_BLOCK_BYTES = 96;
constexpr size_t VALUE_BLOCK_BYTES = 64;
constexpr size_t NUM_LAYERS = 3;
constexpr size_t SLOT_BYTES = NUM_LAYERS * (KEY_BLOCK_BYTES + VALUE_BLOCK_BYTES);

KVCacheDiskLayout make_layout() {
    return KVCacheDiskLayout(std::vector<size_t>(NUM_LAYERS, KEY_BLOCK_BYTES),
                             std::vector<size_t>(NUM_LAYERS, VALUE_BLOCK_BYTES));
}

CacheOffloadConfig make_config(size_t num_slots) {
    CacheOffloadConfig config;
    config.capacity_bytes = num_slots * SLOT_BYTES;
    return config;
}

std::vector<uint8_t> make_pattern(uint8_t seed) {
    std::vector<uint8_t> data(SLOT_BYTES);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(seed + i);
    }
    return data;
}

}  // namespace

TEST(TestKVCacheOffloadManager, DerivesSlotCountFromCapacity) {
    // Capacity that is not a multiple of the slot size must be rounded down.
    KVCacheOffloadManager manager(make_layout(), make_config(4), "CPU");

    EXPECT_EQ(manager.get_slot_size(), SLOT_BYTES);
    EXPECT_EQ(manager.get_num_slots(), 4);
    EXPECT_EQ(manager.get_num_free_slots(), 4);
}

TEST(TestKVCacheOffloadManager, CreatesAndRemovesRunSpecificFile) {
    std::filesystem::path file_path;
    {
        KVCacheOffloadManager manager(make_layout(), make_config(2), "CPU");
        file_path = manager.get_file_path();

        ASSERT_FALSE(file_path.empty());
        EXPECT_TRUE(std::filesystem::exists(file_path));
        EXPECT_EQ(std::filesystem::file_size(file_path), 2 * SLOT_BYTES);
    }
    EXPECT_FALSE(std::filesystem::exists(file_path));
}

TEST(TestKVCacheOffloadManager, SlotRoundTrip) {
    KVCacheOffloadManager manager(make_layout(), make_config(2), "CPU");
    const auto expected = make_pattern(7);

    manager.write_slot(1, expected);

    std::vector<uint8_t> actual;
    manager.read_slot(1, actual);
    EXPECT_EQ(actual, expected);
}

TEST(TestKVCacheOffloadManager, SlotsDoNotOverlap) {
    KVCacheOffloadManager manager(make_layout(), make_config(3), "CPU");
    const auto first = make_pattern(1);
    const auto second = make_pattern(200);

    manager.write_slot(0, first);
    manager.write_slot(2, second);

    std::vector<uint8_t> actual;
    manager.read_slot(0, actual);
    EXPECT_EQ(actual, first);
    manager.read_slot(2, actual);
    EXPECT_EQ(actual, second);
}

TEST(TestKVCacheOffloadManager, AcquireReleaseAndExhaustion) {
    KVCacheOffloadManager manager(make_layout(), make_config(2), "CPU");

    const auto first = manager.acquire_slot();
    const auto second = manager.acquire_slot();
    ASSERT_TRUE(first.has_value());
    ASSERT_TRUE(second.has_value());
    EXPECT_NE(*first, *second);
    EXPECT_EQ(manager.get_num_free_slots(), 0);

    EXPECT_FALSE(manager.acquire_slot().has_value());

    manager.release_slot(*first);
    EXPECT_EQ(manager.get_num_free_slots(), 1);
    EXPECT_EQ(manager.acquire_slot(), first);
}

TEST(TestKVCacheOffloadManager, RejectsDoubleRelease) {
    KVCacheOffloadManager manager(make_layout(), make_config(2), "CPU");
    const auto slot = manager.acquire_slot();
    ASSERT_TRUE(slot.has_value());

    manager.release_slot(*slot);
    EXPECT_THROW(manager.release_slot(*slot), ov::Exception);
}

TEST(TestKVCacheOffloadManager, RejectsInvalidSlotAndBlockSize) {
    KVCacheOffloadManager manager(make_layout(), make_config(2), "CPU");
    std::vector<uint8_t> block_data;

    EXPECT_THROW(manager.write_slot(2, make_pattern(0)), ov::Exception);
    EXPECT_THROW(manager.read_slot(2, block_data), ov::Exception);
    EXPECT_THROW(manager.release_slot(2), ov::Exception);

    std::vector<uint8_t> too_short(SLOT_BYTES - 1, 0);
    EXPECT_THROW(manager.write_slot(0, too_short), ov::Exception);
}

TEST(TestKVCacheOffloadManager, RejectsCapacitySmallerThanOneBlock) {
    CacheOffloadConfig config;
    config.capacity_bytes = SLOT_BYTES - 1;

    EXPECT_THROW(KVCacheOffloadManager(make_layout(), config, "CPU"), ov::Exception);
}

TEST(TestKVCacheOffloadManager, RejectsUnsupportedDeviceAndDirectIO) {
    EXPECT_TRUE(KVCacheOffloadManager::is_supported_device("CPU"));
    EXPECT_TRUE(KVCacheOffloadManager::is_supported_device("GPU"));
    EXPECT_TRUE(KVCacheOffloadManager::is_supported_device("GPU.0"));
    EXPECT_FALSE(KVCacheOffloadManager::is_supported_device("NPU"));

    EXPECT_THROW(KVCacheOffloadManager(make_layout(), make_config(1), "NPU"), ov::Exception);

    CacheOffloadConfig direct_io = make_config(1);
    direct_io.use_page_cache = false;
    EXPECT_THROW(KVCacheOffloadManager(make_layout(), direct_io, "CPU"), ov::Exception);
}

TEST(TestKVCacheOffloadManager, RejectsMissingDirectory) {
    CacheOffloadConfig config = make_config(1);
    config.path = (std::filesystem::temp_directory_path() / "ov_genai_offload_missing_dir").string();
    ASSERT_FALSE(std::filesystem::exists(config.path));

    EXPECT_THROW(KVCacheOffloadManager(make_layout(), config, "CPU"), ov::Exception);
}

TEST(TestKVCacheOffloadManager, UsesConfiguredDirectory) {
    const auto directory = std::filesystem::temp_directory_path() / "ov_genai_offload_test_dir";
    std::filesystem::create_directories(directory);

    CacheOffloadConfig config = make_config(1);
    config.path = directory.string();

    {
        KVCacheOffloadManager manager(make_layout(), config, "CPU");
        EXPECT_EQ(manager.get_file_path().parent_path(), directory);
    }

    std::filesystem::remove_all(directory);
}
