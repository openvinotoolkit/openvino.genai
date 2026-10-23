// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "continuous_batching/cache/kv_cache_disk_layout.hpp"
#include "continuous_batching/cache/store_registry.hpp"

using namespace ov::genai;

namespace {

constexpr size_t BLOCK_SIZE = 32;
constexpr size_t NUM_BLOCKS = 4;

CacheOffloadConfig make_config() {
    CacheOffloadConfig config;
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

void expect_store_is_usable(IKVCacheStore& store) {
    EXPECT_EQ(store.get_block_size(), BLOCK_SIZE);
    EXPECT_EQ(store.get_capacity_blocks(), NUM_BLOCKS);
    EXPECT_FALSE(store.has_block(1));

    const auto expected = make_block(/*seed=*/5);
    ASSERT_TRUE(store.write_block(1, expected));
    ASSERT_TRUE(store.has_block(1));

    std::vector<uint8_t> actual;
    ASSERT_TRUE(store.read_block(1, actual));
    EXPECT_EQ(actual, expected);
}

/// A handful of relative candidates for openvino_genai's own shared library, which also exports
/// the store ABI entry symbol (see abi_file_store_provider.cpp). Used to exercise a genuinely
/// separate, dynamically loaded module without needing a dedicated mock-plugin build target.
std::optional<std::string> find_self_as_plugin_library() {
#ifdef _WIN32
    const char* candidates[] = {
        "../../openvino_genai/openvino_genai.dll",
        "../openvino_genai/openvino_genai.dll",
        "openvino_genai/openvino_genai.dll",
    };
#else
    const char* candidates[] = {
        "../../openvino_genai/libopenvino_genai.so",
        "../openvino_genai/libopenvino_genai.so",
        "openvino_genai/libopenvino_genai.so",
    };
#endif
    for (const char* candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return std::filesystem::absolute(candidate).string();
        }
    }
    return std::nullopt;
}

/// Candidates for samples/c/kv_cache_store_plugin's build output: a genuinely standalone,
/// independently built reference plugin (see that directory), not merely our own library acting
/// as one. Only built when the superbuild has ENABLE_SAMPLES on.
std::optional<std::string> find_sample_plugin_library() {
#ifdef _WIN32
    const char* candidates[] = {
        "../../samples/c/kv_cache_store_plugin/RelWithDebInfo/sample_kv_cache_store_plugin.dll",
        "../../samples/c/kv_cache_store_plugin/sample_kv_cache_store_plugin.dll",
    };
#else
    const char* candidates[] = {
        "../../samples/c/kv_cache_store_plugin/libsample_kv_cache_store_plugin.so",
    };
#endif
    for (const char* candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return std::filesystem::absolute(candidate).string();
        }
    }
    return std::nullopt;
}

}  // namespace

TEST(TestStoreRegistry, DefaultConfigUsesTheInTreeFileStoreDirectly) {
    auto store = open_kv_cache_store(make_layout(), make_config(), "CPU", "kv");
    ASSERT_NE(store, nullptr);
    expect_store_is_usable(*store);
}

TEST(TestStoreRegistry, DisallowedPluginIsIgnoredEvenWhenAPathIsSet) {
    CacheOffloadConfig config = make_config();
    config.store_library_path = "does_not_matter_because_disallowed.dll";
    config.allow_untrusted_store_plugin = false;

    auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
    ASSERT_NE(store, nullptr);
    expect_store_is_usable(*store);
}

TEST(TestStoreRegistry, NonexistentPluginLibraryFallsBackGracefully) {
    CacheOffloadConfig config = make_config();
    config.store_library_path = "this_library_does_not_exist_anywhere.dll";
    config.allow_untrusted_store_plugin = true;

    auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
    ASSERT_NE(store, nullptr);
    expect_store_is_usable(*store);
}

#ifdef _WIN32
TEST(TestStoreRegistry, LibraryMissingTheEntrySymbolFallsBackGracefully) {
    CacheOffloadConfig config = make_config();
    // A real, always-present system DLL that definitely does not export our ABI entry symbol.
    config.store_library_path = "kernel32.dll";
    config.allow_untrusted_store_plugin = true;

    auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
    ASSERT_NE(store, nullptr);
    expect_store_is_usable(*store);
}
#endif

TEST(TestStoreRegistry, LoadsAndUsesAGenuinelySeparateDynamicallyLoadedPlugin) {
    const std::optional<std::string> library_path = find_self_as_plugin_library();
    if (!library_path.has_value()) {
        GTEST_SKIP() << "Could not locate openvino_genai's own shared library next to the test binary";
    }

    CacheOffloadConfig config = make_config();
    config.store_library_path = *library_path;
    config.store_provider_name = "file";
    config.allow_untrusted_store_plugin = true;

    auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
    ASSERT_NE(store, nullptr);
    expect_store_is_usable(*store);
}

TEST(TestStoreRegistry, LoadsAndUsesTheStandaloneSampleThirdPartyPlugin) {
    const std::optional<std::string> library_path = find_sample_plugin_library();
    if (!library_path.has_value()) {
        GTEST_SKIP() << "samples/c/kv_cache_store_plugin was not built (requires -DENABLE_SAMPLES=ON)";
    }

    CacheOffloadConfig config = make_config();
    config.store_library_path = *library_path;
    config.store_provider_name = "sample-in-memory";
    config.allow_untrusted_store_plugin = true;

    auto store = open_kv_cache_store(make_layout(), config, "CPU", "kv");
    ASSERT_NE(store, nullptr);
    expect_store_is_usable(*store);
}

// The ABI opens one store per namespace, so blocks of different groups must not be counted together.
TEST(TestStoreRegistry, SampleThirdPartyPluginKeepsGroupsInSeparateStores) {
    const std::optional<std::string> library_path = find_sample_plugin_library();
    if (!library_path.has_value()) {
        GTEST_SKIP() << "samples/c/kv_cache_store_plugin was not built (requires -DENABLE_SAMPLES=ON)";
    }

    CacheOffloadConfig config = make_config();
    config.store_library_path = *library_path;
    config.store_provider_name = "sample-in-memory";
    config.allow_untrusted_store_plugin = true;

    auto first = open_kv_cache_store(make_layout(), config, "CPU", "namespace_test_first");
    auto second = open_kv_cache_store(make_layout(), config, "CPU", "namespace_test_second");
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);

    ASSERT_TRUE(first->write_block(7, make_block(/*seed=*/1)));
    EXPECT_EQ(first->get_num_free_blocks(), NUM_BLOCKS - 1);
    EXPECT_EQ(second->get_num_free_blocks(), NUM_BLOCKS);
}
