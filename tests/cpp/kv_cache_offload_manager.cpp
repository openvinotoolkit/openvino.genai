// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <filesystem>
#include <map>
#include <numeric>

#include "continuous_batching/cache/offload_plan.hpp"
#include "openvino/core/except.hpp"

using namespace ov::genai;

TEST(TestOffloadPlan, MaxTokensWithinBudgetFillsTheBudgetWithWholeBlocksOfEveryGroup) {
    // A KV block of 100 bytes per 16 tokens and a checkpoint of 1000 bytes per 64 tokens.
    const std::vector<GroupCost> groups{{100, 16}, {1000, 64}};
    // 64 tokens cost 4 * 100 + 1 * 1000 = 1400 bytes; 65 tokens need a second checkpoint.
    EXPECT_EQ(max_tokens_within_budget(groups, 1400), 64u);
    EXPECT_EQ(max_tokens_within_budget(groups, 1399), 48u);
    EXPECT_EQ(max_tokens_within_budget(groups, 0), 0u);
    // The result never costs more than the budget.
    for (const std::uint64_t budget : {500ull, 2500ull, 99999ull}) {
        const size_t tokens = max_tokens_within_budget(groups, budget);
        const std::uint64_t cost = blocks_for_tokens(groups[0], tokens) * 100ull + blocks_for_tokens(groups[1], tokens) * 1000ull;
        EXPECT_LE(cost, budget);
        const size_t more = tokens + 1;
        EXPECT_GT(blocks_for_tokens(groups[0], more) * 100ull + blocks_for_tokens(groups[1], more) * 1000ull, budget);
    }
}

TEST(TestOffloadPlan, ResolveSplitsTheBudgetOverTheGroupsAndLeavesTheConfigAlone) {
    CacheOffloadConfig config;
    config.ssd_cache_capacity_gb = 1.0 / 1024;  // 1 MiB
    config.kv_cache_dir = std::filesystem::temp_directory_path().string();
    config.write_policy = CacheOffloadWritePolicy::WRITE_THROUGH;
    config.plugin = CacheStorePluginConfig{};
    config.plugin->library_path = "plugin.dll";
    config.plugin->properties = {{"key", "value"}};
    config.plugin->timeout_ms = 1234;
    const CacheOffloadConfig before = config;

    const OffloadSettings kv_only = resolve_offload_settings(config, {{1024, 16}});
    EXPECT_EQ(kv_only.capacity_bytes, 1024u * 1024u);
    EXPECT_EQ(kv_only.linear_attention_capacity_bytes, 0u);
    EXPECT_TRUE(kv_only.store_completed_blocks);
    EXPECT_EQ(kv_only.store_library_path, "plugin.dll");
    EXPECT_EQ(kv_only.store_properties, (std::map<std::string, std::string>{{"key", "value"}}));
    EXPECT_EQ(kv_only.store_timeout_ms, 1234u);

    const OffloadSettings hybrid = resolve_offload_settings(config, {{1024, 16}, {64 * 1024, 128}});
    EXPECT_GT(hybrid.capacity_bytes, 0u);
    EXPECT_GT(hybrid.linear_attention_capacity_bytes, 0u);
    EXPECT_LE(hybrid.capacity_bytes + hybrid.linear_attention_capacity_bytes, 1024u * 1024u);
    // Both groups cover the same prefix: the KV blocks reach at least as far as the checkpoints start.
    const size_t kv_tokens = hybrid.capacity_bytes / 1024 * 16;
    const size_t checkpoints = hybrid.linear_attention_capacity_bytes / (64 * 1024);
    EXPECT_GE(kv_tokens, (checkpoints - 1) * 128 + 1);
    EXPECT_LE(kv_tokens, checkpoints * 128);

    EXPECT_TRUE(config == before);
}

TEST(TestOffloadPlan, ResolveRejectsABudgetThatCannotHoldOneBlockOfEveryGroup) {
    CacheOffloadConfig config;
    config.ssd_cache_capacity_gb = 1.0 / 1024;
    config.kv_cache_dir = std::filesystem::temp_directory_path().string();
    EXPECT_THROW(resolve_offload_settings(config, {{1024, 16}, {2 * 1024 * 1024, 128}}), ov::Exception);
}

TEST(TestOffloadPlan, DramAloneIsNotSupportedYetButIsIgnoredNextToAnSsdTier) {
    CacheOffloadConfig dram_only;
    dram_only.dram_cache_capacity_gb = 1;
    EXPECT_THROW(resolve_offload_settings(dram_only, {{1024, 16}}), ov::Exception);

    CacheOffloadConfig both;
    both.dram_cache_capacity_gb = 1;
    both.ssd_cache_capacity_gb = 1.0 / 1024;
    both.kv_cache_dir = std::filesystem::temp_directory_path().string();
    EXPECT_NO_THROW(resolve_offload_settings(both, {{1024, 16}}));
}