// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"

using namespace ov::genai;

namespace {

// Points at an OpenVINO IR text-generation model; the suite is skipped when it is not provided.
constexpr const char* MODEL_PATH_ENV = "OFFLOAD_E2E_MODEL";

// Points at a visual-language (hybrid KV + linear-attention) model directory. ContinuousBatchingPipeline can load such
// a directory directly for plain text generation (it falls back to openvino_language_model.xml).
constexpr const char* VLM_MODEL_PATH_ENV = "OFFLOAD_E2E_VLM_MODEL";

// Inference device for the comparison, defaults to CPU.
constexpr const char* DEVICE_ENV = "OFFLOAD_E2E_DEVICE";

std::string e2e_device() {
    const char* device = std::getenv(DEVICE_ENV);
    return (device == nullptr || *device == '\0') ? std::string{"CPU"} : std::string{device};
}

SchedulerConfig make_scheduler_config() {
    SchedulerConfig config;
    config.max_num_batched_tokens = 512;
    config.max_num_seqs = 4;
    config.dynamic_split_fuse = false;
    config.enable_prefix_caching = true;
    return config;
}

GenerationConfig make_greedy_config(size_t max_new_tokens) {
    GenerationConfig config;
    config.max_new_tokens = max_new_tokens;
    config.do_sample = false;
    config.num_beams = 1;
    config.ignore_eos = true;
    return config;
}

std::string repeat(const std::string& phrase, size_t times) {
    std::string result;
    for (size_t i = 0; i < times; ++i) {
        result += phrase;
    }
    return result;
}

/// Candidates for samples/c/kv_cache_store_plugin's build output (same search pattern as
/// tests/cpp/store_registry.cpp). Only built when the superbuild has ENABLE_SAMPLES on.
std::optional<std::string> find_sample_plugin_library() {
#ifdef _WIN32
    const char* candidates[] = {
        "../../samples/c/kv_cache_store_plugin/RelWithDebInfo/sample_kv_cache_store_plugin.dll",
        "../../samples/c/kv_cache_store_plugin/sample_kv_cache_store_plugin.dll",
    };
#else
    const char* candidates[] = {
        "../../samples/c/kv_cache_store_plugin/libsample_kv_cache_store_plugin.so",
        "../../samples/c/kv_cache_store_plugin/libsample_kv_cache_store_plugin.dylib",
    };
#endif
    for (const char* candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return std::filesystem::absolute(candidate).string();
        }
    }
    return std::nullopt;
}

std::vector<int64_t> generate_ids(ContinuousBatchingPipeline& pipeline,
                                  uint64_t request_id,
                                  const std::string& prompt,
                                  size_t max_new_tokens,
                                  size_t* prefix_cache_hit_tokens = nullptr) {
    auto handle = pipeline.add_request(request_id, prompt, make_greedy_config(max_new_tokens));
    while (pipeline.has_non_finished_requests()) {
        pipeline.step();
    }
    auto outputs = handle->read_all();
    EXPECT_FALSE(outputs.empty());
    if (prefix_cache_hit_tokens != nullptr) {
        *prefix_cache_hit_tokens = handle->get_perf_metrics().get_num_prefix_cache_hit_tokens();
    }
    return outputs.empty() ? std::vector<int64_t>{} : outputs.front().generated_ids;
}

// DIAGNOSTIC: runs the same prompt twice, sequentially, in the same pipeline instance with plain opportunistic
// prefix caching (no offload). The second run should hit the in-memory prefix cache for (almost) the whole prompt and
// produce the exact same first generated token as the first (cold) run. Used to isolate whether a cache-hit can
// silently diverge from a cold run for hybrid (KV + linear-attention) models.
// Takes an explicit cache_interval_multiplier to test whether the LA checkpoint granularity
// (as opposed to the cache-hit/restore mechanism itself) is implicated in any divergence.
void run_repeated_prompt_prefix_cache_hit_diagnostic(const std::string& model_path,
                                                      std::optional<size_t> cache_interval_multiplier = std::nullopt) {
    const std::string prompt = repeat("The quick brown fox jumps over the lazy dog. ", 40);

    SchedulerConfig config = make_scheduler_config();
    if (cache_interval_multiplier.has_value()) {
        config.cache_interval_multiplier = *cache_interval_multiplier;
    }
    ContinuousBatchingPipeline pipeline(model_path, config, e2e_device());

    const auto first_ids = generate_ids(pipeline, /*request_id=*/51, prompt, /*max_new_tokens=*/1);
    ASSERT_EQ(first_ids.size(), 1u);
    const auto second_ids = generate_ids(pipeline, /*request_id=*/52, prompt, /*max_new_tokens=*/1);
    ASSERT_EQ(second_ids.size(), 1u);

    EXPECT_EQ(first_ids, second_ids)
        << "repeating the identical prompt (opportunistic prefix-cache hit) produced a different "
        << "first generated token than the cold run";
}

/// Prefill is split into chunks, which is what makes a hybrid model leave a linear-attention checkpoint at every interval.
SchedulerConfig make_restart_scheduler_config() {
    auto config = make_scheduler_config();
    config.max_num_batched_tokens = 4096;
    config.max_num_seqs = 1;
    config.dynamic_split_fuse = true;
    return config;
}

/// Offload through the sample plugin's persistent provider, which keeps one file per block in `store_dir`, with every
/// completed block stored right after the forward pass that completed it.
SchedulerConfig make_persistent_store_config(const std::string& plugin_path, const std::string& store_dir, bool hybrid) {
    auto config = make_restart_scheduler_config();
    config.use_cache_offload = true;
    config.cache_offload_config.path = store_dir;
    config.cache_offload_config.capacity_bytes = 2048ull * 1024 * 1024;
    if (hybrid) {
        // A linear-attention checkpoint is large, so the store holds only a few of them.
        config.cache_offload_config.linear_attention_capacity_bytes = 4096ull * 1024 * 1024;
    }
    config.cache_offload_config.buffer_slots = 256;
    config.cache_offload_config.store_library_path = plugin_path;
    config.cache_offload_config.store_provider_name = "sample-persistent-file";
    config.cache_offload_config.allow_untrusted_store_plugin = true;
    config.cache_offload_config.store_completed_blocks = true;
    return config;
}

/// A pipeline that is gone leaves its completed blocks in the persistent store, and a new pipeline that gets the same
/// prompt reads them back instead of recomputing them, with the same output as without any store.
void run_restart_reuses_stored_blocks(const std::string& model_path, const std::string& plugin_path, bool hybrid) {
    // A hybrid model only reuses a prompt that crosses at least one linear-attention checkpoint interval.
    const std::string prompt = repeat("The quick brown fox jumps over the lazy dog. ", hybrid ? 310 : 100);
    constexpr size_t NEW_TOKENS = 8;
    const std::string device = e2e_device();

    auto baseline_config = make_restart_scheduler_config();
    std::vector<int64_t> baseline_ids;
    {
        ContinuousBatchingPipeline pipeline(model_path, baseline_config, device);
        baseline_ids = generate_ids(pipeline, /*request_id=*/60, prompt, NEW_TOKENS);
    }
    ASSERT_EQ(baseline_ids.size(), NEW_TOKENS);

    const auto store_dir = std::filesystem::temp_directory_path() / "openvino_genai_persistent_store_e2e";
    std::filesystem::remove_all(store_dir);
    std::filesystem::create_directories(store_dir);
    const auto config = make_persistent_store_config(plugin_path, store_dir.string(), hybrid);

    {
        ContinuousBatchingPipeline pipeline(model_path, config, device);
        size_t first_run_hit_tokens = 0;
        EXPECT_EQ(generate_ids(pipeline, /*request_id=*/61, prompt, NEW_TOKENS, &first_run_hit_tokens), baseline_ids);
        EXPECT_EQ(first_run_hit_tokens, 0u) << "nothing is stored yet, so the first run has nothing to reuse";
    }  // the writing pipeline, and with it every in-memory cache, is gone

    {
        ContinuousBatchingPipeline pipeline(model_path, config, device);
        size_t restart_hit_tokens = 0;
        EXPECT_EQ(generate_ids(pipeline, /*request_id=*/62, prompt, NEW_TOKENS, &restart_hit_tokens), baseline_ids)
            << "reusing stored blocks in a new pipeline diverged from a run without a store";
        std::cout << "[ INFO     ] prefix cache hit tokens after a restart: " << restart_hit_tokens << std::endl;
        EXPECT_GT(restart_hit_tokens, 0u) << "the new pipeline must read the stored blocks instead of recomputing them";
    }
    std::filesystem::remove_all(store_dir);
}

}  // namespace

TEST(TestPersistentStoreEndToEnd, DiagnosticRepeatedPromptPrefixCacheHitMatchesColdRun) {
    const char* model_path = std::getenv(MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << MODEL_PATH_ENV << " is not set, skipping prefix-cache diagnostic test.";
    }
    run_repeated_prompt_prefix_cache_hit_diagnostic(model_path);
}

// Forces the smallest possible LA checkpoint interval (multiplier 1, same granularity as the KV
// block size) so the 409-token prompt crosses MANY checkpoint boundaries instead of staying inside
// a single "incomplete" one. If this still diverges, the checkpoint interval SIZE is not the cause.
TEST(TestPersistentStoreEndToEnd, DiagnosticHybridRepeatedPromptPrefixCacheHitSmallIntervalMatchesColdRun) {
    const char* model_path = std::getenv(VLM_MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << VLM_MODEL_PATH_ENV << " is not set, skipping hybrid prefix-cache diagnostic test.";
    }
    run_repeated_prompt_prefix_cache_hit_diagnostic(model_path, /*cache_interval_multiplier=*/1);
}

TEST(TestPersistentStoreEndToEnd, DiagnosticHybridRepeatedPromptPrefixCacheHitMatchesColdRun) {
    const char* model_path = std::getenv(VLM_MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << VLM_MODEL_PATH_ENV << " is not set, skipping hybrid prefix-cache diagnostic test.";
    }
    run_repeated_prompt_prefix_cache_hit_diagnostic(model_path);
}

TEST(TestPersistentStoreEndToEnd, NewPipelineReusesBlocksStoredByAnEarlierOne) {
    const char* model_path = std::getenv(MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << MODEL_PATH_ENV << " is not set, skipping persistent store end-to-end test.";
    }
    const auto plugin_path = find_sample_plugin_library();
    if (!plugin_path.has_value()) {
        GTEST_SKIP() << "samples/c/kv_cache_store_plugin was not built (requires -DENABLE_SAMPLES=ON)";
    }
    run_restart_reuses_stored_blocks(model_path, *plugin_path, /*hybrid=*/false);
}

TEST(TestPersistentStoreEndToEnd, HybridModelNewPipelineReusesBlocksStoredByAnEarlierOne) {
    const char* model_path = std::getenv(VLM_MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << VLM_MODEL_PATH_ENV << " is not set, skipping hybrid persistent store end-to-end test.";
    }
    const auto plugin_path = find_sample_plugin_library();
    if (!plugin_path.has_value()) {
        GTEST_SKIP() << "samples/c/kv_cache_store_plugin was not built (requires -DENABLE_SAMPLES=ON)";
    }
    run_restart_reuses_stored_blocks(model_path, *plugin_path, /*hybrid=*/true);
}
