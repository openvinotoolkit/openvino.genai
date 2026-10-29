// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "openvino/genai/cache_eviction.hpp"
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"

using namespace ov::genai;

namespace {

// Points at an OpenVINO IR text-generation model; the suite is skipped when it is not provided.
constexpr const char* MODEL_PATH_ENV = "OFFLOAD_E2E_MODEL";

// Points at a visual-language (hybrid KV + linear-attention) model directory; used to additionally
// cover the linear-attention cache payload of the snapshot. ContinuousBatchingPipeline can load such
// a directory directly for plain text generation (it falls back to openvino_language_model.xml).
constexpr const char* VLM_MODEL_PATH_ENV = "OFFLOAD_E2E_VLM_MODEL";

// Inference device for the comparison, defaults to CPU.
constexpr const char* DEVICE_ENV = "OFFLOAD_E2E_DEVICE";

std::string session_snapshot_device() {
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

// Save mid-generation (while the request is still RUNNING - a finished request's cache is freed
// immediately, so there is no window to save it afterwards), destroy the pipeline (simulating a
// process restart), restore into a fresh pipeline instance, and finish generation. The stitched
// (already-generated + resumed) token stream must match an uninterrupted baseline run, proving the
// restored request resumes decoding instead of silently re-running prefill.
void run_round_trip_test(const std::string& model_path, const CacheOffloadConfig& store_config = {}) {
    const std::string prompt = repeat("The quick brown fox jumps over the lazy dog. ", 10) + " Continue the story:";
    constexpr size_t TOTAL_NEW_TOKENS = 20;
    // Large enough that manually driving a handful of steps below cannot reach it naturally.
    constexpr size_t SPLIT_RUN_BUDGET = 1000;
    constexpr size_t NUM_MANUAL_STEPS = 15;

    std::vector<int64_t> baseline_ids;
    {
        ContinuousBatchingPipeline pipeline(model_path, make_scheduler_config(), session_snapshot_device());
        constexpr uint64_t REQUEST_ID = 41;
        auto handle = pipeline.add_request(REQUEST_ID, prompt, make_greedy_config(TOTAL_NEW_TOKENS));
        while (pipeline.has_non_finished_requests()) {
            pipeline.step();
        }
        auto outputs = handle->read_all();
        ASSERT_FALSE(outputs.empty());
        baseline_ids = outputs.front().generated_ids;
    }
    ASSERT_EQ(baseline_ids.size(), TOTAL_NEW_TOKENS);

    const auto snapshot_path = std::filesystem::temp_directory_path() / "openvino_genai_session_snapshot_test.bin";
    std::vector<int64_t> already_generated_ids;
    {
        ContinuousBatchingPipeline pipeline(model_path, make_scheduler_config(), session_snapshot_device());
        constexpr uint64_t REQUEST_ID = 42;
        auto handle = pipeline.add_request(REQUEST_ID, prompt, make_greedy_config(SPLIT_RUN_BUDGET));
        for (size_t i = 0; i < NUM_MANUAL_STEPS && handle->get_status() == GenerationStatus::RUNNING; ++i) {
            pipeline.step();
            while (handle->can_read()) {
                for (auto& [seq_id, output] : handle->read()) {
                    already_generated_ids.insert(already_generated_ids.end(),
                                                 output.generated_ids.begin(),
                                                 output.generated_ids.end());
                }
            }
        }
        ASSERT_EQ(handle->get_status(), GenerationStatus::RUNNING)
            << "the request finished naturally before it could be interrupted, test setup is invalid";
        ASSERT_LT(already_generated_ids.size(), TOTAL_NEW_TOKENS)
            << "manual stepping already produced as many tokens as the baseline, nothing left to resume";

        pipeline.save_state(REQUEST_ID, snapshot_path.string(), store_config);
    }

    std::vector<int64_t> resumed_ids;
    {
        ContinuousBatchingPipeline pipeline(model_path, make_scheduler_config(), session_snapshot_device());
        constexpr uint64_t RESTORED_REQUEST_ID = 43;
        const size_t remaining_tokens = TOTAL_NEW_TOKENS - already_generated_ids.size();
        auto handle =
            pipeline.restore_state(RESTORED_REQUEST_ID, snapshot_path.string(), make_greedy_config(remaining_tokens), store_config);
        while (pipeline.has_non_finished_requests()) {
            pipeline.step();
        }
        auto outputs = handle->read_all();
        ASSERT_FALSE(outputs.empty());
        resumed_ids = outputs.front().generated_ids;
    }
    std::filesystem::remove(snapshot_path);

    std::vector<int64_t> stitched_ids = already_generated_ids;
    stitched_ids.insert(stitched_ids.end(), resumed_ids.begin(), resumed_ids.end());

    EXPECT_EQ(stitched_ids, baseline_ids)
        << "resuming from a session snapshot diverged from the uninterrupted baseline";
}

// DIAGNOSTIC (not a session-snapshot test): runs the same prompt twice, sequentially, in the same
// pipeline instance with plain opportunistic prefix caching (no snapshot, no offload). The second
// run should hit the in-memory prefix cache for (almost) the whole prompt and produce the exact
// same first generated token as the first (cold) run. Used to isolate whether a cache-hit can
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
    ContinuousBatchingPipeline pipeline(model_path, config, session_snapshot_device());

    constexpr uint64_t FIRST_REQUEST_ID = 51;
    auto first_handle = pipeline.add_request(FIRST_REQUEST_ID, prompt, make_greedy_config(1));
    while (pipeline.has_non_finished_requests()) {
        pipeline.step();
    }
    auto first_outputs = first_handle->read_all();
    ASSERT_FALSE(first_outputs.empty());
    const auto first_ids = first_outputs.front().generated_ids;
    ASSERT_EQ(first_ids.size(), 1u);

    constexpr uint64_t SECOND_REQUEST_ID = 52;
    auto second_handle = pipeline.add_request(SECOND_REQUEST_ID, prompt, make_greedy_config(1));
    while (pipeline.has_non_finished_requests()) {
        pipeline.step();
    }
    auto second_outputs = second_handle->read_all();
    ASSERT_FALSE(second_outputs.empty());
    const auto second_ids = second_outputs.front().generated_ids;
    ASSERT_EQ(second_ids.size(), 1u);

    EXPECT_EQ(first_ids, second_ids)
        << "repeating the identical prompt (opportunistic prefix-cache hit) produced a different "
        << "first generated token than the cold run";
}

TEST(TestSessionSnapshotEndToEnd, DiagnosticRepeatedPromptPrefixCacheHitMatchesColdRun) {
    const char* model_path = std::getenv(MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << MODEL_PATH_ENV << " is not set, skipping prefix-cache diagnostic test.";
    }
    run_repeated_prompt_prefix_cache_hit_diagnostic(model_path);
}

// Forces the smallest possible LA checkpoint interval (multiplier 1, same granularity as the KV
// block size) so the 409-token prompt crosses MANY checkpoint boundaries instead of staying inside
// a single "incomplete" one. If this still diverges, the checkpoint interval SIZE is not the cause.
TEST(TestSessionSnapshotEndToEnd, DiagnosticHybridRepeatedPromptPrefixCacheHitSmallIntervalMatchesColdRun) {
    const char* model_path = std::getenv(VLM_MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << VLM_MODEL_PATH_ENV << " is not set, skipping hybrid prefix-cache diagnostic test.";
    }
    run_repeated_prompt_prefix_cache_hit_diagnostic(model_path, /*cache_interval_multiplier=*/1);
}


TEST(TestSessionSnapshotEndToEnd, DiagnosticHybridRepeatedPromptPrefixCacheHitMatchesColdRun) {
    const char* model_path = std::getenv(VLM_MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << VLM_MODEL_PATH_ENV << " is not set, skipping hybrid prefix-cache diagnostic test.";
    }
    run_repeated_prompt_prefix_cache_hit_diagnostic(model_path);
}

}  // namespace

TEST(TestSessionSnapshotEndToEnd, RestoredRequestContinuesIdenticallyToUninterruptedRun) {
    const char* model_path = std::getenv(MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << MODEL_PATH_ENV << " is not set, skipping session snapshot end-to-end test.";
    }
    run_round_trip_test(model_path);
}

// Qwen3.5 is natively multimodal: even plain text-only prompts are routed through an input
// embeddings model (ModelInputType::EMBEDDINGS). This also exercises the linear-attention cache
// payload for real, since Qwen3.5 is a hybrid (KV + linear-attention) model.
TEST(TestSessionSnapshotEndToEnd, HybridModelRestoredRequestContinuesIdenticallyToUninterruptedRun) {
    const char* model_path = std::getenv(VLM_MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << VLM_MODEL_PATH_ENV << " is not set, skipping hybrid session snapshot end-to-end test.";
    }
    run_round_trip_test(model_path);
}

// Routes the snapshot's KV (and, for hybrid models, linear-attention) block payload through the
// standalone sample third-party plugin (see samples/c/kv_cache_store_plugin) instead of embedding
// it inline in the snapshot file, proving save_state()/restore_state() can delegate to the same
// kind of vendor plugin used by the opportunistic offload cache.
TEST(TestSessionSnapshotEndToEnd, RestoredRequestViaStorePluginContinuesIdenticallyToUninterruptedRun) {
    const char* model_path = std::getenv(MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << MODEL_PATH_ENV << " is not set, skipping session snapshot end-to-end test.";
    }
    const std::optional<std::string> library_path = find_sample_plugin_library();
    if (!library_path.has_value()) {
        GTEST_SKIP() << "samples/c/kv_cache_store_plugin was not built (requires -DENABLE_SAMPLES=ON)";
    }

    CacheOffloadConfig store_config;
    store_config.store_library_path = *library_path;
    store_config.store_provider_name = "sample-in-memory";
    store_config.allow_untrusted_store_plugin = true;
    run_round_trip_test(model_path, store_config);
}

TEST(TestSessionSnapshotEndToEnd, RestoreRejectsAMissingFile) {
    const char* model_path = std::getenv(MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << MODEL_PATH_ENV << " is not set, skipping session snapshot end-to-end test.";
    }

    ContinuousBatchingPipeline pipeline(model_path, make_scheduler_config(), session_snapshot_device());
    const auto missing_path =
        std::filesystem::temp_directory_path() / "openvino_genai_session_snapshot_does_not_exist.bin";
    EXPECT_THROW(pipeline.restore_state(1, missing_path.string(), make_greedy_config(5)), ov::Exception);
}

TEST(TestSessionSnapshotEndToEnd, RestoreRejectsACorruptedFile) {
    const char* model_path = std::getenv(MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << MODEL_PATH_ENV << " is not set, skipping session snapshot end-to-end test.";
    }

    const std::string prompt = repeat("The quick brown fox jumps over the lazy dog. ", 10) + " Continue the story:";
    const auto snapshot_path =
        std::filesystem::temp_directory_path() / "openvino_genai_session_snapshot_corrupt_test.bin";
    {
        ContinuousBatchingPipeline pipeline(model_path, make_scheduler_config(), session_snapshot_device());
        constexpr uint64_t REQUEST_ID = 44;
        auto handle = pipeline.add_request(REQUEST_ID, prompt, make_greedy_config(1000));
        for (size_t i = 0; i < 15 && handle->get_status() == GenerationStatus::RUNNING; ++i) {
            pipeline.step();
        }
        ASSERT_EQ(handle->get_status(), GenerationStatus::RUNNING)
            << "the request finished naturally before it could be interrupted, test setup is invalid";
        pipeline.save_state(REQUEST_ID, snapshot_path.string());
    }

    // Flip a byte in the middle of the payload so the stored checksum no longer matches.
    {
        std::fstream file(snapshot_path, std::ios::in | std::ios::out | std::ios::binary);
        ASSERT_TRUE(file.is_open());
        file.seekg(0, std::ios::end);
        const auto size = file.tellg();
        ASSERT_GT(size, 0);
        file.seekp(size / 2);
        char byte = 0;
        file.seekg(size / 2);
        file.read(&byte, 1);
        byte = static_cast<char>(~byte);
        file.seekp(size / 2);
        file.write(&byte, 1);
    }

    ContinuousBatchingPipeline pipeline(model_path, make_scheduler_config(), session_snapshot_device());
    EXPECT_THROW(pipeline.restore_state(45, snapshot_path.string(), make_greedy_config(5)), ov::Exception);

    std::filesystem::remove(snapshot_path);
}

TEST(TestSessionSnapshotEndToEnd, RestoreRespectsAFixedCacheBudget) {
    const char* model_path = std::getenv(MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << MODEL_PATH_ENV << " is not set, skipping session snapshot end-to-end test.";
    }

    const std::string prompt = repeat("The quick brown fox jumps over the lazy dog. ", 40);
    const auto snapshot_path =
        std::filesystem::temp_directory_path() / "openvino_genai_session_snapshot_budget_test.bin";
    {
        // Dynamic cache sizing (num_kv_blocks == 0) so the snapshot can be larger than the budgets below.
        ContinuousBatchingPipeline pipeline(model_path, make_scheduler_config(), session_snapshot_device());
        constexpr uint64_t REQUEST_ID = 47;
        auto handle = pipeline.add_request(REQUEST_ID, prompt, make_greedy_config(1000));
        for (size_t i = 0; i < 8 && handle->get_status() == GenerationStatus::RUNNING; ++i) {
            pipeline.step();
        }
        ASSERT_EQ(handle->get_status(), GenerationStatus::RUNNING)
            << "the request finished naturally before it could be interrupted, test setup is invalid";
        pipeline.save_state(REQUEST_ID, snapshot_path.string());
    }

    {
        auto fixed_config = make_scheduler_config();
        fixed_config.num_kv_blocks = 4;
        ContinuousBatchingPipeline pipeline(model_path, fixed_config, session_snapshot_device());
        EXPECT_THROW(pipeline.restore_state(48, snapshot_path.string(), make_greedy_config(4)), ov::Exception)
            << "a snapshot larger than a fixed num_kv_blocks budget must be rejected, not grow the pool";
        EXPECT_FALSE(pipeline.has_non_finished_requests()) << "a rejected restore must not leave a request behind";
    }

    {
        auto fixed_config = make_scheduler_config();
        fixed_config.num_kv_blocks = 200;  // far more than the ~450-token snapshot needs, still small in memory
        ContinuousBatchingPipeline pipeline(model_path, fixed_config, session_snapshot_device());
        auto handle = pipeline.restore_state(49, snapshot_path.string(), make_greedy_config(4));
        while (pipeline.has_non_finished_requests()) {
            pipeline.step();
        }
        auto outputs = handle->read_all();
        ASSERT_FALSE(outputs.empty());
        EXPECT_EQ(outputs.front().generated_ids.size(), 4u);
    }
    std::filesystem::remove(snapshot_path);
}

TEST(TestSessionSnapshotEndToEnd, SaveRejectsARequestStillInPrefill) {
    const char* model_path = std::getenv(MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << MODEL_PATH_ENV << " is not set, skipping session snapshot end-to-end test.";
    }

    auto scheduler_config = make_scheduler_config();
    scheduler_config.dynamic_split_fuse = true;
    scheduler_config.max_num_batched_tokens = 64;  // forces the prompt below to prefill in several chunks
    ContinuousBatchingPipeline pipeline(model_path, scheduler_config, session_snapshot_device());

    const std::string prompt = repeat("The quick brown fox jumps over the lazy dog. ", 40);
    constexpr uint64_t REQUEST_ID = 50;
    auto handle = pipeline.add_request(REQUEST_ID, prompt, make_greedy_config(1000));
    pipeline.step();
    ASSERT_EQ(handle->get_status(), GenerationStatus::RUNNING);

    const auto snapshot_path =
        std::filesystem::temp_directory_path() / "openvino_genai_session_snapshot_prefill_test.bin";
    EXPECT_THROW(pipeline.save_state(REQUEST_ID, snapshot_path.string()), ov::Exception);
    EXPECT_FALSE(std::filesystem::exists(snapshot_path)) << "a rejected save must not leave a file behind";
}

TEST(TestSessionSnapshotEndToEnd, SaveRejectsCacheEviction) {
    const char* model_path = std::getenv(MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << MODEL_PATH_ENV << " is not set, skipping session snapshot end-to-end test.";
    }

    auto scheduler_config = make_scheduler_config();
    scheduler_config.use_cache_eviction = true;
    scheduler_config.cache_eviction_config =
        CacheEvictionConfig{32, 32, 128, AggregationMode::NORM_SUM};
    ContinuousBatchingPipeline pipeline(model_path, scheduler_config, session_snapshot_device());

    const std::string prompt = repeat("The quick brown fox jumps over the lazy dog. ", 10) + " Continue the story:";
    constexpr uint64_t REQUEST_ID = 46;
    auto handle = pipeline.add_request(REQUEST_ID, prompt, make_greedy_config(5));
    while (pipeline.has_non_finished_requests()) {
        pipeline.step();
    }

    const auto snapshot_path =
        std::filesystem::temp_directory_path() / "openvino_genai_session_snapshot_eviction_test.bin";
    EXPECT_THROW(pipeline.save_state(REQUEST_ID, snapshot_path.string()), ov::Exception);
}

/**
 * @brief Compares the time-to-first-token (TTFT) of resuming a long conversation from scratch
 * (full re-prefill) against restoring it from a session snapshot (no prefill, only the single
 * cached-tail token is re-threaded through one step). This is the exact win the feature targets:
 * skipping re-prefill of the full history after a process restart.
 */
void run_prefill_latency_comparison(const std::string& model_path) {
    // Long enough that prefill cost dominates per-request overhead and the TTFT gap is clearly visible.
    const std::string long_prompt = repeat("The quick brown fox jumps over the lazy dog. ", 400);

    auto scheduler_config = make_scheduler_config();
    // Large enough to prefill the whole prompt in as few steps as possible, so the baseline's TTFT
    // is dominated by actual prefill compute rather than scheduling overhead across many chunks.
    scheduler_config.max_num_batched_tokens = 8192;

    float baseline_ttft_ms = 0.0f;
    {
        ContinuousBatchingPipeline pipeline(model_path, scheduler_config, session_snapshot_device());
        constexpr uint64_t REQUEST_ID = 51;
        auto handle = pipeline.add_request(REQUEST_ID, long_prompt, make_greedy_config(1));
        while (pipeline.has_non_finished_requests()) {
            pipeline.step();
        }
        baseline_ttft_ms = handle->get_perf_metrics().get_ttft().mean;
    }

    const auto snapshot_path =
        std::filesystem::temp_directory_path() / "openvino_genai_session_snapshot_perf_test.bin";
    {
        ContinuousBatchingPipeline pipeline(model_path, scheduler_config, session_snapshot_device());
        constexpr uint64_t REQUEST_ID = 52;
        // Large budget so the request is still RUNNING once the (expensive) prefill has completed
        // and a couple of decode steps have run - save_state() requires a still-active request.
        auto handle = pipeline.add_request(REQUEST_ID, long_prompt, make_greedy_config(50));
        for (size_t i = 0; i < 5 && handle->get_status() == GenerationStatus::RUNNING; ++i) {
            pipeline.step();
        }
        ASSERT_EQ(handle->get_status(), GenerationStatus::RUNNING)
            << "the request finished naturally before it could be interrupted, test setup is invalid";
        pipeline.save_state(REQUEST_ID, snapshot_path.string());
    }

    float restore_ttft_ms = 0.0f;
    {
        ContinuousBatchingPipeline pipeline(model_path, scheduler_config, session_snapshot_device());
        constexpr uint64_t RESTORED_REQUEST_ID = 53;
        auto handle = pipeline.restore_state(RESTORED_REQUEST_ID, snapshot_path.string(), make_greedy_config(1));
        while (pipeline.has_non_finished_requests()) {
            pipeline.step();
        }
        restore_ttft_ms = handle->get_perf_metrics().get_ttft().mean;
    }
    std::filesystem::remove(snapshot_path);

    std::cout << "[ PERF     ] TTFT without session restore (full re-prefill): " << baseline_ttft_ms << " ms\n"
              << "[ PERF     ] TTFT with session restore (no re-prefill):      " << restore_ttft_ms << " ms"
              << std::endl;

    EXPECT_GT(baseline_ttft_ms, 0.0f);
    EXPECT_GT(restore_ttft_ms, 0.0f);
    EXPECT_LT(restore_ttft_ms, baseline_ttft_ms)
        << "restoring from a session snapshot should avoid the full-prefill cost paid by the baseline";
}

TEST(TestSessionSnapshotEndToEnd, RestoreAvoidsFullPrefillLatency) {
    const char* model_path = std::getenv(MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << MODEL_PATH_ENV << " is not set, skipping session snapshot end-to-end test.";
    }
    run_prefill_latency_comparison(model_path);
}

TEST(TestSessionSnapshotEndToEnd, HybridModelRestoreAvoidsFullPrefillLatency) {
    const char* model_path = std::getenv(VLM_MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << VLM_MODEL_PATH_ENV << " is not set, skipping hybrid session snapshot end-to-end test.";
    }
    run_prefill_latency_comparison(model_path);
}

namespace {

constexpr const char* SNAPSHOT_FORMAT_ENV = "OV_GENAI_SESSION_SNAPSHOT_FORMAT";

/// Sets (or clears, when `value` is null) an environment variable for the lifetime of the object.
class ScopedEnv {
public:
    ScopedEnv(const char* name, const char* value) : m_name(name) {
        if (const char* previous = std::getenv(name)) {
            m_previous = previous;
        }
        apply(value);
    }
    ~ScopedEnv() {
        apply(m_previous ? m_previous->c_str() : nullptr);
    }
    void set(const char* value) {
        apply(value);
    }

private:
    void apply(const char* value) {
#ifdef _WIN32
        _putenv_s(m_name.c_str(), value ? value : "");
#else
        if (value) {
            setenv(m_name.c_str(), value, 1);
        } else {
            unsetenv(m_name.c_str());
        }
#endif
    }

    std::string m_name;
    std::optional<std::string> m_previous;
};

SchedulerConfig make_offload_scheduler_config(bool hybrid = false) {
    auto config = make_scheduler_config();
    config.use_cache_offload = true;
    config.cache_offload_config.capacity_bytes = 256u * 1024u * 1024u;
    if (hybrid) {
        // Dynamic device cache; a linear-attention checkpoint is large, so the store holds only a few of them.
        config.cache_offload_config.linear_attention_capacity_bytes = 1024u * 1024u * 1024u;
    } else {
        // A small device cache, so a filler request can push an earlier conversation out of it (about 256 tokens).
        config.num_kv_blocks = session_snapshot_device().find("GPU") != std::string::npos ? 16 : 8;
    }
    return config;
}

// KV block size in tokens: 32 on CPU, 16 on GPU (the paged-attention page size).
size_t kv_block_tokens_for_device(const std::string& device) {
    return device.find("GPU") != std::string::npos ? 16 : 32;
}

/// Runs the pipeline until the request has produced `target` tokens in total.
void step_until_generated(ContinuousBatchingPipeline& pipeline,
                          GenerationHandle& handle,
                          size_t target,
                          std::vector<int64_t>& generated_ids) {
    while (generated_ids.size() < target && handle->get_status() == GenerationStatus::RUNNING) {
        pipeline.step();
        while (handle->can_read()) {
            for (auto& [seq_id, output] : handle->read()) {
                generated_ids.insert(generated_ids.end(), output.generated_ids.begin(), output.generated_ids.end());
            }
        }
    }
}

}  // namespace

// Format 4 keeps no KV/linear-attention payload in the snapshot file: the blocks are persisted in the pipeline's own
// prefix-caching offload store and a restore re-submits the history as an ordinary prompt. Snapshots are
// taken once where the last computed position ends a block and once where it does not (the partial tail).
void run_prefix_offload_round_trip(const std::string& model_path, bool hybrid) {
    ScopedEnv format(SNAPSHOT_FORMAT_ENV, "4");

    const std::string prompt = repeat("The quick brown fox jumps over the lazy dog. ", 10) + " Continue the story:";
    constexpr size_t TOTAL_NEW_TOKENS = 56;
    constexpr size_t SPLIT_RUN_BUDGET = 1000;
    const std::string device = session_snapshot_device();
    const size_t block_tokens = kv_block_tokens_for_device(device);

    std::vector<int64_t> baseline_ids;
    size_t prompt_tokens = 0;
    {
        ContinuousBatchingPipeline pipeline(model_path, make_offload_scheduler_config(hybrid), device);
        auto handle = pipeline.add_request(60, prompt, make_greedy_config(TOTAL_NEW_TOKENS));
        while (pipeline.has_non_finished_requests()) {
            pipeline.step();
        }
        auto outputs = handle->read_all();
        ASSERT_FALSE(outputs.empty());
        baseline_ids = outputs.front().generated_ids;
        // Includes any chat template the pipeline applies, unlike tokenizing the prompt directly.
        prompt_tokens = handle->get_perf_metrics().num_input_tokens;
    }
    ASSERT_EQ(baseline_ids.size(), TOTAL_NEW_TOKENS);
    ASSERT_GT(prompt_tokens, 0u);

    // A request with `g` generated tokens has N = prompt_tokens + g positions and N - 1 computed ones.
    size_t aligned_generated = 1;
    while ((prompt_tokens + aligned_generated - 1) % block_tokens != 0) {
        ++aligned_generated;
    }
    const std::vector<size_t> save_points = {aligned_generated, aligned_generated + 3};
    ASSERT_LT(save_points.back(), TOTAL_NEW_TOKENS);

    std::vector<std::filesystem::path> snapshot_paths;
    std::vector<std::vector<int64_t>> generated_at_save;
    {
        ContinuousBatchingPipeline pipeline(model_path, make_offload_scheduler_config(hybrid), device);
        auto handle = pipeline.add_request(61, prompt, make_greedy_config(SPLIT_RUN_BUDGET));
        std::vector<int64_t> generated_so_far;
        for (size_t point : save_points) {
            step_until_generated(pipeline, handle, point, generated_so_far);
            ASSERT_EQ(generated_so_far.size(), point);
            snapshot_paths.push_back(std::filesystem::temp_directory_path() /
                                     ("openvino_genai_session_snapshot_v4_" + std::to_string(point) + ".bin"));
            pipeline.save_state(61, snapshot_paths.back().string());
            std::cout << "[ SIZE     ] format 4 snapshot after " << point << " generated tokens: "
                      << std::filesystem::file_size(snapshot_paths.back()) << " bytes" << std::endl;
            generated_at_save.push_back(generated_so_far);
        }
        // Restoring next to the still-running original would share blocks it keeps writing to, so end it first.
        handle->stop();
        while (pipeline.has_non_finished_requests()) {
            pipeline.step();
        }

        // Displace the original conversation from the small device cache: what comes back has to be read from
        // the offload store, and the restore only succeeds if every cached position is recovered. A hybrid model
        // keeps its dynamic cache; there the linear-attention checkpoint at the last position is only in the store.
        if (!hybrid) {
            const std::string filler = repeat("Completely unrelated tokens to evict the cached prefix. ", 18);
            auto filler_results = pipeline.generate({filler}, {make_greedy_config(4)});
            ASSERT_FALSE(filler_results.at(0).m_generation_ids.at(0).empty());
        }

        for (size_t i = 0; i < save_points.size(); ++i) {
            const size_t remaining = TOTAL_NEW_TOKENS - save_points[i];
            auto restored = pipeline.restore_state(70 + i, snapshot_paths[i].string(), make_greedy_config(remaining));
            while (pipeline.has_non_finished_requests()) {
                pipeline.step();
            }
            auto outputs = restored->read_all();
            ASSERT_FALSE(outputs.empty());

            std::vector<int64_t> stitched_ids = generated_at_save[i];
            stitched_ids.insert(stitched_ids.end(), outputs.front().generated_ids.begin(), outputs.front().generated_ids.end());
            EXPECT_EQ(stitched_ids, baseline_ids)
                << "resuming from a format 4 snapshot taken after " << save_points[i] << " generated tokens diverged "
                << "from the uninterrupted baseline";
            // Everything but the last position must come from the cache; only that one is recomputed.
            EXPECT_EQ(restored->get_perf_metrics().get_num_prefix_cache_hit_tokens(),
                      prompt_tokens + save_points[i] - 1);
        }
    }
    for (const auto& path : snapshot_paths) {
        std::filesystem::remove(path);
    }
}

TEST(TestSessionSnapshotEndToEnd, PrefixOffloadFormatRestoredRequestContinuesIdenticallyToUninterruptedRun) {
    const char* model_path = std::getenv(MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << MODEL_PATH_ENV << " is not set, skipping session snapshot end-to-end test.";
    }
    run_prefix_offload_round_trip(model_path, /*hybrid=*/false);
}

// Qwen3.5 is hybrid and embeddings-routed: the snapshot also carries its input embeddings and position ids.
TEST(TestSessionSnapshotEndToEnd, PrefixOffloadFormatHybridModelRestoredRequestContinuesIdenticallyToUninterruptedRun) {
    const char* model_path = std::getenv(VLM_MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << VLM_MODEL_PATH_ENV << " is not set, skipping hybrid session snapshot end-to-end test.";
    }
    run_prefix_offload_round_trip(model_path, /*hybrid=*/true);
}

// A fresh pipeline has an empty (run-scoped) offload store, so the blocks a format 4 snapshot refers to are
// gone: the restore must fail explicitly instead of silently re-running prefill.
TEST(TestSessionSnapshotEndToEnd, PrefixOffloadFormatRestoreRejectsABlockThatIsNotInTheStore) {
    const char* model_path = std::getenv(MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << MODEL_PATH_ENV << " is not set, skipping session snapshot end-to-end test.";
    }
    ScopedEnv format(SNAPSHOT_FORMAT_ENV, "4");

    const std::string prompt = repeat("The quick brown fox jumps over the lazy dog. ", 10) + " Continue the story:";
    const auto snapshot_path = std::filesystem::temp_directory_path() / "openvino_genai_session_snapshot_v4_missing.bin";
    {
        ContinuousBatchingPipeline pipeline(model_path, make_offload_scheduler_config(), session_snapshot_device());
        auto handle = pipeline.add_request(62, prompt, make_greedy_config(1000));
        std::vector<int64_t> generated;
        step_until_generated(pipeline, handle, 5, generated);
        pipeline.save_state(62, snapshot_path.string());
    }

    ContinuousBatchingPipeline pipeline(model_path, make_offload_scheduler_config(), session_snapshot_device());
    try {
        pipeline.restore_state(63, snapshot_path.string(), make_greedy_config(5));
        ADD_FAILURE() << "restoring a snapshot whose blocks are not in the offload store must throw";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("missing from the prefix-caching offload store"), std::string::npos)
            << error.what();
    }
    EXPECT_FALSE(pipeline.has_non_finished_requests()) << "a rejected restore must not leave a request behind";
    std::filesystem::remove(snapshot_path);
}

TEST(TestSessionSnapshotEndToEnd, PrefixOffloadFormatSnapshotFileHoldsNoBlockPayload) {
    const char* model_path = std::getenv(MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << MODEL_PATH_ENV << " is not set, skipping session snapshot end-to-end test.";
    }
    ScopedEnv format(SNAPSHOT_FORMAT_ENV, "4");

    // Must fit the small device cache of the offload configuration (about 256 tokens).
    const std::string prompt = repeat("The quick brown fox jumps over the lazy dog. ", 15);
    const auto v4_path = std::filesystem::temp_directory_path() / "openvino_genai_session_snapshot_v4_size.bin";
    const auto v3_path = std::filesystem::temp_directory_path() / "openvino_genai_session_snapshot_v3_size.bin";
    {
        ContinuousBatchingPipeline pipeline(model_path, make_offload_scheduler_config(), session_snapshot_device());
        auto handle = pipeline.add_request(64, prompt, make_greedy_config(1000));
        std::vector<int64_t> generated;
        step_until_generated(pipeline, handle, 5, generated);
        pipeline.save_state(64, v4_path.string());
        format.set(nullptr);  // the same state, in the self-contained format
        pipeline.save_state(64, v3_path.string());
    }

    const auto v4_size = std::filesystem::file_size(v4_path);
    const auto v3_size = std::filesystem::file_size(v3_path);
    std::filesystem::remove(v4_path);
    std::filesystem::remove(v3_path);
    EXPECT_LT(v4_size * 100, v3_size) << "format 4 holds " << v4_size << " bytes against " << v3_size
                                      << " for the self-contained format, so it must be carrying block payload";
}

TEST(TestSessionSnapshotEndToEnd, PrefixOffloadFormatFallsBackToTheSelfContainedFormatWithoutOffload) {
    const char* model_path = std::getenv(MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << MODEL_PATH_ENV << " is not set, skipping session snapshot end-to-end test.";
    }
    ScopedEnv format(SNAPSHOT_FORMAT_ENV, "4");
    run_round_trip_test(model_path);
}
