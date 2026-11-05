// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"

using namespace ov::genai;

namespace {

// Points at an OpenVINO IR text-generation model; the suite is skipped when it is not provided.
constexpr const char* MODEL_PATH_ENV = "OFFLOAD_E2E_MODEL";

// Points at a visual-language model directory, used to cover a hybrid attention stack.
constexpr const char* VLM_MODEL_PATH_ENV = "OFFLOAD_E2E_VLM_MODEL";

// Inference device for the comparison, defaults to CPU.
constexpr const char* DEVICE_ENV = "OFFLOAD_E2E_DEVICE";

std::string offload_e2e_device() {
    const char* device = std::getenv(DEVICE_ENV);
    return (device == nullptr || *device == '\0') ? std::string{"CPU"} : std::string{device};
}

/**
 * @return A block count giving every device the same ~256 token budget.
 *
 * The budget has to sit between the filler prompt and the sum of both prompts, otherwise either the
 * filler cannot be scheduled or it never displaces the shared prefix. GPU pages hold 16 tokens
 * against 32 on CPU, so the count cannot be a single constant.
 */
size_t kv_blocks_for_device(const std::string& device) {
    return device.find("GPU") != std::string::npos ? 16 : 8;
}

std::string repeat(const std::string& phrase, size_t times) {
    std::string result;
    for (size_t i = 0; i < times; ++i) {
        result += phrase;
    }
    return result;
}

SchedulerConfig make_scheduler_config(bool use_offload) {
    SchedulerConfig config;
    config.num_kv_blocks = kv_blocks_for_device(offload_e2e_device());
    config.max_num_batched_tokens = 512;
    config.max_num_seqs = 4;
    config.dynamic_split_fuse = false;
    config.enable_prefix_caching = true;
    config.use_cache_offload = use_offload;
    if (use_offload) {
        // Room for far more blocks than the device cache holds, so eviction lands on disk.
        config.cache_offload_config.capacity_bytes = 256u * 1024u * 1024u;
        // Also exercise linear attention cache offload for hybrid models (e.g. Qwen3.5); harmless
        // when the model has no linear attention layers, since enable_linear_attention_cache_offload
        // is only invoked when a linear attention cache manager was actually registered.
        config.cache_offload_config.linear_attention_capacity_bytes = 128u * 1024u * 1024u;
        // Freeing a sequence releases its whole block table at once, so a staging pool smaller than
        // that burst silently drops most of the blocks and the disk cache fills up only slowly.
        config.cache_offload_config.buffer_slots = 128;
    }
    return config;
}

GenerationConfig make_greedy_config() {
    GenerationConfig config;
    config.max_new_tokens = 10;
    config.do_sample = false;
    config.num_beams = 1;
    config.ignore_eos = true;
    return config;
}

/**
 * @brief Drives a request sequence that forces the shared prefix out of the device cache.
 *
 * The unrelated middle request fills the cache, so serving the last request again requires the
 * prefix to come back from wherever it was kept.
 */
std::vector<std::string> run_sequence(const std::string& model_path, bool use_offload) {
    ContinuousBatchingPipeline pipeline(model_path, make_scheduler_config(use_offload), offload_e2e_device());

    const std::string shared_prefix = repeat("The quick brown fox jumps over the lazy dog. ", 6);
    const std::string filler = repeat("Completely unrelated tokens to evict the cached prefix. ", 18);
    const auto generation_config = make_greedy_config();

    std::vector<std::string> outputs;
    for (const std::string& prompt : {shared_prefix + " First continuation:",
                                      filler,
                                      shared_prefix + " Second continuation:"}) {
        auto results = pipeline.generate({prompt}, {generation_config});
        EXPECT_EQ(results.size(), 1);
        EXPECT_FALSE(results[0].m_generation_ids.empty());
        // An empty completion would make the comparison below pass without proving anything.
        EXPECT_FALSE(results[0].m_generation_ids.at(0).empty()) << "the pipeline produced no text";
        outputs.push_back(results[0].m_generation_ids.at(0));
    }
    return outputs;
}

/// @return Bytes the GPU driver reports as resident in device memory, host allocations excluded.
uint64_t gpu_device_bytes() {
    ov::Core core;
    uint64_t device_bytes = 0;
    for (const auto& [category, bytes] : core.get_property("GPU", ov::intel_gpu::memory_statistics)) {
        if (category == "cl_mem" || category == "usm_device") {
            device_bytes += bytes;
        }
    }
    return device_bytes;
}

/// @return Device bytes held while the pipeline is alive, measured against the current baseline.
uint64_t measure_device_bytes(const std::string& model_path, bool use_offload) {
    const uint64_t before = gpu_device_bytes();
    uint64_t held = 0;
    {
        ContinuousBatchingPipeline pipeline(model_path, make_scheduler_config(use_offload), offload_e2e_device());
        pipeline.generate({repeat("The quick brown fox jumps over the lazy dog. ", 6)}, {make_greedy_config()});
        held = gpu_device_bytes() - before;
    }
    return held;
}

/**
 * @brief Mean time to first token of the request that follows the cache-filling one.
 *
 * That request is the only one whose prefix may still be available, so its prefill cost is what
 * separates a cache hit from a full recomputation. Prompts are long on purpose: with a short one the
 * per-request overhead dwarfs the prefill and the comparison measures nothing.
 */
float measure_repeated_prefix_ttft(const std::string& model_path, size_t num_kv_blocks, bool use_offload) {
    auto scheduler_config = make_scheduler_config(use_offload);
    scheduler_config.num_kv_blocks = num_kv_blocks;
    scheduler_config.max_num_batched_tokens = 4096;
    ContinuousBatchingPipeline pipeline(model_path, scheduler_config, offload_e2e_device());

    const std::string shared_prefix = repeat("The quick brown fox jumps over the lazy dog. ", 80);
    // Long enough to displace the shared prefix from the halved cache, short enough to still fit in
    // it; a filler that does not fit is never scheduled and then nothing gets displaced at all.
    const std::string filler = repeat("Completely unrelated tokens to evict the cached prefix. ", 120);
    const auto generation_config = make_greedy_config();

    // One untimed round first: on GPU the very first requests pay kernel compilation, which would
    // otherwise be charged entirely to whichever configuration runs first.
    auto run_round = [&](size_t iteration) {
        pipeline.generate({shared_prefix + " First continuation:"}, {generation_config});
        // The filler has to differ every round, otherwise it is served from the cache from the
        // second round on and stops displacing the shared prefix.
        auto filler_results =
            pipeline.generate({"Round " + std::to_string(iteration) + ". " + filler}, {generation_config});
        EXPECT_FALSE(filler_results.at(0).m_generation_ids.at(0).empty())
            << "the filler request was dropped, so it never displaced the shared prefix";
        return pipeline.generate({shared_prefix + " Second continuation:"}, {generation_config});
    };

    constexpr size_t NUM_ITERATIONS = 5;
    run_round(0);

    float total_ttft_ms = 0.0f;
    for (size_t iteration = 0; iteration < NUM_ITERATIONS; ++iteration) {
        total_ttft_ms += run_round(iteration + 1).at(0).perf_metrics.get_ttft().mean;
    }
    return total_ttft_ms / NUM_ITERATIONS;
}

/// @return A block count holding the filler prompt but not the filler plus the shared prefix.
size_t perf_halved_blocks(const std::string& device) {
    return device.find("GPU") != std::string::npos ? 128 : 64;
}

}  // namespace

TEST(TestKVCacheOffloadEndToEnd, ProducesTheSameTextAsWithoutOffload) {
    const char* model_path = std::getenv(MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << MODEL_PATH_ENV << " is not set, skipping KV cache offload end-to-end test.";
    }

    const auto baseline = run_sequence(model_path, /*use_offload=*/false);
    const auto with_offload = run_sequence(model_path, /*use_offload=*/true);

    ASSERT_EQ(baseline.size(), with_offload.size());
    for (size_t i = 0; i < baseline.size(); ++i) {
        EXPECT_EQ(with_offload[i], baseline[i]) << "request " << i << " diverged once offload was enabled";
    }
}

TEST(TestKVCacheOffloadEndToEnd, DoesNotGrowDeviceMemory) {
    const char* model_path = std::getenv(MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << MODEL_PATH_ENV << " is not set, skipping KV cache offload end-to-end test.";
    }
    if (offload_e2e_device().find("GPU") == std::string::npos) {
        GTEST_SKIP() << "Device memory statistics are only published by the GPU plugin.";
    }

    const uint64_t without_offload = measure_device_bytes(model_path, /*use_offload=*/false);
    const uint64_t with_offload = measure_device_bytes(model_path, /*use_offload=*/true);

    std::cout << "[ MEMORY   ] device bytes without offload: " << without_offload << "\n"
              << "[ MEMORY   ] device bytes with offload:    " << with_offload << std::endl;

    // Offload keeps the device cache the same size and stages transfers through host memory, so it
    // must not add device allocations of its own.
    EXPECT_LE(with_offload, without_offload)
        << "enabling offload increased device memory by " << (with_offload - without_offload) << " bytes";
}

TEST(TestKVCacheOffloadEndToEnd, ComparesHalvedCacheWithAndWithoutOffload) {
    const char* model_path = std::getenv(MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << MODEL_PATH_ENV << " is not set, skipping KV cache offload end-to-end test.";
    }

    // The halved cache cannot hold the shared prefix across the filler request, the full one can.
    const size_t halved_blocks = perf_halved_blocks(offload_e2e_device());
    const size_t full_blocks = halved_blocks * 2;

    const float full_cache = measure_repeated_prefix_ttft(model_path, full_blocks, /*use_offload=*/false);
    const float halved_cache = measure_repeated_prefix_ttft(model_path, halved_blocks, /*use_offload=*/false);
    const float halved_cache_offloaded = measure_repeated_prefix_ttft(model_path, halved_blocks, /*use_offload=*/true);

    std::cout << "[ PERF     ] TTFT of the repeated prefix, mean over 5 rounds\n"
              << "[ PERF     ]   A  " << full_blocks << " blocks, no offload : " << full_cache << " ms\n"
              << "[ PERF     ]   B  " << halved_blocks << " blocks, no offload : " << halved_cache << " ms\n"
              << "[ PERF     ]   C  " << halved_blocks << " blocks, offload    : " << halved_cache_offloaded << " ms"
              << std::endl;

    // Timings are reported rather than asserted: whether restoring from disk beats recomputing the
    // prefix depends on how expensive prefill is for the model under test.
    EXPECT_GT(full_cache, 0.0f);
    EXPECT_GT(halved_cache, 0.0f);
    EXPECT_GT(halved_cache_offloaded, 0.0f);
}

TEST(TestKVCacheOffloadEndToEnd, RejectsConfigurationsTheBackendCannotServe) {
    const char* model_path = std::getenv(MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << MODEL_PATH_ENV << " is not set, skipping KV cache offload end-to-end test.";
    }

    auto without_prefix_caching = make_scheduler_config(/*use_offload=*/true);
    without_prefix_caching.enable_prefix_caching = false;
    EXPECT_THROW(ContinuousBatchingPipeline(model_path, without_prefix_caching, offload_e2e_device()), ov::Exception);

    auto with_eviction = make_scheduler_config(/*use_offload=*/true);
    with_eviction.use_cache_eviction = true;
    EXPECT_THROW(ContinuousBatchingPipeline(model_path, with_eviction, offload_e2e_device()), ov::Exception);
}

namespace {

/// Same request sequence as above, driven through a visual-language pipeline with text-only prompts.
std::vector<std::string> run_vlm_sequence(const std::string& model_path, bool use_offload, size_t num_kv_blocks) {
    auto scheduler_config = make_scheduler_config(use_offload);
    scheduler_config.num_kv_blocks = num_kv_blocks;
    const bool is_gpu = offload_e2e_device().find("GPU") != std::string::npos;
    if (is_gpu) {
        scheduler_config.cache_interval_multiplier = 4;
        scheduler_config.num_linear_attention_blocks = 6;
    }
    // The filler prompt below tokenizes much longer under this model's tokenizer than the
    // default 512 budget assumes.
    scheduler_config.max_num_batched_tokens = 2048;

    ov::AnyMap properties;
    properties.insert(ov::genai::scheduler_config(scheduler_config));
    VLMPipeline pipeline(model_path, offload_e2e_device(), properties);

    const std::string shared_prefix = repeat("The quick brown fox jumps over the lazy dog. ", is_gpu ? 12 : 40);
    const std::string filler = repeat("Completely unrelated tokens to evict the cached prefix. ", is_gpu ? 30 : 80);
    const auto generation_config = make_greedy_config();

    std::vector<std::string> outputs;
    for (const std::string& prompt : {shared_prefix + " First continuation:",
                                      filler,
                                      shared_prefix + " Second continuation:"}) {
        const auto started = std::chrono::steady_clock::now();
        auto result = pipeline.generate(prompt, ov::genai::generation_config(generation_config));
        const auto elapsed = std::chrono::steady_clock::now() - started;
        std::cout << "[offload=" << std::boolalpha << use_offload << "] request " << outputs.size() << " took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() << " ms" << std::endl;
        EXPECT_FALSE(result.texts.empty());
        outputs.push_back(result.texts.at(0));
    }
    return outputs;
}

}  // namespace

TEST(TestKVCacheOffloadEndToEnd, HybridModelProducesTheSameTextAsWithoutOffload) {
    const char* model_path = std::getenv(VLM_MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << VLM_MODEL_PATH_ENV << " is not set, skipping hybrid KV cache offload end-to-end test.";
    }

    // The GPU prompts are shortened to fit the device cache while still forcing prefix eviction.
    const size_t hybrid_num_kv_blocks = offload_e2e_device().find("GPU") != std::string::npos ? 24 : 32;

    const auto baseline = run_vlm_sequence(model_path, /*use_offload=*/false, hybrid_num_kv_blocks);
    const auto with_offload = run_vlm_sequence(model_path, /*use_offload=*/true, hybrid_num_kv_blocks);

    ASSERT_EQ(baseline.size(), with_offload.size());
    for (size_t i = 0; i < baseline.size(); ++i) {
        EXPECT_EQ(with_offload[i], baseline[i]) << "request " << i << " diverged once offload was enabled";
    }
}
