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
        config.cache_offload_config.capacity_bytes = 8u * 1024u * 1024u;
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

    ov::AnyMap properties;
    properties.insert(ov::genai::scheduler_config(scheduler_config));
    VLMPipeline pipeline(model_path, "CPU", properties);

    const std::string shared_prefix = repeat("The quick brown fox jumps over the lazy dog. ", 40);
    const std::string filler = repeat("Completely unrelated tokens to evict the cached prefix. ", 80);
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

    // Large enough to hold a request, small enough that the shared prefix cannot survive the filler.
    constexpr size_t HYBRID_NUM_KV_BLOCKS = 24;

    const auto baseline = run_vlm_sequence(model_path, /*use_offload=*/false, HYBRID_NUM_KV_BLOCKS);
    const auto with_offload = run_vlm_sequence(model_path, /*use_offload=*/true, HYBRID_NUM_KV_BLOCKS);

    ASSERT_EQ(baseline.size(), with_offload.size());
    for (size_t i = 0; i < baseline.size(); ++i) {
        EXPECT_EQ(with_offload[i], baseline[i]) << "request " << i << " diverged once offload was enabled";
    }
}
