// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"

using namespace ov::genai;

namespace {

// Points at an OpenVINO IR text-generation model; the suite is skipped when it is not provided.
constexpr const char* MODEL_PATH_ENV = "OFFLOAD_E2E_MODEL";

// Small enough that the shared prefix cannot stay resident once an unrelated request is served.
constexpr size_t NUM_KV_BLOCKS = 8;

std::string repeat(const std::string& phrase, size_t times) {
    std::string result;
    for (size_t i = 0; i < times; ++i) {
        result += phrase;
    }
    return result;
}

SchedulerConfig make_scheduler_config(bool use_offload) {
    SchedulerConfig config;
    config.num_kv_blocks = NUM_KV_BLOCKS;
    config.max_num_batched_tokens = 256;
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
    ContinuousBatchingPipeline pipeline(model_path, make_scheduler_config(use_offload), "CPU");

    const std::string shared_prefix = repeat("The quick brown fox jumps over the lazy dog. ", 12);
    const std::string filler = repeat("Completely unrelated tokens to evict the cached prefix. ", 16);
    const auto generation_config = make_greedy_config();

    std::vector<std::string> outputs;
    for (const std::string& prompt : {shared_prefix + " First continuation:",
                                      filler,
                                      shared_prefix + " Second continuation:"}) {
        auto results = pipeline.generate({prompt}, {generation_config});
        EXPECT_EQ(results.size(), 1);
        EXPECT_FALSE(results[0].m_generation_ids.empty());
        outputs.push_back(results[0].m_generation_ids.at(0));
    }
    return outputs;
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

TEST(TestKVCacheOffloadEndToEnd, RejectsConfigurationsTheBackendCannotServe) {
    const char* model_path = std::getenv(MODEL_PATH_ENV);
    if (model_path == nullptr || *model_path == '\0') {
        GTEST_SKIP() << MODEL_PATH_ENV << " is not set, skipping KV cache offload end-to-end test.";
    }

    auto without_prefix_caching = make_scheduler_config(/*use_offload=*/true);
    without_prefix_caching.enable_prefix_caching = false;
    EXPECT_THROW(ContinuousBatchingPipeline(model_path, without_prefix_caching, "CPU"), ov::Exception);

    auto with_eviction = make_scheduler_config(/*use_offload=*/true);
    with_eviction.use_cache_eviction = true;
    EXPECT_THROW(ContinuousBatchingPipeline(model_path, with_eviction, "CPU"), ov::Exception);
}
