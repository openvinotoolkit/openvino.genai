// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/runtime/core.hpp"

using namespace ov::genai;

namespace {

// Overrides the model path below; the suite falls back to a known-good local Qwen3-8B export so it
// runs out of the box on this machine.
constexpr const char* MODEL_PATH_ENV = "CHUNKED_KV_CACHE_E2E_MODEL";
constexpr const char* DEFAULT_MODEL_PATH = "C:\\Users\\gta\\Downloads\\qwen3-8b\\pytorch\\ov\\OV_FP16-4BIT_DEFAULT";

std::string model_path() {
    const char* path = std::getenv(MODEL_PATH_ENV);
    return (path == nullptr || *path == '\0') ? std::string{DEFAULT_MODEL_PATH} : std::string{path};
}

SchedulerConfig make_scheduler_config(bool use_chunked_kv_cache) {
    SchedulerConfig config;
    config.num_kv_blocks = 64;
    config.max_num_batched_tokens = 512;
    config.max_num_seqs = 2;
    config.dynamic_split_fuse = false;
    config.use_chunked_kv_cache = use_chunked_kv_cache;
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

std::string run_generation(bool use_chunked_kv_cache) {
    // Chunked KV cache currently only supports the uncompressed cache path;
    // GPU defaults to a quantized u8 cache, so it must be forced back to f16 for both runs to
    // get a valid, apples-to-apples comparison.
    ov::AnyMap properties{{"KV_CACHE_PRECISION", ov::element::f16}};
    ContinuousBatchingPipeline pipeline(model_path(), make_scheduler_config(use_chunked_kv_cache), "GPU", properties);

    const std::string prompt = repeat("The quick brown fox jumps over the lazy dog. ", 20);
    auto results = pipeline.generate({prompt}, {make_greedy_config(/*max_new_tokens=*/40)});
    EXPECT_EQ(results.size(), 1);
    EXPECT_FALSE(results[0].m_generation_ids.empty());
    EXPECT_FALSE(results[0].m_generation_ids.at(0).empty()) << "the pipeline produced no text";
    return results[0].m_generation_ids.at(0);
}

}  // namespace

// The first real-model end-to-end check for use_chunked_kv_cache=true.
// Chunking only changes how KV cache device memory is allocated/addressed; it must not change what
// the model computes, so greedy decoding must produce byte-for-byte identical text either way.
TEST(TestChunkedKVCache, ProducesTheSameTextAsWithoutChunking) {
    auto devices = ov::Core().get_available_devices();
    if (std::find(devices.begin(), devices.end(), "GPU") == devices.end()) {
        GTEST_SKIP() << "No GPU device available on this machine.";
    }
    if (!std::filesystem::exists(model_path())) {
        GTEST_SKIP() << "Model not found at " << model_path() << "; set " << MODEL_PATH_ENV << " to override.";
    }

    const auto baseline = run_generation(/*use_chunked_kv_cache=*/false);
    const auto chunked = run_generation(/*use_chunked_kv_cache=*/true);

    EXPECT_EQ(chunked, baseline) << "generated text diverged once chunking was enabled";
}
