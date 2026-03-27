// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include "openvino/runtime/infer_request.hpp"
#include "openvino/genai/scheduler_config.hpp"
#include "continuous_batching/cache/cache_orchestrator.hpp"
#include "continuous_batching/cache/kv_cache_manager.hpp"
#include "utils.hpp"

namespace ov::genai {

/**
 * @brief Detect model cache types, create managers and block managers, normalize config, and
 *        return a fully populated CacheOrchestrator.
 *
 * For each known cache type a static heuristic checks whether the compiled model has the
 * corresponding inputs.  When a type is detected its ICacheManager and BlockManager are
 * created and registered in the orchestrator.
 *
 * @param infer_request         The inference request (provides compiled model info).
 * @param[in,out] config        Scheduler configuration.  On input, may have zero num_kv_blocks
 *                              (to be derived from cache_size).  On output, num_kv_blocks is
 *                              filled in if it was zero and cache_size > 0.
 * @param get_available_memory  Callable that returns available device memory in bytes given
 *                              the device string and number of decoder layers.
 * @return Fully populated CacheOrchestrator.  Metadata (num_layers, block_size, device, etc.)
 *         is available through the orchestrator's query methods.
 */
inline std::shared_ptr<CacheOrchestrator> setup_cache(
        ov::InferRequest& infer_request,
        SchedulerConfig& config,
        std::function<size_t(const std::string&, size_t)> get_available_memory) {
    ov::CompiledModel compiled_model = infer_request.get_compiled_model();

    auto orchestrator = std::make_shared<CacheOrchestrator>();

    // Aggregate block-size-in-bytes across all detected cache types (for config normalization).
    size_t aggregate_block_size_in_bytes = 0;
    size_t total_num_layers = 0;

    // ------------------------------------------------------------------
    //  KV Cache detection
    // ------------------------------------------------------------------
    if (KVCacheManager::has_cache_inputs(compiled_model)) {
        auto kv_manager = std::make_shared<KVCacheManager>(infer_request);

        total_num_layers += kv_manager->get_num_decoder_layers();
        aggregate_block_size_in_bytes += kv_manager->get_block_size_in_bytes();

        // Layer IDs for KV cache: layers [0 .. num_decoder_layers)
        std::vector<size_t> layer_ids(kv_manager->get_num_decoder_layers());
        std::iota(layer_ids.begin(), layer_ids.end(), 0);

        // Config normalization (must happen before block manager creation).
        // Query available memory now that num_decoder_layers is known.
        size_t total_available_memory = get_available_memory(kv_manager->get_device(), total_num_layers);
        if (config.num_kv_blocks == 0 && config.cache_size > 0) {
            size_t size_in_bytes = config.cache_size * 1024 * 1024 * 1024;
            OPENVINO_ASSERT(size_in_bytes <= total_available_memory,
                            "Requested KV-cache size is larger than available memory size on the system.");
            config.num_kv_blocks = size_in_bytes / aggregate_block_size_in_bytes;
        }
        if (config.num_kv_blocks > 0) {
            size_t size_in_bytes = aggregate_block_size_in_bytes * config.num_kv_blocks;
            OPENVINO_ASSERT(size_in_bytes <= total_available_memory,
                            "Requested number of KV-blocks require more memory than available on the system.");
        }

        auto block_manager = std::make_shared<BlockManager>(
            config.num_kv_blocks,
            config.enable_prefix_caching,
            kv_manager->get_block_size(),
            kv_manager->get_num_decoder_layers());

        orchestrator->register_cache_type(CacheType::KV_CACHE, kv_manager, block_manager, layer_ids);
    }

    // Future cache types (e.g. LINEAR_ATTENTION, SLIDING_WINDOW) follow the same pattern:
    //   if (SomeManager::has_cache_inputs(compiled_model)) { ... }

    OPENVINO_ASSERT(!orchestrator->get_registered_types().empty(),
                    "No supported cache types detected in the model");

    return orchestrator;
}

}  // namespace ov::genai
