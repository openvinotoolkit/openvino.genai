// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>

namespace ov::genai {
class CacheOrchestrator;
struct SchedulerConfig;
}

static constexpr size_t TEST_BLOCK_SIZE = 4;

std::shared_ptr<ov::genai::CacheOrchestrator> init_cache_orchestrator(
    ov::genai::SchedulerConfig scheduler_config,
    size_t block_size = TEST_BLOCK_SIZE,
    size_t num_layers = 1);
std::shared_ptr<ov::genai::CacheOrchestrator> init_hybrid_cache_orchestrator(
    ov::genai::SchedulerConfig scheduler_config,
    size_t kv_block_size = TEST_BLOCK_SIZE,
    size_t kv_num_layers = 1,
    size_t la_num_layers = 1,
    bool cap_la_pool = false);
