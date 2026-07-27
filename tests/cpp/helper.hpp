// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/runtime/core.hpp"

#include "continuous_batching/cache/cache_orchestrator.hpp"

std::shared_ptr<ov::Model> get_dummy_model(ov::Core core, size_t num_layers);

std::shared_ptr<ov::Model> get_dummy_hybrid_model(ov::Core core, size_t kv_num_layers, size_t la_num_layers);

// The owned linear-attention rows that are not the current live one (scratch rows), derived from
// the public registry accessors. Mirrors what the scheduler builds inline for speculative paging.
std::vector<size_t> linear_attention_scratch_blocks(const std::shared_ptr<ov::genai::CacheOrchestrator>& orchestrator,
                                                     uint64_t seq_id);
