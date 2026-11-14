// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/runtime/core.hpp"

std::shared_ptr<ov::Model> get_dummy_model(ov::Core core, size_t num_layers);

// Same shape as get_dummy_model, plus a global `kv_chunk_table` (i32, dynamic [-1]) input, so
// KVCacheManager's has_kv_chunk_table_input() detects a chunked-capable model.
std::shared_ptr<ov::Model> get_dummy_model_with_kv_chunk_table(ov::Core core, size_t num_layers);

std::shared_ptr<ov::Model> get_dummy_hybrid_model(ov::Core core, size_t kv_num_layers, size_t la_num_layers);
