
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>

// for CPU we use block_size = 1 instead of vLLM's = 1
constexpr size_t BLOCK_SIZE = 16;

// TODO: extract from the model
constexpr int64_t SPECIAL_EOS_TOKEN = 2; // llm_model->get_rt_info()["eos_token_id"].as<int64_t>();

// TODO: compute based on the available memory
constexpr size_t NUM_BLOCKS = 36400;

// TODO: make as a parameter
constexpr auto kv_cache_precision = ov::element::f16;

// TODO: take from model
constexpr size_t NUM_KV_HEADS = 12, NUM_HEADS = 12, HIDDEN_DIMS = 768, HEAD_SIZE = HIDDEN_DIMS / NUM_HEADS;
constexpr size_t NUM_DECODER_LAYERS = 12; // num KV cache pairs
