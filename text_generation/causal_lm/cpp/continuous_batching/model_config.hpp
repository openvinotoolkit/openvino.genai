
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>

constexpr size_t BLOCK_SIZE = 16;

// TODO: extract from the model
constexpr int64_t SPECIAL_EOS_TOKEN = 2; // llm_model->get_rt_info()["eos_token_id"].as<int64_t>();

// TODO: compute based on the available memory
constexpr size_t NUM_BLOCKS = 3640;

// TODO: make as a parameter
constexpr auto kv_cache_precision = ov::element::f32;

const size_t X = BLOCK_SIZE / kv_cache_precision.size();
// TODO: take from model
constexpr size_t NUM_KV_HEADS = 12, NUM_HEADS = 12, HIDDEN_DIMS = 768, HEAD_SIZE = HIDDEN_DIMS / NUM_HEADS;
constexpr size_t NUM_DECODER_LAYERS = 12; // num KV cache pairs
