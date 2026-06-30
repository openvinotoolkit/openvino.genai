// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include <cstdarg>

#include <openvino/openvino.hpp>

#include "gguf_utils/gguf.hpp"

ov::Output<ov::Node> make_lm_head(
    const std::string& key,
    const ov::Output<ov::Node>& input,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    const ov::Output<ov::Node>& embeddings_node,
    gguf_tensor_type qtype);

ov::Output<ov::Node> make_rms_norm(
    const std::string& key,
    const ov::Output<ov::Node>& input,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    float epsilon);

std::tuple<ov::Output<ov::Node>, ov::Output<ov::Node>> make_embedding(
    const std::string& key,
    const ov::Output<ov::Node>& input,
    const std::unordered_map<std::string, ov::Tensor>& consts,
    gguf_tensor_type qtype);

std::tuple<ov::Output<ov::Node>, 
           ov::SinkVector,
           ov::Output<ov::Node>,
           std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>,
           std::shared_ptr<ov::Node>> 
    layer(const std::map<std::string, GGUFMetaData>& configs,
        std::unordered_map<std::string, ov::Tensor>& consts,
        std::unordered_map<std::string, gguf_tensor_type>& qtypes,
        int layer_idx,
        const ov::Output<ov::Node>& hidden_states,
        const ov::Output<ov::Node>& attn_mask,
        const ov::Output<ov::Node>& causal_mask,
        const ov::Output<ov::Node>& position_ids,
        const ov::Output<ov::Node>& rope_const,
        const ov::Output<ov::Node>& beam_idx,
        const ov::Output<ov::Node>& batch_dim,
        const ov::Output<ov::Node>& hidden_dim,
        const std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>& cos_sin_cached,
        const std::shared_ptr<ov::Node>& output_shape);

// RoPE scaling parameters. Defaults are a no-op (type == "none"/"") so that
// non-scaled models are unaffected. When type == "yarn", NTK-by-parts (YaRN)
// frequency interpolation is applied using the beta_fast/beta_slow ramp.
struct RopeScalingParams {
    std::string type = "";            // "", "none", "linear", or "yarn"
    float factor = 1.0f;              // scaling factor (s)
    int64_t original_context_length = 0;  // pre-extension context length
    float yarn_beta_fast = 32.0f;     // llama.cpp default
    float yarn_beta_slow = 1.0f;      // llama.cpp default
};

ov::Output<ov::Node> init_rope(
    int64_t head_dim,
    int64_t max_position_embeddings = 2048,
    float base = 10000.0f,
    float scaling_factor = 1.0f,
    const RopeScalingParams& rope_scaling = RopeScalingParams{});
