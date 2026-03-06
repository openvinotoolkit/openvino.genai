// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <array>
#include "openvino/runtime/tensor.hpp"

namespace ov::genai::module {

struct Qwen3_5VisionConfig {
    std::string model_type = "qwen3_5";
    int32_t depth = 0;
    int32_t hidden_size = 0;
    std::string hidden_act = "gelu_pytorch_tanh";
    int32_t intermediate_size = 0;
    int32_t num_heads = 0;
    int32_t in_channels = 3;
    int32_t patch_size = 16;
    int32_t spatial_merge_size = 2;
    int32_t temporal_patch_size = 2;
    int32_t out_hidden_size = 0;
    int32_t num_position_embeddings = 0;
    std::vector<int32_t> deepstack_visual_indexes;
    float initializer_range = 0.02f;

    static Qwen3_5VisionConfig from_json_file(const std::filesystem::path& path);
    int32_t head_dim() const;
};

struct Qwen3_5VisionPreprocessConfig {
    int64_t min_pixels = 56 * 56;
    int64_t max_pixels = 28 * 28 * 1280;
    int32_t patch_size = 16;
    int32_t temporal_patch_size = 2;
    int32_t merge_size = 2;
    std::array<float, 3> image_mean = {0.5f, 0.5f, 0.5f};
    std::array<float, 3> image_std = {0.5f, 0.5f, 0.5f};
    bool do_resize = true;

    static Qwen3_5VisionPreprocessConfig from_json_file(const std::filesystem::path& path);
};

struct Qwen3_5VisionEmbeddingResult {
    ov::Tensor position_ids;
    ov::Tensor visual_pos_mask;
    ov::Tensor rope_deltas;
    ov::Tensor visual_embeds;
};

}