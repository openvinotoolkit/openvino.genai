// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/visibility.hpp"
#include "diffusion_model_type.hpp"
#include <openvino/runtime/properties.hpp>
#include <filesystem>
#include <variant>

namespace ov::genai {

class TransformerConfig {
public:
    std::vector<int> all_f_patch_size {1};
    std::vector<int> all_patch_size {2};
    std::vector<int> axes_dims {32, 48, 48};
    std::vector<int> axes_lens {1536, 512, 512};
    int cap_feat_dim {2560};
    int dim {3840};
    int in_channels {16};
    int n_heads {30};
    int n_kv_heads {30};
    int num_hidden_layers {36};
    int n_layers {30};
    int n_refiner_layers {2};
    double norm_eps {1e-05};
    std::variant<bool, std::string> qk_norm {true};
    double rope_theta {256.0};
    double t_scale {1000.0};

    TransformerConfig() = default;

    explicit TransformerConfig(const std::filesystem::path &config_path);

    TransformerConfig(const TransformerConfig &) = default;
};

}