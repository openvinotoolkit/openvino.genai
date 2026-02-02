// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
 
#include "transformer_config.hpp"
#include "json_utils.hpp"

#include <fstream>

namespace ov::genai {

TransformerConfig::TransformerConfig(const std::filesystem::path &config_path) {
    std::ifstream stream(config_path);
    OPENVINO_ASSERT(stream.is_open(), "Failed to open '", config_path, "' with transformer config");
    nlohmann::json parsed = nlohmann::json::parse(stream);
    using ov::genai::utils::read_json_param;
    read_json_param(parsed, "all_f_patch_size", all_f_patch_size);
    read_json_param(parsed, "all_patch_size", all_patch_size);
    read_json_param(parsed, "axes_dims", axes_dims);
    read_json_param(parsed, "axes_lens", axes_lens);
    read_json_param(parsed, "cap_feat_dim", cap_feat_dim);
    read_json_param(parsed, "dim", dim);
    read_json_param(parsed, "in_channels", in_channels);
    read_json_param(parsed, "n_heads", n_heads);
    read_json_param(parsed, "n_kv_heads", n_kv_heads);
    read_json_param(parsed, "n_layers", n_layers);
    read_json_param(parsed, "num_hidden_layers", num_hidden_layers);
    read_json_param(parsed, "n_refiner_layers", n_refiner_layers);
    read_json_param(parsed, "norm_eps", norm_eps);
    if (parsed.contains("qk_norm")) {
        if (parsed["qk_norm"].is_boolean()) {
            bool qk_norm_bool;
            read_json_param(parsed, "qk_norm", qk_norm_bool);
            qk_norm = qk_norm_bool;
        } else if (parsed["qk_norm"].is_string()) {
            std::string qk_norm_str;
            read_json_param(parsed, "qk_norm", qk_norm_str);
            qk_norm = qk_norm_str;
        }
    }
    read_json_param(parsed, "rope_theta", rope_theta);
    read_json_param(parsed, "t_scale", t_scale);
}

}

