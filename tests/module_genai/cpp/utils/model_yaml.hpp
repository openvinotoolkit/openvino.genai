
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace TEST_MODEL {
std::string get_device();

// Return model full path for Qwen2.5-VL-3B-Instruct INT4 model.
std::string Qwen2_5_VL_3B_Instruct_INT4();
// Return yaml content string for Qwen2.5-VL-3B-Instruct model pipeline configuration.
std::string get_qwen2_5_vl_config_yaml(const std::string& model_path, const std::string& device = "CPU");
};  // namespace TEST_MODEL