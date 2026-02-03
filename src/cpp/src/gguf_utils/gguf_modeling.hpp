// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstring>

#include "openvino/openvino.hpp"
#include "openvino/genai/llm_pipeline.hpp"

std::shared_ptr<ov::Model> create_from_gguf(const std::string& model_path, const ov::genai::OVModelQuantizeMode& quantize_mode, bool should_save_file);
