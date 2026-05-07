// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstring>

#include "openvino/openvino.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"

std::shared_ptr<ov::Model> create_from_gguf(const std::string& model_path, const bool enable_save_ov_model);

ov::genai::ModelsMap create_models_map_from_gguf(const std::filesystem::path& gguf_path, bool enable_save_ov_model);
