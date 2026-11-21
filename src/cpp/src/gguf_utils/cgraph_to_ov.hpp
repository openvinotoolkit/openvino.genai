// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

#include "llama.h"
#include "openvino/core/model.hpp"

namespace ov::genai::gguf_utils {

LLAMA_API std::shared_ptr<ov::Model> convert_gguf_with_llama_cpp(const std::string& model_path);

}  // namespace ov::genai::gguf_utils
