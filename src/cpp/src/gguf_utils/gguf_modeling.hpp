// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstring>

#include "openvino/openvino.hpp"

/// \brief Convert a .gguf file into an ov::Model.
///
/// \param model_path            path to the .gguf file.
/// \param enable_save_ov_model  serialize the converted model next to the .gguf for re-use.
/// \param use_legacy_reader     select the pre-frontend GGUF reader instead of the OpenVINO GGUF
///                              frontend (ov::genai::gguf_reader property). Temporary fallback;
///                              its output model lacks the frontend's tokenizer rt_info, so
///                              callers must build the tokenizer from the .gguf path instead.
std::shared_ptr<ov::Model> create_from_gguf(const std::string& model_path,
                                            bool enable_save_ov_model,
                                            bool use_legacy_reader = false);
