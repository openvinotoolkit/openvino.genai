// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "visual_language/embedding_model.hpp"

namespace ov::genai {

EmbeddingsModel::Ptr create_qwen3_asr_embeddings(const std::filesystem::path& model_dir,
                                                 float scale_emb,
                                                 const std::string& device,
                                                 const ov::AnyMap& properties);

EmbeddingsModel::Ptr create_qwen3_asr_embeddings(const std::string& model,
                                                 const ov::Tensor& weights,
                                                 float scale_emb,
                                                 const std::string& device,
                                                 const ov::AnyMap& properties);

}  // namespace ov::genai
