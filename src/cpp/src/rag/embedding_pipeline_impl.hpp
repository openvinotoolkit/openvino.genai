// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/rag/embedding_pipeline.hpp"

namespace ov::genai {

class EmbeddingPipelineImpl {
public:
    virtual ~EmbeddingPipelineImpl() = default;

    virtual EmbedResult embed(const StringInputs& text, const ov::AnyMap& properties) = 0;

    virtual EmbedResult embed(const StringInputs& text,
                              const std::vector<ov::Tensor>& images,
                              const std::vector<ov::Tensor>& videos,
                              const std::vector<VideoMetadata>& videos_metadata,
                              const ov::AnyMap& properties) = 0;
};

}  // namespace ov::genai
