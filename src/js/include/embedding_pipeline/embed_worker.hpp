// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>
#include <vector>
#include "openvino/genai/rag/embedding_pipeline.hpp"

using namespace Napi;

class EmbeddingPipelineEmbedWorker : public AsyncWorker {
    public:
        EmbeddingPipelineEmbedWorker(
            Function& callback,
            std::shared_ptr<ov::genai::EmbeddingPipeline>& pipe,
            ov::genai::StringInputs text,
            std::vector<ov::Tensor> images,
            std::vector<ov::Tensor> videos,
            std::vector<ov::genai::VideoMetadata> videos_metadata,
            ov::AnyMap properties
        );
        virtual ~EmbeddingPipelineEmbedWorker(){}

        void Execute() override;
        void OnOK() override;
    private:
        std::shared_ptr<ov::genai::EmbeddingPipeline>& pipe;
        ov::genai::StringInputs text;
        std::vector<ov::Tensor> images;
        std::vector<ov::Tensor> videos;
        std::vector<ov::genai::VideoMetadata> videos_metadata;
        ov::AnyMap properties;
        ov::genai::EmbedResult embed_result;
};
