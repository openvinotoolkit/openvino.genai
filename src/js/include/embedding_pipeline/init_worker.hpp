// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <napi.h>
#include "openvino/genai/rag/embedding_pipeline.hpp"

using namespace Napi;

class EmbeddingPipelineInitWorker : public AsyncWorker {
    public:
        EmbeddingPipelineInitWorker(
            Function& callback,
            std::shared_ptr<ov::genai::EmbeddingPipeline>& pipe,
            std::filesystem::path model_path,
            std::string device,
            Object properties
        );
        virtual ~EmbeddingPipelineInitWorker(){}
        void Execute() override;
        void OnOK() override;
    private:
        std::shared_ptr<ov::genai::EmbeddingPipeline>& pipe;
        std::filesystem::path model_path;
        std::string device;
        ov::AnyMap properties;
};
