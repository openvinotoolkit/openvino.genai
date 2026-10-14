// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <napi.h>
#include "openvino/genai/rag/embedding_pipeline.hpp"

class EmbeddingPipelineWrapper : public Napi::ObjectWrap<EmbeddingPipelineWrapper> {
    public:
        EmbeddingPipelineWrapper(const Napi::CallbackInfo& info);
        static Napi::Function get_class(Napi::Env env);
        Napi::Value init(const Napi::CallbackInfo& info);
        Napi::Value embed(const Napi::CallbackInfo& info);
    private:
        std::shared_ptr<ov::genai::EmbeddingPipeline> pipe = nullptr;
        std::shared_ptr<std::atomic<bool>> is_embedding = std::make_shared<std::atomic<bool>>(false);
};
