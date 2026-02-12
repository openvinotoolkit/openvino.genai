// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/genai/rag/text_rerank_pipeline.hpp"

class TextRerankPipelineWrapper : public Napi::ObjectWrap<TextRerankPipelineWrapper> {
public:
    TextRerankPipelineWrapper(const Napi::CallbackInfo& info);
    static Napi::Function get_class(Napi::Env env);
    Napi::Value init(const Napi::CallbackInfo& info);
    Napi::Value rerank(const Napi::CallbackInfo& info);

private:
    std::shared_ptr<ov::genai::TextRerankPipeline> pipe = nullptr;
    std::shared_ptr<bool> is_initializing = std::make_shared<bool>(false);
    std::shared_ptr<bool> is_reranking = std::make_shared<bool>(false);
};
