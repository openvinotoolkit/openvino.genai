// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include <atomic>
#include <thread>

#include "openvino/genai/llm_pipeline.hpp"

class LLMPipelineWrapper : public Napi::ObjectWrap<LLMPipelineWrapper> {
public:
    LLMPipelineWrapper(const Napi::CallbackInfo& info);

    static Napi::Function get_class(Napi::Env env);

    Napi::Value init(const Napi::CallbackInfo& info);
    Napi::Value generate(const Napi::CallbackInfo& info);
    Napi::Value start_chat(const Napi::CallbackInfo& info);
    Napi::Value finish_chat(const Napi::CallbackInfo& info);
    Napi::Value get_tokenizer(const Napi::CallbackInfo& info);
    Napi::Value get_generation_config(const Napi::CallbackInfo& info);
    Napi::Value set_generation_config(const Napi::CallbackInfo& info);

private:
    std::shared_ptr<ov::genai::LLMPipeline> pipe = nullptr;
    std::shared_ptr<std::atomic<bool>> is_initializing = std::make_shared<std::atomic<bool>>(false);
    std::shared_ptr<std::atomic<bool>> is_generating = std::make_shared<std::atomic<bool>>(false);
};
