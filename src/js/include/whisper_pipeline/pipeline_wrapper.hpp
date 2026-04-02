// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include <atomic>

#include "openvino/genai/whisper_pipeline.hpp"

class WhisperPipelineWrapper : public Napi::ObjectWrap<WhisperPipelineWrapper> {
public:
    WhisperPipelineWrapper(const Napi::CallbackInfo& info);
    static Napi::Function get_class(Napi::Env env);
    Napi::Value init(const Napi::CallbackInfo& info);
    Napi::Value generate(const Napi::CallbackInfo& info);
    Napi::Value get_tokenizer(const Napi::CallbackInfo& info);
    Napi::Value get_generation_config(const Napi::CallbackInfo& info);
    Napi::Value set_generation_config(const Napi::CallbackInfo& info);

private:
    std::shared_ptr<ov::genai::WhisperPipeline> pipe = nullptr;
    std::shared_ptr<std::atomic<bool>> is_initializing = std::make_shared<std::atomic<bool>>(false);
    std::shared_ptr<std::atomic<bool>> is_generating = std::make_shared<std::atomic<bool>>(false);
};
