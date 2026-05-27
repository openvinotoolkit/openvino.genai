// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>

#include <napi.h>

#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

class Text2SpeechPipelineWrapper : public Napi::ObjectWrap<Text2SpeechPipelineWrapper> {
public:
    Text2SpeechPipelineWrapper(const Napi::CallbackInfo& info);
    static Napi::Function get_class(Napi::Env env);
    Napi::Value init(const Napi::CallbackInfo& info);
    Napi::Value generate(const Napi::CallbackInfo& info);
    Napi::Value get_generation_config(const Napi::CallbackInfo& info);
    Napi::Value set_generation_config(const Napi::CallbackInfo& info);

private:
    std::shared_ptr<ov::genai::Text2SpeechPipeline> pipe = nullptr;
    std::shared_ptr<std::atomic<bool>> is_initializing = std::make_shared<std::atomic<bool>>(false);
    std::shared_ptr<std::atomic<bool>> is_generating = std::make_shared<std::atomic<bool>>(false);
};
