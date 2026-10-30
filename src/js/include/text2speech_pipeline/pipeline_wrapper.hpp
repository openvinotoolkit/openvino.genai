// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "include/base/pipeline_state.hpp"
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
    std::shared_ptr<PipelineState> state = std::make_shared<PipelineState>();
};
