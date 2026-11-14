// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "include/base/pipeline_state.hpp"
#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"

class ASRPipelineWrapper : public Napi::ObjectWrap<ASRPipelineWrapper> {
public:
    ASRPipelineWrapper(const Napi::CallbackInfo& info);
    static Napi::Function get_class(Napi::Env env);
    Napi::Value init(const Napi::CallbackInfo& info);
    Napi::Value generate(const Napi::CallbackInfo& info);
    Napi::Value get_tokenizer(const Napi::CallbackInfo& info);
    Napi::Value get_generation_config(const Napi::CallbackInfo& info);
    Napi::Value set_generation_config(const Napi::CallbackInfo& info);

private:
    std::shared_ptr<ov::genai::ASRPipeline> pipe = nullptr;
    std::shared_ptr<PipelineState> state = std::make_shared<PipelineState>();
};
