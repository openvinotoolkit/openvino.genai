// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "include/base/pipeline_state.hpp"
#include "openvino/genai/image_generation/text2image_pipeline.hpp"

class Text2ImagePipelineWrapper : public Napi::ObjectWrap<Text2ImagePipelineWrapper> {
public:
    Text2ImagePipelineWrapper(const Napi::CallbackInfo& info);
    static Napi::Function get_class(Napi::Env env);

    Napi::Value init(const Napi::CallbackInfo& info);
    Napi::Value generate(const Napi::CallbackInfo& info);
    Napi::Value decode(const Napi::CallbackInfo& info);
    Napi::Value get_performance_metrics(const Napi::CallbackInfo& info);
    Napi::Value get_generation_config(const Napi::CallbackInfo& info);
    Napi::Value set_generation_config(const Napi::CallbackInfo& info);

private:
    std::shared_ptr<ov::genai::Text2ImagePipeline> pipe = nullptr;
    std::shared_ptr<PipelineState> state = std::make_shared<PipelineState>();
};
