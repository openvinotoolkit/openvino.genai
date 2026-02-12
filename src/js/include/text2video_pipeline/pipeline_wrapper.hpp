// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include <thread>

#include "openvino/genai/video_generation/text2video_pipeline.hpp"

class Text2VideoPipelineWrapper : public Napi::ObjectWrap<Text2VideoPipelineWrapper> {
public:
    Text2VideoPipelineWrapper(const Napi::CallbackInfo& info);

    static Napi::Function get_class(Napi::Env env);

    Napi::Value init(const Napi::CallbackInfo& info);
    Napi::Value generate(const Napi::CallbackInfo& info);
    Napi::Value get_generation_config(const Napi::CallbackInfo& info);
    Napi::Value set_generation_config(const Napi::CallbackInfo& info);

private:
    std::shared_ptr<ov::genai::Text2VideoPipeline> pipe = nullptr;
    std::shared_ptr<bool> is_initializing = std::make_shared<bool>(false);
    std::shared_ptr<bool> is_generating = std::make_shared<bool>(false);
};
