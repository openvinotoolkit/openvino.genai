// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include <thread>

#include "openvino/genai/image_generation/text2image_pipeline.hpp"

class Text2VideoPipelineWrapper : public Napi::ObjectWrap<Text2VideoPipelineWrapper> {
public:
    Text2VideoPipelineWrapper(const Napi::CallbackInfo& info);

    static Napi::Function get_class(Napi::Env env);

    Napi::Value init(const Napi::CallbackInfo& info);
    Napi::Value generate(const Napi::CallbackInfo& info);

private:
    // Using Text2ImagePipeline as placeholder until Text2VideoPipeline C++ API exists
    std::shared_ptr<ov::genai::Text2ImagePipeline> pipe = nullptr;
    std::shared_ptr<bool> is_initializing = std::make_shared<bool>(false);
    std::shared_ptr<bool> is_generating = std::make_shared<bool>(false);
};
