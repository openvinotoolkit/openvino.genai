// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include <thread>

#include "openvino/genai/visual_language/pipeline.hpp"

class VLMPipelineWrapper : public Napi::ObjectWrap<VLMPipelineWrapper> {
public:
    VLMPipelineWrapper(const Napi::CallbackInfo& info);

    static Napi::Function get_class(Napi::Env env);

    Napi::Value init(const Napi::CallbackInfo& info);
    Napi::Value generate(const Napi::CallbackInfo& info);
    Napi::Value start_chat(const Napi::CallbackInfo& info);
    Napi::Value finish_chat(const Napi::CallbackInfo& info);
    Napi::Value get_tokenizer(const Napi::CallbackInfo& info);
    Napi::Value set_chat_template(const Napi::CallbackInfo& info);
    Napi::Value get_generation_config(const Napi::CallbackInfo& info);
    Napi::Value set_generation_config(const Napi::CallbackInfo& info);

private:
    std::shared_ptr<ov::genai::VLMPipeline> pipe = nullptr;
    std::shared_ptr<bool> is_initializing = std::make_shared<bool>(false);
    std::shared_ptr<bool> is_generating = std::make_shared<bool>(false);
};
