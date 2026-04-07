// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/genai/video_generation/text2video_pipeline.hpp"

using namespace Napi;

class Text2VideoInitWorker : public AsyncWorker {
public:
    Text2VideoInitWorker(Function& callback,
                         std::shared_ptr<ov::genai::Text2VideoPipeline>& pipe,
                         std::shared_ptr<bool> is_initializing,
                         const std::string model_path,
                         std::string device,
                         ov::AnyMap properties);
    virtual ~Text2VideoInitWorker() {}

    void Execute() override;
    void OnOK() override;
    void OnError(const Error& e) override;

private:
    std::shared_ptr<ov::genai::Text2VideoPipeline>& pipe;
    std::shared_ptr<bool> is_initializing;
    std::string model_path;
    std::string device;
    ov::AnyMap properties;
};
