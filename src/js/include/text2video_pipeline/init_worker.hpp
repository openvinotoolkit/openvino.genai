// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/genai/image_generation/text2image_pipeline.hpp"

using namespace Napi;

class Text2VideoInitWorker : public AsyncWorker {
public:
    Text2VideoInitWorker(Function& callback,
                         std::shared_ptr<ov::genai::Text2ImagePipeline>& pipe,
                         std::shared_ptr<bool> is_initializing,
                         const std::string model_path,
                         std::string device,
                         ov::AnyMap properties);
    virtual ~Text2VideoInitWorker() {}

    void Execute() override;
    void OnOK() override;
    void OnError(const Error& e) override;

private:
    std::shared_ptr<ov::genai::Text2ImagePipeline>& pipe;
    std::shared_ptr<bool> is_initializing;
    std::string model_path;
    std::string device;
    ov::AnyMap properties;
};
