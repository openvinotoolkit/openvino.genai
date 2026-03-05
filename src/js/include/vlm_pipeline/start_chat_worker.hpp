// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/genai/visual_language/pipeline.hpp"

using namespace Napi;

class VLMStartChatWorker : public AsyncWorker {
public:
    VLMStartChatWorker(Function& callback, std::shared_ptr<ov::genai::VLMPipeline>& pipe, std::string system_message);
    virtual ~VLMStartChatWorker() {}

    void Execute() override;
    void OnOK() override;

private:
    std::shared_ptr<ov::genai::VLMPipeline>& pipe;
    std::string system_message;
};
