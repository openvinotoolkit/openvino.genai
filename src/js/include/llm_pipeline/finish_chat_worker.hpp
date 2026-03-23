// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/genai/llm_pipeline.hpp"

using namespace Napi;

class FinishChatWorker : public AsyncWorker {
public:
    FinishChatWorker(Function& callback, std::shared_ptr<ov::genai::LLMPipeline>& pipe);
    virtual ~FinishChatWorker() {}

    void Execute() override;
    void OnOK() override;

private:
    std::shared_ptr<ov::genai::LLMPipeline>& pipe;
};
