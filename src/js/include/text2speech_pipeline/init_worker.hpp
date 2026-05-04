// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>

#include <napi.h>

#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

class Text2SpeechInitWorker : public Napi::AsyncWorker {
public:
    Text2SpeechInitWorker(Napi::Function& callback,
                          std::shared_ptr<ov::genai::Text2SpeechPipeline>& pipe,
                          std::shared_ptr<std::atomic<bool>> is_initializing,
                          std::string&& model_path,
                          std::string&& device,
                          ov::AnyMap&& properties);
    virtual ~Text2SpeechInitWorker() {}
    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error& e) override;

private:
    std::shared_ptr<ov::genai::Text2SpeechPipeline>& pipe;
    std::shared_ptr<std::atomic<bool>> is_initializing;
    std::string model_path;
    std::string device;
    ov::AnyMap properties;
};
