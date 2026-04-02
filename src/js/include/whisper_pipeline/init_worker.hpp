// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include <atomic>

#include "openvino/genai/whisper_pipeline.hpp"

class WhisperInitWorker : public Napi::AsyncWorker {
public:
    WhisperInitWorker(
        Napi::Function& callback,
        std::shared_ptr<ov::genai::WhisperPipeline>& pipe,
        std::shared_ptr<std::atomic<bool>> is_initializing,
        std::string&& model_path,
        std::string&& device,
        ov::AnyMap&& properties
    );
    virtual ~WhisperInitWorker() {}
    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error& e) override;

private:
    std::shared_ptr<ov::genai::WhisperPipeline>& pipe;
    std::shared_ptr<std::atomic<bool>> is_initializing;
    std::string model_path;
    std::string device;
    ov::AnyMap properties;
};
