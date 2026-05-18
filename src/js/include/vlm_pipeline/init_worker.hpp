// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <filesystem>
#include <napi.h>

#include "openvino/genai/visual_language/pipeline.hpp"

using namespace Napi;

class VLMInitWorker : public AsyncWorker {
public:
    VLMInitWorker(Function& callback,
                  std::shared_ptr<ov::genai::VLMPipeline>& pipe,
                  std::shared_ptr<std::atomic<bool>> is_initializing,
                  std::filesystem::path model_path,
                  std::string device,
                  ov::AnyMap properties);
    virtual ~VLMInitWorker() {}

    void Execute() override;
    void OnOK() override;
    void OnError(const Error& e) override;

private:
    std::shared_ptr<ov::genai::VLMPipeline>& pipe;
    std::shared_ptr<std::atomic<bool>> is_initializing;
    std::filesystem::path model_path;
    std::string device;
    ov::AnyMap properties;
};
