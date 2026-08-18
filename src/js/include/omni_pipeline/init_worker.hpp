// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <filesystem>
#include <napi.h>

#include "openvino/genai/omni/pipeline.hpp"

using namespace Napi;

class OmniInitWorker : public AsyncWorker {
public:
    OmniInitWorker(Function& callback,
                   std::shared_ptr<ov::genai::OmniPipeline>& pipe,
                   std::shared_ptr<std::atomic<bool>> is_initializing,
                   std::filesystem::path model_path,
                   std::string device,
                   ov::AnyMap properties);
    virtual ~OmniInitWorker() {}

    void Execute() override;
    void OnOK() override;
    void OnError(const Error& e) override;

private:
    std::shared_ptr<ov::genai::OmniPipeline>& pipe;
    std::shared_ptr<std::atomic<bool>> is_initializing;
    std::filesystem::path model_path;
    std::string device;
    ov::AnyMap properties;
};
