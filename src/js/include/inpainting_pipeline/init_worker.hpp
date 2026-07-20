// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <filesystem>

#include <napi.h>

#include "openvino/genai/image_generation/inpainting_pipeline.hpp"

class InpaintingInitWorker : public Napi::AsyncWorker {
public:
    InpaintingInitWorker(Napi::Function& callback,
                         std::shared_ptr<ov::genai::InpaintingPipeline>& pipe,
                         std::shared_ptr<std::atomic<bool>> is_initializing,
                         std::filesystem::path model_path,
                         std::string device,
                         ov::AnyMap properties);
    ~InpaintingInitWorker() override = default;

    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error& e) override;

private:
    std::shared_ptr<ov::genai::InpaintingPipeline>& pipe;
    std::shared_ptr<std::atomic<bool>> is_initializing;
    std::filesystem::path model_path;
    std::string device;
    ov::AnyMap properties;
};
