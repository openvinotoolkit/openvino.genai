// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <filesystem>

#include <napi.h>

#include "openvino/genai/image_generation/image2image_pipeline.hpp"

class Image2ImageInitWorker : public Napi::AsyncWorker {
public:
    Image2ImageInitWorker(Napi::Function& callback,
                          std::shared_ptr<ov::genai::Image2ImagePipeline>& pipe,
                          std::shared_ptr<std::atomic<bool>> is_initializing,
                          std::filesystem::path model_path,
                          std::string device,
                          ov::AnyMap properties);
    ~Image2ImageInitWorker() override = default;

    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error& e) override;

private:
    std::shared_ptr<ov::genai::Image2ImagePipeline>& pipe;
    std::shared_ptr<std::atomic<bool>> is_initializing;
    std::filesystem::path model_path;
    std::string device;
    ov::AnyMap properties;
};
