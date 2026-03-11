// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/genai/rag/text_rerank_pipeline.hpp"

class RerankInitWorker : public Napi::AsyncWorker {
public:
    RerankInitWorker(Napi::Function& callback,
                     std::shared_ptr<ov::genai::TextRerankPipeline>& pipe,
                     std::shared_ptr<bool> is_initializing,
                     std::string&& model_path,
                     std::string&& device,
                     ov::AnyMap&& config,
                     ov::AnyMap&& properties);
    virtual ~RerankInitWorker() {}
    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error& e) override;

private:
    std::shared_ptr<ov::genai::TextRerankPipeline>& pipe;
    std::shared_ptr<bool> is_initializing;
    std::string model_path;
    std::string device;
    ov::AnyMap config;
    ov::AnyMap properties;
};
