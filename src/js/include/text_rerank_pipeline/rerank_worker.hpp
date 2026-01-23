// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/genai/rag/text_rerank_pipeline.hpp"

class RerankWorker : public Napi::AsyncWorker {
public:
    RerankWorker(Napi::Function& callback,
                 std::shared_ptr<ov::genai::TextRerankPipeline> pipe,
                 std::shared_ptr<bool> is_reranking,
                 std::string&& query,
                 std::vector<std::string>&& documents);
    virtual ~RerankWorker() {}
    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error& e) override;

private:
    std::shared_ptr<ov::genai::TextRerankPipeline> pipe;
    std::shared_ptr<bool> is_reranking;
    std::string query;
    std::vector<std::string> documents;
    std::vector<std::pair<size_t, float>> rerank_result;
};
