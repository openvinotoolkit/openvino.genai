// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/text_rerank_pipeline/rerank_worker.hpp"

#include "include/helper.hpp"

RerankWorker::RerankWorker(Napi::Function& callback,
                           std::shared_ptr<ov::genai::TextRerankPipeline> pipe,
                           std::shared_ptr<bool> is_reranking,
                           std::string&& query,
                           std::vector<std::string>&& documents)
    : Napi::AsyncWorker(callback),
      pipe(pipe),
      is_reranking(is_reranking),
      query(std::move(query)),
      documents(std::move(documents)) {}

void RerankWorker::Execute() {
    this->rerank_result = this->pipe->rerank(this->query, this->documents);
}

void RerankWorker::OnOK() {
    *this->is_reranking = false;
    Callback().Call(
        {Env().Null(), cpp_to_js<std::vector<std::pair<size_t, float>>, Napi::Value>(Env(), this->rerank_result)});
};

void RerankWorker::OnError(const Napi::Error& e) {
    *this->is_reranking = false;
    Callback().Call({Napi::Error::New(Env(), e.Message()).Value()});
};
