// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/text_rerank_pipeline/init_worker.hpp"

#include "include/helper.hpp"

RerankInitWorker::RerankInitWorker(Napi::Function& callback,
                                   std::shared_ptr<ov::genai::TextRerankPipeline>& pipe,
                                   std::shared_ptr<bool> is_initializing,
                                   std::string&& model_path,
                                   std::string&& device,
                                   ov::AnyMap&& config,
                                   ov::AnyMap&& properties)
    : Napi::AsyncWorker(callback),
      pipe(pipe),
      is_initializing(is_initializing),
      model_path(std::move(model_path)),
      device(std::move(device)),
      config(std::move(config)),
      properties(std::move(properties)) {}

void RerankInitWorker::Execute() {
    ov::genai::TextRerankPipeline::Config config(this->config);
    this->pipe =
        std::make_shared<ov::genai::TextRerankPipeline>(this->model_path, this->device, config, this->properties);
}

void RerankInitWorker::OnOK() {
    *this->is_initializing = false;
    Callback().Call({Env().Null()});
};

void RerankInitWorker::OnError(const Napi::Error& e) {
    *this->is_initializing = false;
    Callback().Call({Napi::Error::New(Env(), e.Message()).Value()});
};
