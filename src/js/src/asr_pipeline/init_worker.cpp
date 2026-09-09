// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/asr_pipeline/init_worker.hpp"

#include "include/helper.hpp"

ASRInitWorker::ASRInitWorker(Napi::Function& callback,
                             std::shared_ptr<ov::genai::ASRPipeline>& pipe,
                             std::shared_ptr<std::atomic<bool>> is_initializing,
                             std::filesystem::path model_path,
                             std::string device,
                             ov::AnyMap properties)
    : Napi::AsyncWorker(callback),
      pipe(pipe),
      is_initializing(is_initializing),
      model_path(std::move(model_path)),
      device(std::move(device)),
      properties(std::move(properties)) {}

void ASRInitWorker::Execute() {
    this->pipe = std::make_shared<ov::genai::ASRPipeline>(this->model_path, this->device, this->properties);
}

void ASRInitWorker::OnOK() {
    *this->is_initializing = false;
    Callback().Call({Env().Null()});
}

void ASRInitWorker::OnError(const Napi::Error& e) {
    *this->is_initializing = false;
    Callback().Call({Napi::Error::New(Env(), e.Message()).Value()});
}
