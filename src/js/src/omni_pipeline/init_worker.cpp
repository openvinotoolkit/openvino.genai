// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/omni_pipeline/init_worker.hpp"

OmniInitWorker::OmniInitWorker(Function& callback,
                               std::shared_ptr<ov::genai::OmniPipeline>& pipe,
                               std::shared_ptr<std::atomic<bool>> is_initializing,
                               std::filesystem::path model_path,
                               std::string device,
                               ov::AnyMap properties)
    : AsyncWorker(callback),
      pipe(pipe),
      is_initializing(is_initializing),
      model_path(std::move(model_path)),
      device(std::move(device)),
      properties(std::move(properties)) {};

void OmniInitWorker::Execute() {
    *this->is_initializing = true;
    this->pipe = std::make_shared<ov::genai::OmniPipeline>(this->model_path, this->device, this->properties);
};

void OmniInitWorker::OnOK() {
    *this->is_initializing = false;
    Callback().Call({Env().Null()});
};

void OmniInitWorker::OnError(const Error& e) {
    *this->is_initializing = false;
    Callback().Call({Napi::Error::New(Env(), e.Message()).Value()});
};
