// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/image2image_pipeline/init_worker.hpp"

#include <filesystem>

Image2ImageInitWorker::Image2ImageInitWorker(Napi::Function& callback,
                                             std::shared_ptr<ov::genai::Image2ImagePipeline>& pipe,
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

void Image2ImageInitWorker::Execute() {
    this->pipe = std::make_shared<ov::genai::Image2ImagePipeline>(this->model_path, this->device, this->properties);
}

void Image2ImageInitWorker::OnOK() {
    this->is_initializing->store(false);
    Callback().Call({Env().Null()});
}

void Image2ImageInitWorker::OnError(const Napi::Error& e) {
    this->is_initializing->store(false);
    Callback().Call({Napi::Error::New(Env(), e.Message()).Value()});
}
