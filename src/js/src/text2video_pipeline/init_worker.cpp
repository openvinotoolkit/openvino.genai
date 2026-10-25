// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/text2video_pipeline/init_worker.hpp"

#include <filesystem>

Text2VideoInitWorker::Text2VideoInitWorker(Napi::Function& callback,
                                           std::shared_ptr<ov::genai::Text2VideoPipeline>& pipe,
                                           std::shared_ptr<std::atomic<bool>> is_initializing,
                                           std::filesystem::path model_path,
                                           std::string device,
                                           ov::AnyMap properties)
    : Napi::AsyncWorker(callback),
      pipe(pipe),
      is_initializing(std::move(is_initializing)),
      model_path(std::move(model_path)),
      device(std::move(device)),
      properties(std::move(properties)) {}

void Text2VideoInitWorker::Execute() {
    this->pipe = std::make_shared<ov::genai::Text2VideoPipeline>(this->model_path, this->device, this->properties);
}

void Text2VideoInitWorker::OnOK() {
    this->is_initializing->store(false);
    Callback().Call({Env().Null()});
}

void Text2VideoInitWorker::OnError(const Napi::Error& e) {
    this->is_initializing->store(false);
    Callback().Call({Napi::Error::New(Env(), e.Message()).Value()});
}
