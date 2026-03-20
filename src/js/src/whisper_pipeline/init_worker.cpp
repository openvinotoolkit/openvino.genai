// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/whisper_pipeline/init_worker.hpp"

#include <filesystem>

#include "include/helper.hpp"

WhisperInitWorker::WhisperInitWorker(Napi::Function& callback,
                                     std::shared_ptr<ov::genai::WhisperPipeline>& pipe,
                                     std::shared_ptr<std::atomic<bool>> is_initializing,
                                     std::string&& model_path,
                                     std::string&& device,
                                     ov::AnyMap&& properties)
    : Napi::AsyncWorker(callback),
      pipe(pipe),
      is_initializing(is_initializing),
      model_path(std::move(model_path)),
      device(std::move(device)),
      properties(std::move(properties)) {}

void WhisperInitWorker::Execute() {
    this->pipe = std::make_shared<ov::genai::WhisperPipeline>(std::filesystem::path(this->model_path),
                                                              this->device,
                                                              this->properties);
}

void WhisperInitWorker::OnOK() {
    *this->is_initializing = false;
    Callback().Call({Env().Null()});
}

void WhisperInitWorker::OnError(const Napi::Error& e) {
    *this->is_initializing = false;
    Callback().Call({Napi::Error::New(Env(), e.Message()).Value()});
}
