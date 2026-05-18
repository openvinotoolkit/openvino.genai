// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/text2speech_pipeline/init_worker.hpp"

#include <atomic>
#include <filesystem>

#include "include/helper.hpp"

Text2SpeechInitWorker::Text2SpeechInitWorker(Napi::Function& callback,
                                             std::shared_ptr<ov::genai::Text2SpeechPipeline>& pipe,
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

void Text2SpeechInitWorker::Execute() {
    this->pipe = std::make_shared<ov::genai::Text2SpeechPipeline>(std::filesystem::path(this->model_path),
                                                                  this->device,
                                                                  this->properties);
}

void Text2SpeechInitWorker::OnOK() {
    this->is_initializing->store(false);
    Callback().Call({Env().Null()});
}

void Text2SpeechInitWorker::OnError(const Napi::Error& e) {
    this->is_initializing->store(false);
    Callback().Call({Napi::Error::New(Env(), e.Message()).Value()});
}
