// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/text2video_pipeline/init_worker.hpp"

Text2VideoInitWorker::Text2VideoInitWorker(Function& callback,
                                           std::shared_ptr<ov::genai::Text2VideoPipeline>& pipe,
                                           std::shared_ptr<bool> is_initializing,
                                           const std::string model_path,
                                           const std::string device,
                                           const ov::AnyMap properties)
    : AsyncWorker(callback),
      pipe(pipe),
      is_initializing(is_initializing),
      model_path(model_path),
      device(device),
      properties(properties) {};

void Text2VideoInitWorker::Execute() {
    *this->is_initializing = true;
    this->pipe = std::make_shared<ov::genai::Text2VideoPipeline>(this->model_path, this->device, this->properties);
};

void Text2VideoInitWorker::OnOK() {
    *this->is_initializing = false;
    Callback().Call({Env().Null()});
};

void Text2VideoInitWorker::OnError(const Error& e) {
    *this->is_initializing = false;
    Callback().Call({Napi::Error::New(Env(), e.Message()).Value()});
};
