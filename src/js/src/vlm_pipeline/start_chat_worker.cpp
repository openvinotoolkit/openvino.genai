// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/vlm_pipeline/start_chat_worker.hpp"

VLMStartChatWorker::VLMStartChatWorker(Function& callback,
                                       std::shared_ptr<ov::genai::VLMPipeline>& pipe,
                                       std::string system_message)
    : AsyncWorker(callback),
      pipe(pipe),
      system_message(system_message) {};

void VLMStartChatWorker::Execute() {
    this->pipe->start_chat(this->system_message);
};

void VLMStartChatWorker::OnOK() {
    Callback().Call({Env().Null()});
};
