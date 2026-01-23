// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/vlm_pipeline/finish_chat_worker.hpp"

VLMFinishChatWorker::VLMFinishChatWorker(Function& callback, std::shared_ptr<ov::genai::VLMPipeline>& pipe)
    : AsyncWorker(callback),
      pipe(pipe) {};

void VLMFinishChatWorker::Execute() {
    this->pipe->finish_chat();
};

void VLMFinishChatWorker::OnOK() {
    Callback().Call({Env().Null()});
};
