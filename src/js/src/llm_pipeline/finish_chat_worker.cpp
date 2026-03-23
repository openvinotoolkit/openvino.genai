// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/llm_pipeline/finish_chat_worker.hpp"

#include <chrono>
#include <thread>

FinishChatWorker::FinishChatWorker(Function& callback, std::shared_ptr<ov::genai::LLMPipeline>& pipe)
    : AsyncWorker(callback),
      pipe(pipe) {};

void FinishChatWorker::Execute() {
    this->pipe->finish_chat();
};

void FinishChatWorker::OnOK() {
    Callback().Call({Env().Null()});
};
