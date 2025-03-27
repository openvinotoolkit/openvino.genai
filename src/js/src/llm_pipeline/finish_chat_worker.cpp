#include "include/llm_pipeline/finish_chat_worker.hpp"
#include <chrono>
#include <thread>

FinishChatWorker::FinishChatWorker(Function& callback, std::shared_ptr<ov::genai::LLMPipeline>& pipe)
    : AsyncWorker(callback), pipe(pipe) {};

void FinishChatWorker::Execute() {
  this->pipe->finish_chat();
};

void FinishChatWorker::OnOK() {
  Callback().Call({ Env().Null() });
};
