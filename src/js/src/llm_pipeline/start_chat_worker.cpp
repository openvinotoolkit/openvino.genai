#include "include/llm_pipeline/start_chat_worker.hpp"
#include <chrono>
#include <thread>

StartChatWorker::StartChatWorker(Function& callback, std::shared_ptr<ov::genai::LLMPipeline>& pipe)
    : AsyncWorker(callback), pipe(pipe) {};

void StartChatWorker::Execute() {
  this->pipe->start_chat();
};

void StartChatWorker::OnOK() {
  Callback().Call({ Env().Null() });
};
