#include "include/llm_pipeline/start_chat_worker.hpp"
#include <chrono>
#include <thread>

StartChatWorker::StartChatWorker(Function& callback, std::shared_ptr<ov::genai::LLMPipeline>& pipe, std::string system_message)
    : AsyncWorker(callback), pipe(pipe), system_message(system_message) {};

void StartChatWorker::Execute() {
  this->pipe->start_chat(this->system_message);
};

void StartChatWorker::OnOK() {
  Callback().Call({ Env().Null() });
};
