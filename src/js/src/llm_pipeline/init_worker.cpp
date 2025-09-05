#include "include/llm_pipeline/init_worker.hpp"
#include <chrono>
#include <thread>

InitWorker::InitWorker(
  Function& callback,
  std::shared_ptr<ov::genai::LLMPipeline>& pipe,
  const std::string model_path,
  const std::string device,
  const ov::AnyMap properties
) : AsyncWorker(callback), pipe(pipe), model_path(model_path), device(device), properties(properties) {};

void InitWorker::Execute() {
  this->pipe = std::make_shared<ov::genai::LLMPipeline>(this->model_path, this->device, this->properties);
};

void InitWorker::OnOK() {
  Callback().Call({ Env().Null() });
};
