#pragma once

#include <napi.h>
#include "openvino/genai/llm_pipeline.hpp"

using namespace Napi;

class StartChatWorker : public AsyncWorker {
 public:
  StartChatWorker(Function& callback, std::shared_ptr<ov::genai::LLMPipeline>& pipe);
  virtual ~StartChatWorker(){};

  void Execute();
  void OnOK();

 private:
  std::shared_ptr<ov::genai::LLMPipeline>& pipe;
};