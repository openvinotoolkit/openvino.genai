#pragma once

#include <napi.h>
#include "openvino/genai/llm_pipeline.hpp"

using namespace Napi;

class FinishChatWorker : public AsyncWorker {
 public:
  FinishChatWorker(Function& callback, std::shared_ptr<ov::genai::LLMPipeline>& pipe);
  virtual ~FinishChatWorker(){}

  void Execute() override;
  void OnOK() override;

 private:
  std::shared_ptr<ov::genai::LLMPipeline>& pipe;
};
