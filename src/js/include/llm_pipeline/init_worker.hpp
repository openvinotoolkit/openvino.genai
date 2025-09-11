#pragma once

#include <napi.h>
#include "openvino/genai/llm_pipeline.hpp"

using namespace Napi;

class InitWorker : public AsyncWorker {
public:
    InitWorker(Function& callback,
               std::shared_ptr<ov::genai::LLMPipeline>& pipe,
               const std::string model_path,
               std::string device,
               ov::AnyMap properties);
    virtual ~InitWorker() {}

    void Execute() override;
    void OnOK() override;

private:
    std::shared_ptr<ov::genai::LLMPipeline>& pipe;
    std::string model_path;
    std::string device;
    ov::AnyMap properties;
};
