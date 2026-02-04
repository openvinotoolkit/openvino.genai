#include "include/llm_pipeline/init_worker.hpp"

InitWorker::InitWorker(Function& callback,
                       std::shared_ptr<ov::genai::LLMPipeline>& pipe,
                       std::shared_ptr<bool> is_initializing,
                       const std::string model_path,
                       const std::string device,
                       const ov::AnyMap properties)
    : AsyncWorker(callback),
      pipe(pipe),
      is_initializing(is_initializing),
      model_path(model_path),
      device(device),
      properties(properties) {};

void InitWorker::Execute() {
    *this->is_initializing = true;
    this->pipe = std::make_shared<ov::genai::LLMPipeline>(this->model_path, this->device, this->properties);
};

void InitWorker::OnOK() {
    *this->is_initializing = false;
    Callback().Call({Env().Null()});
};

void InitWorker::OnError(const Error& e) {
    *this->is_initializing = false;
    Callback().Call({Napi::Error::New(Env(), e.Message()).Value()});
};
