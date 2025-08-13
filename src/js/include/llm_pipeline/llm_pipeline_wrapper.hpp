#pragma once

#include <thread>
#include <napi.h>
#include "openvino/genai/llm_pipeline.hpp"

class LLMPipelineWrapper : public Napi::ObjectWrap<LLMPipelineWrapper> {
public:
    LLMPipelineWrapper(const Napi::CallbackInfo& info);

    static Napi::Function get_class(Napi::Env env);

    Napi::Value init(const Napi::CallbackInfo& info);
    Napi::Value generate(const Napi::CallbackInfo& info);
    Napi::Value start_chat(const Napi::CallbackInfo& info);
    Napi::Value finish_chat(const Napi::CallbackInfo& info);
    Napi::Value get_tokenizer(const Napi::CallbackInfo& info);
private:
    bool is_loaded = false;
    bool is_initialized = false;
    bool is_running = false;

    std::string model_path;
    std::string device;

    std::shared_ptr<ov::genai::LLMPipeline> pipe = nullptr;
    std::function<bool(std::string)> streamer;
};
