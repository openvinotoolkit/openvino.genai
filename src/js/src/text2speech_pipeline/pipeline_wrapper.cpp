// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/text2speech_pipeline/pipeline_wrapper.hpp"

#include <iostream>
#include <memory>
#include <thread>

#include "include/helper.hpp"
#include "include/text2speech_pipeline/init_worker.hpp"

struct Text2SpeechTsfnContext {
    Text2SpeechTsfnContext(ov::genai::StringInputs inputs,
                           ov::Tensor speaker_embedding,
                           ov::AnyMap properties,
                           std::shared_ptr<std::atomic<bool>> is_generating)
        : inputs(std::move(inputs)),
          speaker_embedding(std::move(speaker_embedding)),
          properties(std::move(properties)),
          is_generating(is_generating) {}
    ~Text2SpeechTsfnContext() = default;

    std::thread native_thread;
    Napi::ThreadSafeFunction callback_tsfn;

    ov::genai::StringInputs inputs;
    ov::Tensor speaker_embedding;
    ov::AnyMap properties;
    std::shared_ptr<std::atomic<bool>> is_generating;
    std::shared_ptr<ov::genai::Text2SpeechPipeline> pipe = nullptr;
};

void text2speechPerformInferenceThread(Text2SpeechTsfnContext* context) {
    auto report_error = [context](const std::string& message) {
        auto status = context->callback_tsfn.BlockingCall([message](Napi::Env env, Napi::Function js_callback) {
            try {
                js_callback.Call(
                    {Napi::Error::New(env, "text2speechPerformInferenceThread error. " + message).Value(), env.Null()});
            } catch (const std::exception& err) {
                std::cerr
                    << "The callback failed when attempting to return an error from text2speechPerformInferenceThread. "
                       "Details:\n"
                    << err.what() << std::endl;
                std::cerr << "Original error message:\n" << message << std::endl;
            }
        });
        if (status != napi_ok) {
            std::cerr << "The BlockingCall failed with status " << status
                      << " when trying to return an error from text2speechPerformInferenceThread." << std::endl;
            std::cerr << "Original error message:\n" << message << std::endl;
        }
    };
    auto finalize = [context]() {
        context->callback_tsfn.Release();
    };

    ov::genai::Text2SpeechDecodedResults result;

    try {
        const auto& speaker = context->speaker_embedding;
        const auto& props = context->properties;
        std::visit(overloaded{[context, &result, &speaker, &props](const std::string& text) {
                                  result = context->pipe->generate(text, speaker, props);
                              },
                              [context, &result, &speaker, &props](const std::vector<std::string>& texts) {
                                  result = context->pipe->generate(texts, speaker, props);
                              }},
                   context->inputs);
    } catch (const std::exception& e) {
        context->is_generating->store(false);
        report_error(e.what());
        finalize();
        return;
    }

    context->is_generating->store(false);

    try {
        std::shared_ptr<ov::genai::Text2SpeechDecodedResults> final_result =
            std::make_shared<ov::genai::Text2SpeechDecodedResults>(std::move(result));
        std::shared_ptr<std::string> final_callback_error = std::make_shared<std::string>();

        napi_status status = context->callback_tsfn.BlockingCall(
            [final_result, final_callback_error](Napi::Env env, Napi::Function js_callback) {
                try {
                    js_callback.Call({env.Null(), to_text2speech_decoded_result(env, *final_result)});
                } catch (const std::exception& err) {
                    *final_callback_error = "The final callback failed. Details:\n" + std::string(err.what());
                }
            });
        if (status != napi_ok) {
            report_error("The final BlockingCall failed with status " + std::to_string(static_cast<int>(status)));
        } else if (!final_callback_error->empty()) {
            report_error(*final_callback_error);
        }
    } catch (const std::exception& e) {
        report_error(e.what());
    }
    finalize();
}

Text2SpeechPipelineWrapper::Text2SpeechPipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<Text2SpeechPipelineWrapper>(info) {}

Napi::Function Text2SpeechPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "Text2SpeechPipeline",
                       {
                           InstanceMethod("init", &Text2SpeechPipelineWrapper::init),
                           InstanceMethod("generate", &Text2SpeechPipelineWrapper::generate),
                           InstanceMethod("getGenerationConfig", &Text2SpeechPipelineWrapper::get_generation_config),
                           InstanceMethod("setGenerationConfig", &Text2SpeechPipelineWrapper::set_generation_config),
                       });
}

Napi::Value Text2SpeechPipelineWrapper::init(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(!this->pipe, "Pipeline is already initialized");
        OPENVINO_ASSERT(!this->is_initializing->load(), "Pipeline is already initializing");
        this->is_initializing->store(true);

        VALIDATE_ARGS_COUNT(info, 4, "init()");
        auto model_path = js_to_cpp<std::string>(env, info[0]);
        auto device = js_to_cpp<std::string>(env, info[1]);
        auto properties = js_to_cpp<ov::AnyMap>(env, info[2]);
        OPENVINO_ASSERT(info[3].IsFunction(), "init callback is not a function");
        Napi::Function callback = info[3].As<Napi::Function>();

        auto async_worker = new Text2SpeechInitWorker(callback,
                                                      this->pipe,
                                                      this->is_initializing,
                                                      std::move(model_path),
                                                      std::move(device),
                                                      std::move(properties));
        async_worker->Queue();
    } catch (const std::exception& ex) {
        this->is_initializing->store(false);
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }

    return env.Undefined();
}

Napi::Value Text2SpeechPipelineWrapper::generate(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "Text2SpeechPipeline is not initialized");
        OPENVINO_ASSERT(!this->is_generating->load(), "Another generate is already in progress");
        this->is_generating->store(true);

        VALIDATE_ARGS_COUNT(info, 4, "generate()");

        auto inputs = js_to_cpp<ov::genai::StringInputs>(env, info[0]);

        ov::Tensor speaker_embedding;

        if (!info[1].IsNull() && !info[1].IsUndefined()) {
            speaker_embedding = js_to_cpp<ov::Tensor>(env, info[1]);
        }

        auto properties = js_to_cpp<ov::AnyMap>(env, info[2]);

        OPENVINO_ASSERT(info[3].IsFunction(), "generate callback is not a function");
        Napi::Function callback = info[3].As<Napi::Function>();

        auto* context =
            new Text2SpeechTsfnContext(std::move(inputs), std::move(speaker_embedding), std::move(properties), this->is_generating);
        context->pipe = this->pipe;

        context->callback_tsfn =
            Napi::ThreadSafeFunction::New(env, callback, "Text2Speech_generate_callback", 0, 1, [context](Napi::Env) {
                context->native_thread.join();
                delete context;
            });

        context->native_thread = std::thread(text2speechPerformInferenceThread, context);
    } catch (const std::exception& ex) {
        this->is_generating->store(false);
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }

    return env.Undefined();
}

Napi::Value Text2SpeechPipelineWrapper::get_generation_config(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "Text2SpeechPipeline is not initialized");
        auto config = this->pipe->get_generation_config();
        return cpp_to_js<ov::genai::SpeechGenerationConfig, Napi::Value>(env, config);
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}

Napi::Value Text2SpeechPipelineWrapper::set_generation_config(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "Text2SpeechPipeline is not initialized");
        VALIDATE_ARGS_COUNT(info, 1, "setGenerationConfig()");
        if (info[0].IsUndefined() || info[0].IsNull()) {
            OPENVINO_THROW("Generation config cannot be undefined or null");
        }
        this->pipe->set_generation_config(js_to_cpp<ov::genai::SpeechGenerationConfig>(env, info[0]));
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}
