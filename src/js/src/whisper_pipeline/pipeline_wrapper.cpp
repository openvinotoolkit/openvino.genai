// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/whisper_pipeline/pipeline_wrapper.hpp"

#include <thread>

#include "include/base/inference_thread.hpp"
#include "include/helper.hpp"
#include "include/tokenizer.hpp"
#include "include/whisper_pipeline/init_worker.hpp"

WhisperPipelineWrapper::WhisperPipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<WhisperPipelineWrapper>(info) {}

Napi::Function WhisperPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "WhisperPipeline",
                       {
                           InstanceMethod("init", &WhisperPipelineWrapper::init),
                           InstanceMethod("generate", &WhisperPipelineWrapper::generate),
                           InstanceMethod("getTokenizer", &WhisperPipelineWrapper::get_tokenizer),
                           InstanceMethod("getGenerationConfig", &WhisperPipelineWrapper::get_generation_config),
                           InstanceMethod("setGenerationConfig", &WhisperPipelineWrapper::set_generation_config),
                       });
}

Napi::Value WhisperPipelineWrapper::init(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(!this->pipe, "Pipeline is already initialized");
        OPENVINO_ASSERT(!*this->is_initializing, "Pipeline is already initializing");
        *this->is_initializing = true;

        VALIDATE_ARGS_COUNT(info, 4, "init()");
        auto model_path = js_to_cpp<std::filesystem::path>(env, info[0]);
        auto device = js_to_cpp<std::string>(env, info[1]);
        auto properties = js_to_cpp<ov::AnyMap>(env, info[2]);
        OPENVINO_ASSERT(info[3].IsFunction(), "init callback is not a function");
        Napi::Function callback = info[3].As<Napi::Function>();

        auto async_worker = new WhisperInitWorker(callback,
                                                  this->pipe,
                                                  this->is_initializing,
                                                  std::move(model_path),
                                                  std::move(device),
                                                  std::move(properties));
        async_worker->Queue();
    } catch (const std::exception& ex) {
        *this->is_initializing = false;
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }

    return env.Undefined();
}

Napi::Value WhisperPipelineWrapper::generate(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "WhisperPipeline is not initialized");
        OPENVINO_ASSERT(!*this->is_generating, "Another generate is already in progress");
        *this->is_generating = true;

        VALIDATE_ARGS_COUNT(info, 4, "generate()");
        auto raw_speech = js_to_cpp<std::vector<float>>(env, info[0]);
        auto generation_config = js_to_cpp<ov::AnyMap>(env, info[1]);
        OPENVINO_ASSERT(info[3].IsFunction(), "generate callback is not a function");
        Napi::Function callback = info[3].As<Napi::Function>();

        auto* context = new InferenceThreadContext(this->is_generating, "whisperPerformInferenceThread");
        auto pipe = this->pipe;
        context->run_generate = [context, pipe, raw_speech = std::move(raw_speech),
                                 generation_config = std::move(generation_config)]() mutable -> JsResultProducer {
            ov::genai::WhisperDecodedResults result;
            if (context->streamer_tsfn.has_value()) {
                ov::genai::StreamerVariant streamer = make_text_streamer(context);
                auto config = pipe->get_generation_config();
                config.update_generation_config(generation_config);
                result = pipe->generate(raw_speech, config, streamer);
            } else {
                result = pipe->generate(raw_speech, generation_config);
            }
            return [result = std::move(result)](Napi::Env env) -> Napi::Value {
                return to_whisper_decoded_result(env, result);
            };
        };

        context->callback_tsfn =
            Napi::ThreadSafeFunction::New(env, callback, "Whisper_generate_callback", 0, 1, [context](Napi::Env) {
                if (context->native_thread.joinable()) {
                    context->native_thread.join();
                }
                delete context;
            });

        if (info[2].IsFunction()) {
            context->streamer_tsfn =
                Napi::ThreadSafeFunction::New(env, info[2].As<Napi::Function>(), "Whisper_generate_streamer", 0, 1);
        }

        context->native_thread = std::thread(perform_generate_thread, context);
    } catch (const std::exception& ex) {
        *this->is_generating = false;
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }

    return env.Undefined();
}

Napi::Value WhisperPipelineWrapper::get_tokenizer(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "WhisperPipeline is not initialized");
        auto tokenizer = this->pipe->get_tokenizer();
        return TokenizerWrapper::wrap(env, tokenizer);
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}

Napi::Value WhisperPipelineWrapper::get_generation_config(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "WhisperPipeline is not initialized");
        auto config = this->pipe->get_generation_config();
        return cpp_to_js<ov::genai::WhisperGenerationConfig, Napi::Value>(env, config);
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}

Napi::Value WhisperPipelineWrapper::set_generation_config(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "WhisperPipeline is not initialized");
        VALIDATE_ARGS_COUNT(info, 1, "setGenerationConfig()");
        if (info[0].IsUndefined() || info[0].IsNull()) {
            return env.Undefined();
        }
        this->pipe->set_generation_config(js_to_cpp<ov::genai::WhisperGenerationConfig>(env, info[0]));
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}
