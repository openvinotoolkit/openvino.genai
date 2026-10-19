// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/llm_pipeline/llm_pipeline_wrapper.hpp"

#include "include/addon.hpp"
#include "include/base/inference_thread.hpp"
#include "include/helper.hpp"
#include "include/llm_pipeline/finish_chat_worker.hpp"
#include "include/llm_pipeline/init_worker.hpp"
#include "include/llm_pipeline/start_chat_worker.hpp"
#include "include/perf_metrics.hpp"
#include "include/tokenizer.hpp"

LLMPipelineWrapper::LLMPipelineWrapper(const Napi::CallbackInfo& info) : Napi::ObjectWrap<LLMPipelineWrapper>(info) {};

Napi::Function LLMPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "LLMPipeline",
                       {InstanceMethod("init", &LLMPipelineWrapper::init),
                        InstanceMethod("generate", &LLMPipelineWrapper::generate),
                        InstanceMethod("getTokenizer", &LLMPipelineWrapper::get_tokenizer),
                        InstanceMethod("getGenerationConfig", &LLMPipelineWrapper::get_generation_config),
                        InstanceMethod("setGenerationConfig", &LLMPipelineWrapper::set_generation_config),
                        InstanceMethod("startChat", &LLMPipelineWrapper::start_chat),
                        InstanceMethod("finishChat", &LLMPipelineWrapper::finish_chat)});
}

Napi::Value LLMPipelineWrapper::init(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(!this->pipe, "Pipeline is already initialized");
        OPENVINO_ASSERT(!*this->is_initializing, "Pipeline is already initializing");
        VALIDATE_ARGS_COUNT(info, 4, "init()");
        auto model_path = js_to_cpp<std::filesystem::path>(env, info[0]);
        auto device = js_to_cpp<std::string>(env, info[1]);
        auto properties = js_to_cpp<ov::AnyMap>(env, info[2]);
        OPENVINO_ASSERT(info[3].IsFunction(), "init callback is not a function");
        auto callback = info[3].As<Napi::Function>();

        auto* asyncWorker = new InitWorker(callback,
                                           this->pipe,
                                           this->is_initializing,
                                           std::move(model_path),
                                           std::move(device),
                                           std::move(properties));
        asyncWorker->Queue();
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value LLMPipelineWrapper::generate(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "LLMPipeline is not initialized");
        OPENVINO_ASSERT(!*this->is_generating, "Another generation is already in progress");
        *this->is_generating = true;
        VALIDATE_ARGS_COUNT(info, 4, "generate()");
        auto inputs = js_to_cpp<GenerateInputs>(env, info[0]);
        auto generation_config = js_to_cpp<ov::AnyMap>(env, info[1]);
        OPENVINO_ASSERT(info[2].IsFunction() || info[2].IsUndefined(), "streamer callback is not a function");
        auto streamer_arg = info[2];
        OPENVINO_ASSERT(info[3].IsFunction(), "generate callback is not a function");
        auto callback = info[3].As<Napi::Function>();

        auto* context =
            new InferenceThreadContext(this->is_generating, "performInferenceThread", "Streamer exceptions occurred:");
        auto pipe = this->pipe;
        context->run_generate = [context, pipe, inputs = std::move(inputs),
                                 generation_config = std::move(generation_config)]() mutable -> JsResultProducer {
            ov::genai::GenerationConfig config;
            config.update_generation_config(generation_config);

            ov::genai::StreamerVariant streamer = std::monostate();
            if (context->streamer_tsfn.has_value()) {
                streamer = make_text_streamer(context);
            }

            ov::genai::DecodedResults result;
            std::visit(overloaded{[&](ov::genai::StringInputs& prompt_inputs) {
                                      result = pipe->generate(prompt_inputs, config, streamer);
                                  },
                                  [&](ov::genai::ChatHistory& chat_inputs) {
                                      result = pipe->generate(chat_inputs, config, streamer);
                                  },
                                  [&](auto&) {
                                      OPENVINO_THROW("Unsupported type for generate inputs.");
                                  }},
                       inputs);

            return [result = std::move(result)](Napi::Env env) -> Napi::Value {
                return to_decoded_result(env, result);
            };
        };

        context->callback_tsfn = Napi::ThreadSafeFunction::New(env,
                                                              callback,
                                                              "LLM_generate_callback",
                                                              0,
                                                              1,
                                                              [context](Napi::Env) {
                                                                  if (context->native_thread.joinable()) {
                                                                      context->native_thread.join();
                                                                  }
                                                                  delete context;
                                                              });
        if (streamer_arg.IsFunction()) {
            context->streamer_tsfn =
                Napi::ThreadSafeFunction::New(env, streamer_arg.As<Napi::Function>(), "LLM_generate_streamer", 0, 1);
        }
        context->native_thread = std::thread(perform_generate_thread, context);
    } catch (const std::exception& ex) {
        *this->is_generating = false;
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value LLMPipelineWrapper::start_chat(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "LLMPipeline is not initialized");
        VALIDATE_ARGS_COUNT(info, 2, "startChat()");
        auto system_message = js_to_cpp<std::string>(env, info[0]);
        OPENVINO_ASSERT(info[1].IsFunction(), "startChat callback is not a function");
        auto callback = info[1].As<Napi::Function>();

        auto* asyncWorker = new StartChatWorker(callback, this->pipe, system_message);
        asyncWorker->Queue();
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value LLMPipelineWrapper::finish_chat(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "LLMPipeline is not initialized");
        VALIDATE_ARGS_COUNT(info, 1, "finishChat()");
        OPENVINO_ASSERT(info[0].IsFunction(), "finishChat callback is not a function");
        auto callback = info[0].As<Napi::Function>();

        FinishChatWorker* asyncWorker = new FinishChatWorker(callback, this->pipe);
        asyncWorker->Queue();
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value LLMPipelineWrapper::get_tokenizer(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "LLMPipeline is not initialized");
        auto tokenizer = this->pipe->get_tokenizer();
        return TokenizerWrapper::wrap(env, tokenizer);
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value LLMPipelineWrapper::get_generation_config(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "LLMPipeline is not initialized");
        return cpp_to_js<ov::genai::GenerationConfig, Napi::Value>(env, this->pipe->get_generation_config());
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value LLMPipelineWrapper::set_generation_config(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "LLMPipeline is not initialized");
        VALIDATE_ARGS_COUNT(info, 1, "setGenerationConfig()");
        this->pipe->set_generation_config(js_to_cpp<ov::genai::GenerationConfig>(env, info[0]));
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}
