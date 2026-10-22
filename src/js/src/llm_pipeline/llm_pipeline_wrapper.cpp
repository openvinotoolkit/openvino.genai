// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/llm_pipeline/llm_pipeline_wrapper.hpp"

#include "include/base/inference_thread.hpp"
#include "include/tokenizer.hpp"

LLMPipelineWrapper::LLMPipelineWrapper(const Napi::CallbackInfo& info) : Napi::ObjectWrap<LLMPipelineWrapper>(info) {}

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
    return initialize_pipeline(info, pipe, state);
}

Napi::Value LLMPipelineWrapper::generate(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        const auto env = info.Env();
        OPENVINO_ASSERT(pipe, "LLMPipeline is not initialized");
        auto operation = std::make_shared<PipelineOperationGuard>(
            state, PipelineOperation::GENERATE, "Another generation is already in progress");
        VALIDATE_ARGS_COUNT(info, 4, "generate()");
        auto inputs = js_to_cpp<GenerateInputs>(env, info[0]);
        auto generation_config = js_to_cpp<ov::AnyMap>(env, info[1]);
        OPENVINO_ASSERT(info[2].IsFunction() || info[2].IsUndefined(), "streamer callback is not a function");
        OPENVINO_ASSERT(info[3].IsFunction(), "generate callback is not a function");
        auto context = std::make_unique<InferenceThreadContext>(std::move(operation), "performInferenceThread");
        if (info[2].IsFunction()) {
            context->streamer = std::make_unique<ThreadSafeCallback>(env, info[2].As<Napi::Function>(), "LLM_streamer");
        }
        context->run_generate = [context = context.get(),
                                 pipe = pipe,
                                 inputs = std::move(inputs),
                                 properties = std::move(generation_config)]() mutable -> JsResultProducer {
            auto config = pipe->get_generation_config();
            config.update_generation_config(properties);
            ov::genai::StreamerVariant streamer = std::monostate();
            if (context->streamer) {
                streamer = make_text_streamer(context);
            }
            ov::genai::DecodedResults result;
            std::visit(overloaded{[&](ov::genai::StringInputs& prompts) {
                                      result = pipe->generate(prompts, config, streamer);
                                  },
                                  [&](const ov::genai::ChatHistory& history) {
                                      result = pipe->generate(history, config, streamer);
                                  }},
                       inputs);
            return [result = std::move(result)](Napi::Env env) -> Napi::Value {
                return to_decoded_result(env, result);
            };
        };
        start_inference(info, info[3].As<Napi::Function>(), std::move(context));
        return env.Undefined();
    });
}

Napi::Value LLMPipelineWrapper::start_chat(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        const auto env = info.Env();
        OPENVINO_ASSERT(pipe, "LLMPipeline is not initialized");
        auto operation =
            std::make_shared<PipelineOperationGuard>(state, PipelineOperation::ACCESS, "Pipeline is busy");
        VALIDATE_ARGS_COUNT(info, 2, "startChat()");
        auto system_message = js_to_cpp<std::string>(env, info[0]);
        OPENVINO_ASSERT(info[1].IsFunction(), "startChat callback is not a function");
        queue_pipeline_worker(info,
                              info[1].As<Napi::Function>(),
                              std::move(operation),
                              [pipe = pipe, system_message = std::move(system_message)] {
                                  pipe->start_chat(system_message);
                                  return [](Napi::Env env) -> Napi::Value {
                                      return env.Undefined();
                                  };
                              });
        return env.Undefined();
    });
}

Napi::Value LLMPipelineWrapper::finish_chat(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        OPENVINO_ASSERT(pipe, "LLMPipeline is not initialized");
        auto operation =
            std::make_shared<PipelineOperationGuard>(state, PipelineOperation::ACCESS, "Pipeline is busy");
        VALIDATE_ARGS_COUNT(info, 1, "finishChat()");
        OPENVINO_ASSERT(info[0].IsFunction(), "finishChat callback is not a function");
        queue_pipeline_worker(info, info[0].As<Napi::Function>(), std::move(operation), [pipe = pipe] {
            pipe->finish_chat();
            return [](Napi::Env env) -> Napi::Value {
                return env.Undefined();
            };
        });
        return info.Env().Undefined();
    });
}

Napi::Value LLMPipelineWrapper::get_tokenizer(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        OPENVINO_ASSERT(pipe, "LLMPipeline is not initialized");
        return TokenizerWrapper::wrap(info.Env(), pipe->get_tokenizer());
    });
}

Napi::Value LLMPipelineWrapper::get_generation_config(const Napi::CallbackInfo& info) {
    return get_pipeline_generation_config(info, pipe, state, "LLMPipeline");
}

Napi::Value LLMPipelineWrapper::set_generation_config(const Napi::CallbackInfo& info) {
    return set_pipeline_generation_config(info, pipe, state, "LLMPipeline");
}
