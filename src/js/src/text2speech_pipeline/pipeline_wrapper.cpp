// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/text2speech_pipeline/pipeline_wrapper.hpp"

#include "include/base/inference_thread.hpp"

Text2SpeechPipelineWrapper::Text2SpeechPipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<Text2SpeechPipelineWrapper>(info) {}

Napi::Function Text2SpeechPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "Text2SpeechPipeline",
                       {InstanceMethod("init", &Text2SpeechPipelineWrapper::init),
                        InstanceMethod("generate", &Text2SpeechPipelineWrapper::generate),
                        InstanceMethod("getGenerationConfig", &Text2SpeechPipelineWrapper::get_generation_config),
                        InstanceMethod("setGenerationConfig", &Text2SpeechPipelineWrapper::set_generation_config)});
}

Napi::Value Text2SpeechPipelineWrapper::init(const Napi::CallbackInfo& info) {
    return initialize_pipeline(info, pipe, state);
}

Napi::Value Text2SpeechPipelineWrapper::generate(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        const auto env = info.Env();
        OPENVINO_ASSERT(pipe, "Text2SpeechPipeline is not initialized");
        auto operation = std::make_shared<PipelineOperationGuard>(
            state, PipelineOperation::GENERATE, "Another generate is already in progress");
        VALIDATE_ARGS_COUNT(info, 4, "generate()");
        auto inputs = js_to_cpp<ov::genai::StringInputs>(env, info[0]);
        ov::Tensor speaker_embedding;
        set_if_defined(env, info[1], speaker_embedding);
        auto properties = js_to_cpp<ov::AnyMap>(env, info[2]);
        OPENVINO_ASSERT(info[3].IsFunction(), "generate callback is not a function");
        auto context = std::make_unique<InferenceThreadContext>(std::move(operation), "text2speechPerformInferenceThread");
        context->run_generate = [pipe = pipe,
                                 inputs = std::move(inputs),
                                 speaker_embedding = std::move(speaker_embedding),
                                 properties = std::move(properties)]() mutable -> JsResultProducer {
            ov::genai::Text2SpeechDecodedResults result;
            std::visit(overloaded{[&](const std::string& text) {
                                      result = pipe->generate(text, speaker_embedding, properties);
                                  },
                                  [&](const std::vector<std::string>& texts) {
                                      result = pipe->generate(texts, speaker_embedding, properties);
                                  }},
                       inputs);
            return [result = std::move(result)](Napi::Env env) -> Napi::Value {
                return to_text2speech_decoded_result(env, result);
            };
        };
        start_inference(info, info[3].As<Napi::Function>(), std::move(context));
        return env.Undefined();
    });
}

Napi::Value Text2SpeechPipelineWrapper::get_generation_config(const Napi::CallbackInfo& info) {
    return get_pipeline_generation_config(info, pipe, state, "Text2SpeechPipeline");
}

Napi::Value Text2SpeechPipelineWrapper::set_generation_config(const Napi::CallbackInfo& info) {
    return set_pipeline_generation_config(info, pipe, state, "Text2SpeechPipeline");
}
