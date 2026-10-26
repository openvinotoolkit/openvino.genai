// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/omni_pipeline/omni_pipeline_wrapper.hpp"

#include "include/base/inference_thread.hpp"

OmniPipelineWrapper::OmniPipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<OmniPipelineWrapper>(info) {}

Napi::Function OmniPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "OmniPipeline",
                       {InstanceMethod("init", &OmniPipelineWrapper::init),
                        InstanceMethod("generate", &OmniPipelineWrapper::generate)});
}

Napi::Value OmniPipelineWrapper::init(const Napi::CallbackInfo& info) {
    return initialize_pipeline(info, pipe, state);
}

Napi::Value OmniPipelineWrapper::generate(const Napi::CallbackInfo& info) {
    return with_pipeline_errors(info, [&]() -> Napi::Value {
        const auto env = info.Env();
        OPENVINO_ASSERT(pipe, "OmniPipeline is not initialized");
        auto operation = std::make_shared<PipelineOperationGuard>(
            state, PipelineOperation::GENERATE, "Another generation is already in progress");
        VALIDATE_ARGS_COUNT(info, 10, "generate()");
        auto inputs = js_to_cpp<VLMGenerateInputs>(env, info[0]);
        std::vector<ov::Tensor> images, videos, audios;
        std::vector<ov::genai::VideoMetadata> videos_metadata;
        set_if_defined(env, info[1], images);
        set_if_defined(env, info[2], videos);
        set_if_defined(env, info[3], videos_metadata);
        set_if_defined(env, info[4], audios);
        OPENVINO_ASSERT(info[5].IsFunction() || info[5].IsUndefined(), "streamer must be a function or undefined");
        OPENVINO_ASSERT(info[6].IsFunction() || info[6].IsUndefined(), "speechStreamer must be a function or undefined");
        auto text_config = js_to_cpp<ov::genai::GenerationConfig>(env, info[7]);
        ov::genai::OmniTalkerSpeechConfig speech_config;
        set_if_defined(env, info[8], speech_config);
        OPENVINO_ASSERT(info[9].IsFunction(), "generate callback is not a function");
        auto context = std::make_unique<InferenceThreadContext>(std::move(operation), "omniPerformInferenceThread");
        if (info[5].IsFunction()) {
            context->streamer =
                std::make_unique<ThreadSafeCallback>(env, info[5].As<Napi::Function>(), "Omni_text_streamer");
        }
        if (info[6].IsFunction()) {
            context->speech_streamer =
                std::make_unique<ThreadSafeCallback>(env, info[6].As<Napi::Function>(), "Omni_speech_streamer");
        }
        context->run_generate = [context = context.get(),
                                 pipe = pipe,
                                 inputs = std::move(inputs),
                                 images = std::move(images),
                                 videos = std::move(videos),
                                 audios = std::move(audios),
                                 videos_metadata = std::move(videos_metadata),
                                 text_config = std::move(text_config),
                                 speech_config = std::move(speech_config)]() mutable -> JsResultProducer {
            ov::genai::StreamerVariant streamer = std::monostate();
            if (context->streamer) {
                streamer = make_text_streamer(context);
            }
            ov::genai::OmniSpeechStreamerVariant speech_streamer = std::monostate();
            if (context->speech_streamer) {
                speech_streamer = [context](const ov::Tensor& chunk) {
                    try {
                        return context->speech_streamer->call_and_wait<ov::genai::StreamingStatus>(
                            [chunk](Napi::Env env, Napi::Function callback) {
                                return call_js_callback(callback, {cpp_to_js<ov::Tensor, Napi::Value>(env, chunk)});
                            },
                            to_streaming_status);
                    } catch (const std::exception&) {
                        context->record_callback_exception(std::current_exception());
                        return ov::genai::StreamingStatus::CANCEL;
                    }
                };
            }
            ov::genai::OmniDecodedResults result;
            std::visit(
                [&](const auto& prompt) {
                    result = pipe->generate(prompt,
                                            images,
                                            videos,
                                            videos_metadata,
                                            audios,
                                            text_config,
                                            speech_config,
                                            streamer,
                                            speech_streamer);
                },
                inputs);
            return [result = std::move(result)](Napi::Env env) -> Napi::Value {
                return to_omni_decoded_result(env, result);
            };
        };
        start_inference(info, info[9].As<Napi::Function>(), std::move(context));
        return env.Undefined();
    });
}
