// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/whisper_pipeline/pipeline_wrapper.hpp"

#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <thread>

#include "include/helper.hpp"
#include "include/tokenizer.hpp"
#include "include/whisper_pipeline/init_worker.hpp"

struct WhisperTsfnContext {
    WhisperTsfnContext(std::vector<float> raw_speech, ov::AnyMap generation_config, std::shared_ptr<std::atomic<bool>> is_generating)
        : raw_speech(std::move(raw_speech)),
          generation_config(std::move(generation_config)),
          is_generating(is_generating) {}
    ~WhisperTsfnContext() = default;

    std::thread native_thread;
    Napi::ThreadSafeFunction callback_tsfn;
    std::optional<Napi::ThreadSafeFunction> streamer_tsfn;

    std::vector<float> raw_speech;
    ov::AnyMap generation_config;
    std::shared_ptr<std::atomic<bool>> is_generating;
    std::shared_ptr<ov::genai::WhisperPipeline> pipe = nullptr;
};

void whisperPerformInferenceThread(WhisperTsfnContext* context) {
    auto report_error = [context](const std::string& message) {
        auto status = context->callback_tsfn.BlockingCall([message](Napi::Env env, Napi::Function js_callback) {
            try {
                js_callback.Call(
                    {Napi::Error::New(env, "whisperPerformInferenceThread error. " + message).Value(), env.Null()});
            } catch (const std::exception& err) {
                std::cerr
                    << "The callback failed when attempting to return an error from whisperPerformInferenceThread. "
                       "Details:\n"
                    << err.what() << std::endl;
                std::cerr << "Original error message:\n" << message << std::endl;
            }
        });
        if (status != napi_ok) {
            std::cerr << "The BlockingCall failed with status " << status
                      << " when trying to return an error from whisperPerformInferenceThread." << std::endl;
            std::cerr << "Original error message:\n" << message << std::endl;
        }
    };
    auto finalize = [context]() {
        context->callback_tsfn.Release();
        if (context->streamer_tsfn.has_value()) {
            context->streamer_tsfn->Release();
        }
    };
    std::vector<std::string> streamer_exceptions;
    ov::genai::WhisperDecodedResults result;

    try {
        ov::genai::StreamerVariant streamer_var = std::monostate();
        if (context->streamer_tsfn.has_value()) {
            streamer_var = [context, &streamer_exceptions](std::string word) {
                std::promise<ov::genai::StreamingStatus> result_promise;
                napi_status status = context->streamer_tsfn->BlockingCall(
                    [word, &result_promise, &streamer_exceptions](Napi::Env env, Napi::Function js_callback) {
                        try {
                            auto callback_result = js_callback.Call({Napi::String::New(env, word)});
                            if (callback_result.IsNumber()) {
                                result_promise.set_value(static_cast<ov::genai::StreamingStatus>(
                                    callback_result.As<Napi::Number>().Int32Value()));
                            } else {
                                result_promise.set_value(ov::genai::StreamingStatus::RUNNING);
                            }
                        } catch (const std::exception& err) {
                            streamer_exceptions.push_back(err.what());
                            result_promise.set_value(ov::genai::StreamingStatus::CANCEL);
                        }
                    });

                if (status != napi_ok) {
                    streamer_exceptions.push_back("The streamer callback BlockingCall failed with status: " +
                                                  std::to_string(static_cast<int>(status)));
                    return ov::genai::StreamingStatus::CANCEL;
                }
                return result_promise.get_future().get();
            };
        }

        if (context->streamer_tsfn.has_value()) {
            auto config = context->pipe->get_generation_config();
            config.update_generation_config(context->generation_config);
            result = context->pipe->generate(context->raw_speech, config, streamer_var);
        } else {
            result = context->pipe->generate(context->raw_speech, context->generation_config);
        }
    } catch (const std::exception& e) {
        *context->is_generating = false;
        report_error(e.what());
        finalize();
        return;
    }

    *context->is_generating = false;

    try {
        if (!streamer_exceptions.empty()) {
            std::string combined_error = "Streamer exceptions occurred:\n";
            for (size_t i = 0; i < streamer_exceptions.size(); ++i) {
                combined_error += "[" + std::to_string(i + 1) + "] " + streamer_exceptions[i] + "\n";
            }
            report_error(combined_error);
        } else {
            std::shared_ptr<ov::genai::WhisperDecodedResults> final_result =
                std::make_shared<ov::genai::WhisperDecodedResults>(std::move(result));
            std::shared_ptr<std::string> final_callback_error = std::make_shared<std::string>();

            napi_status status = context->callback_tsfn.BlockingCall(
                [final_result, final_callback_error](Napi::Env env, Napi::Function js_callback) {
                    try {
                        js_callback.Call({env.Null(), to_whisper_decoded_result(env, *final_result)});
                    } catch (const std::exception& err) {
                        *final_callback_error = "The final callback failed. Details:\n" + std::string(err.what());
                    }
                });
            if (status != napi_ok) {
                report_error("The final BlockingCall failed with status " + std::to_string(static_cast<int>(status)));
            } else if (!final_callback_error->empty()) {
                report_error(*final_callback_error);
            }
        }
    } catch (const std::exception& e) {
        report_error(e.what());
    }
    finalize();
}

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

        auto* context =
            new WhisperTsfnContext(std::move(raw_speech), std::move(generation_config), this->is_generating);
        context->pipe = this->pipe;

        context->callback_tsfn =
            Napi::ThreadSafeFunction::New(env, callback, "Whisper_generate_callback", 0, 1, [context](Napi::Env) {
                context->native_thread.join();
                delete context;
            });

        if (info[2].IsFunction()) {
            context->streamer_tsfn =
                Napi::ThreadSafeFunction::New(env, info[2].As<Napi::Function>(), "Whisper_generate_streamer", 0, 1);
        }

        context->native_thread = std::thread(whisperPerformInferenceThread, context);
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
