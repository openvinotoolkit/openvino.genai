// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/inpainting_pipeline/pipeline_wrapper.hpp"

#include <filesystem>
#include <future>
#include <iostream>
#include <optional>
#include <thread>

#include "include/helper.hpp"
#include "include/image_decode_worker.hpp"
#include "include/inpainting_pipeline/init_worker.hpp"

namespace {

struct InpaintingTsfnContext {
    InpaintingTsfnContext(std::string prompt,
                          ov::Tensor image,
                          ov::Tensor mask_image,
                          ov::AnyMap generation_properties,
                          std::shared_ptr<std::atomic<bool>> is_busy,
                          std::shared_ptr<std::atomic<bool>> is_generating)
        : prompt(std::move(prompt)),
          image(std::move(image)),
          mask_image(std::move(mask_image)),
          generation_properties(std::move(generation_properties)),
          is_busy(std::move(is_busy)),
          is_generating(std::move(is_generating)) {}

    std::thread native_thread;
    Napi::ThreadSafeFunction callback_tsfn;
    std::optional<Napi::ThreadSafeFunction> streamer_tsfn;
    std::string prompt;
    ov::Tensor image;
    ov::Tensor mask_image;
    ov::AnyMap generation_properties;
    std::vector<std::string> callback_exceptions;
    std::shared_ptr<std::atomic<bool>> is_busy;
    std::shared_ptr<std::atomic<bool>> is_generating;
    std::shared_ptr<ov::genai::InpaintingPipeline> pipe = nullptr;
};

void inpainting_perform_inference_thread(InpaintingTsfnContext* context) {
    auto report_error = [context](const std::string& message) {
        auto status = context->callback_tsfn.BlockingCall([message](Napi::Env env, Napi::Function js_callback) {
            try {
                js_callback.Call({Napi::Error::New(env, "inpainting_perform_inference_thread error. " + message).Value(),
                                  env.Null()});
            } catch (const std::exception& err) {
                std::cerr << "The callback failed when returning an error from inpainting_perform_inference_thread. Details:\n"
                          << err.what() << std::endl;
                std::cerr << "Original error message:\n" << message << std::endl;
            }
        });
        if (status != napi_ok) {
            std::cerr << "The BlockingCall failed with status " << status
                      << " when trying to return an error from inpainting_perform_inference_thread." << std::endl;
            std::cerr << "Original error message:\n" << message << std::endl;
        }
    };
    auto finalize = [context]() {
        context->callback_tsfn.Release();
        if (context->streamer_tsfn.has_value()) {
            context->streamer_tsfn->Release();
        }
    };

    try {
        if (context->streamer_tsfn.has_value()) {
            context->generation_properties["callback"] =
                std::function<bool(size_t, size_t, ov::Tensor&)>(
                    [context](size_t step, size_t num_steps, ov::Tensor& latent) -> bool {
                        auto result_promise = std::make_shared<std::promise<bool>>();
                        auto result_future = result_promise->get_future();
                        // Release the inference request while the JS step callback runs so it may call decode().
                        context->is_busy->store(false);
                        napi_status status = context->streamer_tsfn->BlockingCall(
                            [step, num_steps, &latent, result_promise, context](
                                Napi::Env env, Napi::Function js_callback) {
                                try {
                                    auto js_result =
                                        js_callback.Call({Napi::Number::New(env, static_cast<double>(step)),
                                                          Napi::Number::New(env, static_cast<double>(num_steps)),
                                                          cpp_to_js<ov::Tensor, Napi::Value>(env, latent)});
                                    if (js_result.IsBoolean()) {
                                        result_promise->set_value(js_result.As<Napi::Boolean>().Value());
                                    } else if (js_result.IsPromise()) {
                                        Napi::Object promise = js_result.As<Napi::Object>();
                                        Napi::Function then = promise.Get("then").As<Napi::Function>();
                                        auto on_fulfilled = Napi::Function::New(
                                            env, [result_promise, context](const Napi::CallbackInfo& cb) {
                                                if (cb.Length() > 0 && cb[0].IsBoolean()) {
                                                    result_promise->set_value(cb[0].As<Napi::Boolean>().Value());
                                                } else {
                                                    context->callback_exceptions.push_back(
                                                        "Step callback must resolve to a boolean.");
                                                    result_promise->set_value(true);  // stop on invalid resolved value
                                                }
                                            });
                                        auto on_rejected = Napi::Function::New(
                                            env, [result_promise, context](const Napi::CallbackInfo& cb) {
                                                std::string message = "Step callback promise rejected";
                                                if (cb.Length() > 0 && cb[0].IsObject()) {
                                                    Napi::Value msg = cb[0].As<Napi::Object>().Get("message");
                                                    if (msg.IsString()) {
                                                        message = msg.As<Napi::String>().Utf8Value();
                                                    }
                                                }
                                                context->callback_exceptions.push_back(message);
                                                result_promise->set_value(true);  // stop on rejection
                                            });
                                        then.Call(promise, {on_fulfilled, on_rejected});
                                    } else {
                                        context->callback_exceptions.push_back(
                                            "Step callback must return a boolean or a Promise<boolean>.");
                                        result_promise->set_value(true);  // stop on invalid return type
                                    }
                                } catch (const std::exception& err) {
                                    context->callback_exceptions.push_back(err.what());
                                    result_promise->set_value(true);  // stop on exception
                                }
                            });
                        if (status != napi_ok) {
                            context->is_busy->store(true);
                            context->callback_exceptions.push_back(
                                "Step callback BlockingCall failed with status: " +
                                std::to_string(static_cast<int>(status)));
                            return true;  // stop
                        }
                        bool stop = result_future.get();
                        context->is_busy->store(true);
                        return stop;
                    });
        }

        ov::Tensor result = context->pipe->generate(context->prompt,
                                                    context->image,
                                                    context->mask_image,
                                                    context->generation_properties);
        context->is_busy->store(false);
        context->is_generating->store(false);

        if (!context->callback_exceptions.empty()) {
            std::string combined_error = "Step callback exceptions occurred:\n";
            for (size_t i = 0; i < context->callback_exceptions.size(); ++i) {
                combined_error += "[" + std::to_string(i + 1) + "] " + context->callback_exceptions[i] + "\n";
            }
            report_error(combined_error);
        } else {
            auto final_result = std::make_shared<ov::Tensor>(std::move(result));
            auto final_callback_error = std::make_shared<std::string>();

            napi_status status = context->callback_tsfn.BlockingCall(
                [final_result, final_callback_error](Napi::Env env, Napi::Function js_callback) {
                    try {
                        js_callback.Call({env.Null(), cpp_to_js<ov::Tensor, Napi::Value>(env, *final_result)});
                    } catch (const std::exception& err) {
                        *final_callback_error = "The final callback failed. Details:\n" + std::string(err.what());
                    }
                });
            if (status != napi_ok) {
                report_error("The final BlockingCall failed with status " +
                             std::to_string(static_cast<int>(status)));
            } else if (!final_callback_error->empty()) {
                report_error(*final_callback_error);
            }
        }
    } catch (const std::exception& ex) {
        context->is_busy->store(false);
        context->is_generating->store(false);
        report_error(ex.what());
    }

    finalize();
}

}  // namespace

InpaintingPipelineWrapper::InpaintingPipelineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<InpaintingPipelineWrapper>(info) {}

Napi::Function InpaintingPipelineWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "InpaintingPipeline",
                       {
                           InstanceMethod("init", &InpaintingPipelineWrapper::init),
                           InstanceMethod("generate", &InpaintingPipelineWrapper::generate),
                           InstanceMethod("decode", &InpaintingPipelineWrapper::decode),
                           InstanceMethod("getPerformanceMetrics",
                                          &InpaintingPipelineWrapper::get_performance_metrics),
                           InstanceMethod("getGenerationConfig", &InpaintingPipelineWrapper::get_generation_config),
                           InstanceMethod("setGenerationConfig", &InpaintingPipelineWrapper::set_generation_config),
                       });
}

Napi::Value InpaintingPipelineWrapper::init(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(!this->pipe, "Pipeline is already initialized");
        OPENVINO_ASSERT(!this->is_initializing->load(), "Pipeline is already initializing");
        this->is_initializing->store(true);

        VALIDATE_ARGS_COUNT(info, 4, "init()");
        auto model_path = js_to_cpp<std::filesystem::path>(env, info[0]);
        auto device = js_to_cpp<std::string>(env, info[1]);
        auto properties = js_to_cpp<ov::AnyMap>(env, info[2]);
        OPENVINO_ASSERT(info[3].IsFunction(), "init callback is not a function");
        Napi::Function callback = info[3].As<Napi::Function>();

        auto async_worker = new InpaintingInitWorker(callback,
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

Napi::Value InpaintingPipelineWrapper::generate(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "InpaintingPipeline is not initialized");
        OPENVINO_ASSERT(!this->is_busy->load() && !this->is_generating->load(),
                        "generate() cannot run while another generate() or decode() is in progress");
        this->is_busy->store(true);
        this->is_generating->store(true);

        // generate(prompt, image, mask, properties, streamer, doneCallback)
        VALIDATE_ARGS_COUNT(info, 6, "generate()");
        OPENVINO_ASSERT(info[0].IsString(), "InpaintingPipeline.generate() expects a prompt string.");
        OPENVINO_ASSERT(info[4].IsUndefined() || info[4].IsNull() || info[4].IsFunction(),
                        "InpaintingPipeline.generate() expects streamer to be a function, null or undefined.");
        OPENVINO_ASSERT(info[5].IsFunction(), "generate done callback is not a function");

        auto prompt = js_to_cpp<std::string>(env, info[0]);
        auto image = js_to_cpp<ov::Tensor>(env, info[1]);
        const auto& image_shape = image.get_shape();
        OPENVINO_ASSERT(image.get_element_type() == ov::element::u8,
                        "InpaintingPipeline.generate() expects image tensor with u8 element type, got ",
                        image.get_element_type(),
                        ".");
        OPENVINO_ASSERT(image_shape.size() == 4 && image_shape[0] == 1 && image_shape[3] == 3,
                        "InpaintingPipeline.generate() expects image tensor with batched NHWC shape [1, H, W, 3], got ",
                        image_shape,
                        ".");
        auto mask_image = js_to_cpp<ov::Tensor>(env, info[2]);
        const auto& mask_shape = mask_image.get_shape();
        OPENVINO_ASSERT(mask_image.get_element_type() == ov::element::u8,
                        "InpaintingPipeline.generate() expects mask tensor with u8 element type, got ",
                        mask_image.get_element_type(),
                        ".");
        OPENVINO_ASSERT(mask_shape.size() == 4 && mask_shape[0] == 1 && mask_shape[3] == 3,
                        "InpaintingPipeline.generate() expects mask tensor with batched NHWC shape [1, H, W, 3], got ",
                        mask_shape,
                        ".");
        auto generation_properties = js_to_cpp<ov::AnyMap>(env, info[3]);
        Napi::Function callback = info[5].As<Napi::Function>();

        auto* context = new InpaintingTsfnContext(std::move(prompt),
                                                  std::move(image),
                                                  std::move(mask_image),
                                                  std::move(generation_properties),
                                                  this->is_busy,
                                                  this->is_generating);
        context->pipe = this->pipe;

        // streamer at index 4 can be a function or null/undefined
        if (info[4].IsFunction()) {
            context->streamer_tsfn =
                Napi::ThreadSafeFunction::New(env, info[4].As<Napi::Function>(), "Inpainting_streamer", 0, 1);
        }

        context->callback_tsfn =
            Napi::ThreadSafeFunction::New(env, callback, "Inpainting_generate_callback", 0, 1, [context](Napi::Env) {
                if (context->native_thread.joinable()) {
                    context->native_thread.join();
                }
                delete context;
            });

        context->native_thread = std::thread(inpainting_perform_inference_thread, context);
    } catch (const std::exception& ex) {
        this->is_busy->store(false);
        this->is_generating->store(false);
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }

    return env.Undefined();
}

Napi::Value InpaintingPipelineWrapper::decode(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "InpaintingPipeline is not initialized");
        OPENVINO_ASSERT(!this->is_busy->load(),
                        "decode() cannot run while another generate() or decode() is in progress");
        VALIDATE_ARGS_COUNT(info, 2, "decode()");
        auto latent = js_to_cpp<ov::Tensor>(env, info[0]);
        OPENVINO_ASSERT(info[1].IsFunction(), "decode callback is not a function");
        Napi::Function callback = info[1].As<Napi::Function>();

        this->is_busy->store(true);
        auto* async_worker =
            new ImageDecodeWorker<ov::genai::InpaintingPipeline>(callback, this->pipe, std::move(latent), this->is_busy);
        async_worker->Queue();
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value InpaintingPipelineWrapper::get_generation_config(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "InpaintingPipeline is not initialized");
        return cpp_to_js<ov::genai::ImageGenerationConfig, Napi::Value>(env, this->pipe->get_generation_config());
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}

Napi::Value InpaintingPipelineWrapper::set_generation_config(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "InpaintingPipeline is not initialized");
        VALIDATE_ARGS_COUNT(info, 1, "setGenerationConfig()");
        OPENVINO_ASSERT(!info[0].IsUndefined() && !info[0].IsNull(), "Generation config cannot be undefined or null");
        auto config = js_to_cpp<ov::genai::ImageGenerationConfig>(env, info[0]);
        this->pipe->set_generation_config(config);
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

Napi::Value InpaintingPipelineWrapper::get_performance_metrics(const Napi::CallbackInfo& info) {
    auto env = info.Env();
    try {
        OPENVINO_ASSERT(this->pipe, "InpaintingPipeline is not initialized");
        return cpp_to_js<ov::genai::ImageGenerationPerfMetrics, Napi::Value>(
            env,
            this->pipe->get_performance_metrics());
    } catch (const std::exception& ex) {
        Napi::Error::New(env, ex.what()).ThrowAsJavaScriptException();
        return env.Undefined();
    }
}
