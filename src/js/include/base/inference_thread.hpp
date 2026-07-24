// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <napi.h>

#include "openvino/genai/streamer_base.hpp"

// Type-erased producer that converts the typed generate() result into a JS value. It is built on the worker thread
// once generation has finished and invoked on the Node.js thread inside the callback ThreadSafeFunction.
using JsResultProducer = std::function<Napi::Value(Napi::Env)>;

// Shared state and hooks for running any pipeline's asynchronous generate() on a dedicated worker thread and
// reporting the result or an error back to JS through ThreadSafeFunctions. Every pipeline fills run_generate and,
// when needed, on_finished with its own logic; perform_generate_thread drives the identical worker loop for all.
struct InferenceThreadContext {
    InferenceThreadContext(std::shared_ptr<std::atomic<bool>> is_generating,
                           std::string thread_name,
                           std::string streamer_exception_header = "Streamer exceptions occurred:")
        : is_generating(std::move(is_generating)),
          thread_name(std::move(thread_name)),
          streamer_exception_header(std::move(streamer_exception_header)) {}

    std::thread native_thread;
    Napi::ThreadSafeFunction callback_tsfn;
    std::optional<Napi::ThreadSafeFunction> streamer_tsfn;
    std::vector<std::string> callback_exceptions;
    std::shared_ptr<std::atomic<bool>> is_generating;

    // Runs the pipeline's generate() (building its streamer from this context as needed) and returns a producer
    // that converts the typed result into a JS value on the Node.js thread.
    std::function<JsResultProducer()> run_generate;
    // Releases any extra busy flags held during inference, e.g. the image-generation request guard. No-op by default.
    std::function<void()> on_finished = [] {};
    // Names the worker thread in diagnostics reported back to JS.
    std::string thread_name;
    // Header prefixing the collected callback exceptions reported back to JS. Image-generation pipelines override it
    // because they collect exceptions from a step callback instead of a streamer.
    std::string streamer_exception_header;
};

// Builds a text streamer that forwards each produced word to the JS streamer ThreadSafeFunction and returns the
// StreamingStatus reported by JS. Exceptions and failed BlockingCalls are recorded in callback_exceptions and stop
// generation. Shared by the text-producing pipelines (LLM, VLM, Whisper).
std::function<ov::genai::StreamingStatus(std::string)> make_text_streamer(InferenceThreadContext* context);

// Drives the worker thread shared by every pipeline: runs generate() through context->run_generate, releases the
// busy flags, then reports either the collected callback exceptions or the final result back to JS.
void perform_generate_thread(InferenceThreadContext* context);
