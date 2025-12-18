// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/parser.hpp"

#include <future>

#include "include/addon.hpp"
#include "include/helper.hpp"

JSParser::JSParser(Napi::Env env, Napi::Object jsParser) {
    js_parser_ref = Napi::Persistent(jsParser);
    js_parser_ref.SuppressDestruct();
    Napi::Function parseFn = jsParser.Get("parse").As<Napi::Function>();
    parser_tsfn = Napi::ThreadSafeFunction::New(env,
                                                parseFn,            // js callback
                                                "js_parser_parse",  // resource name
                                                0,                  // unlimited queue
                                                1                   // initialThreadCount
    );
};

void JSParser::parse(ov::genai::JsonContainer& message) {
    std::promise<ov::genai::JsonContainer> promise;
    parser_tsfn.BlockingCall([&message, &promise, this](Napi::Env env, Napi::Function jsCallback) {
        auto js_message = cpp_to_js<ov::genai::JsonContainer, Napi::Value>(env, message);
        jsCallback.Call(js_parser_ref.Value(), {js_message});
        promise.set_value(js_to_cpp<ov::genai::JsonContainer>(env, js_message));
    });
    message = promise.get_future().get();
}
