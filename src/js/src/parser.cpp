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

void parse_with_parser(const Napi::CallbackInfo& info, std::shared_ptr<ov::genai::Parser> parser) {
    Napi::Env env = info.Env();
    
    try {
        OPENVINO_ASSERT(info.Length() >= 1 && info[0].IsObject(), "Expected an object as the first argument");
        auto js_obj = info[0].As<Napi::Object>();
        auto message = js_to_cpp<ov::genai::JsonContainer>(env, info[0]);
        parser->parse(message);

        // Update JS object with parsed content
        Napi::Object global = env.Global();
        Napi::Object obj = global.Get("Object").As<Napi::Object>();
        Napi::Function assign = obj.Get("assign").As<Napi::Function>();
        assign.Call(obj, {js_obj, cpp_to_js<ov::genai::JsonContainer, Napi::Value>(env, message)});
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    }
}

ReasoningParserWrapper::ReasoningParserWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<ReasoningParserWrapper>(info) {
    Napi::Env env = info.Env();

    try {
        // Parse optional arguments
        if (info.Length() > 0 && info[0].IsObject()) {
            Napi::Object options = info[0].As<Napi::Object>();

            bool expect_open_tag = true;
            bool keep_original_content = true;
            std::string open_tag = "<think>";
            std::string close_tag = "</think>";

            if (options.Has("expectOpenTag") && options.Get("expectOpenTag").IsBoolean()) {
                expect_open_tag = options.Get("expectOpenTag").As<Napi::Boolean>().Value();
            }

            if (options.Has("keepOriginalContent") && options.Get("keepOriginalContent").IsBoolean()) {
                keep_original_content = options.Get("keepOriginalContent").As<Napi::Boolean>().Value();
            }

            if (options.Has("openTag") && options.Get("openTag").IsString()) {
                open_tag = options.Get("openTag").As<Napi::String>().Utf8Value();
            }

            if (options.Has("closeTag") && options.Get("closeTag").IsString()) {
                close_tag = options.Get("closeTag").As<Napi::String>().Utf8Value();
            }

            _parser = std::make_shared<ov::genai::ReasoningParser>(expect_open_tag,
                                                                   keep_original_content,
                                                                   open_tag,
                                                                   close_tag);
        } else {
            // Use default parameters from C++ constructor
            _parser = std::make_shared<ov::genai::ReasoningParser>();
        }
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    }
}

Napi::Function ReasoningParserWrapper::get_class(Napi::Env env) {
    return DefineClass(env, "ReasoningParser", {InstanceMethod("parse", &ReasoningParserWrapper::parse)});
}

void ReasoningParserWrapper::parse(const Napi::CallbackInfo& info) {
    parse_with_parser(info, _parser);
}

std::shared_ptr<ov::genai::ReasoningParser> ReasoningParserWrapper::get_parser() {
    return _parser;
}
