// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/parser.hpp"

#include <future>

#include "include/addon.hpp"
#include "include/helper.hpp"

JSParser::JSParser(Napi::Env env, Napi::Object jsParser) {
    js_parser_ref = Napi::Persistent(jsParser);
    Napi::Function js_parse_fn = jsParser.Get("parse").As<Napi::Function>();
    parser_tsfn = Napi::ThreadSafeFunction::New(env,
                                                js_parse_fn,        // js callback
                                                "js_parser_parse",  // resource name
                                                0,                  // unlimited queue
                                                1                   // initialThreadCount
    );
};

void JSParser::parse(ov::genai::JsonContainer& message) {
    std::promise<ov::genai::JsonContainer> promise;
    parser_tsfn.BlockingCall([&message, &promise, this](Napi::Env env, Napi::Function js_parse_fn) {
        auto js_message = cpp_to_js<ov::genai::JsonContainer, Napi::Value>(env, message);
        js_parse_fn.Call(js_parser_ref.Value(), {js_message});
        promise.set_value(js_to_cpp<ov::genai::JsonContainer>(env, js_message));
    });
    message = promise.get_future().get();
}

Napi::Object JSParser::get_js_object(Napi::Env env) const {
    return js_parser_ref.Value();
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

void ReasoningParserWrapper::set_parser(std::shared_ptr<ov::genai::ReasoningParser> parser) {
    _parser = std::move(parser);
}

Napi::Object ReasoningParserWrapper::wrap(Napi::Env env, std::shared_ptr<ov::genai::ReasoningParser> parser) {
    const auto& prototype = env.GetInstanceData<AddonData>()->reasoning_parser;
    OPENVINO_ASSERT(prototype, "Invalid pointer to ReasoningParser prototype.");
    Napi::Object obj = prototype.Value().As<Napi::Function>().New({});
    Napi::ObjectWrap<ReasoningParserWrapper>::Unwrap(obj)->set_parser(std::move(parser));
    return obj;
}

DeepSeekR1ReasoningParserWrapper::DeepSeekR1ReasoningParserWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<DeepSeekR1ReasoningParserWrapper>(info) {
    try {
        _parser = std::make_shared<ov::genai::DeepSeekR1ReasoningParser>();
    } catch (const std::exception& e) {
        Napi::Error::New(info.Env(), e.what()).ThrowAsJavaScriptException();
    }
}

Napi::Function DeepSeekR1ReasoningParserWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "DeepSeekR1ReasoningParser",
                       {InstanceMethod("parse", &DeepSeekR1ReasoningParserWrapper::parse)});
}

void DeepSeekR1ReasoningParserWrapper::parse(const Napi::CallbackInfo& info) {
    parse_with_parser(info, _parser);
}

std::shared_ptr<ov::genai::DeepSeekR1ReasoningParser> DeepSeekR1ReasoningParserWrapper::get_parser() {
    return _parser;
}

void DeepSeekR1ReasoningParserWrapper::set_parser(std::shared_ptr<ov::genai::DeepSeekR1ReasoningParser> parser) {
    _parser = std::move(parser);
}

Napi::Object DeepSeekR1ReasoningParserWrapper::wrap(Napi::Env env,
                                                    std::shared_ptr<ov::genai::DeepSeekR1ReasoningParser> parser) {
    const auto& prototype = env.GetInstanceData<AddonData>()->deepseek_r1_reasoning_parser;
    OPENVINO_ASSERT(prototype, "Invalid pointer to DeepSeekR1ReasoningParser prototype.");
    Napi::Object obj = prototype.Value().As<Napi::Function>().New({});
    Napi::ObjectWrap<DeepSeekR1ReasoningParserWrapper>::Unwrap(obj)->set_parser(std::move(parser));
    return obj;
}

Phi4ReasoningParserWrapper::Phi4ReasoningParserWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<Phi4ReasoningParserWrapper>(info) {
    Napi::Env env = info.Env();

    try {
        _parser = std::make_shared<ov::genai::Phi4ReasoningParser>();
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    }
}

Napi::Function Phi4ReasoningParserWrapper::get_class(Napi::Env env) {
    return DefineClass(env, "Phi4ReasoningParser", {InstanceMethod("parse", &Phi4ReasoningParserWrapper::parse)});
}

void Phi4ReasoningParserWrapper::parse(const Napi::CallbackInfo& info) {
    parse_with_parser(info, _parser);
}

std::shared_ptr<ov::genai::Phi4ReasoningParser> Phi4ReasoningParserWrapper::get_parser() {
    return _parser;
}

void Phi4ReasoningParserWrapper::set_parser(std::shared_ptr<ov::genai::Phi4ReasoningParser> parser) {
    _parser = std::move(parser);
}

Napi::Object Phi4ReasoningParserWrapper::wrap(Napi::Env env, std::shared_ptr<ov::genai::Phi4ReasoningParser> parser) {
    const auto& prototype = env.GetInstanceData<AddonData>()->phi4_reasoning_parser;
    OPENVINO_ASSERT(prototype, "Invalid pointer to Phi4ReasoningParser prototype.");
    Napi::Object obj = prototype.Value().As<Napi::Function>().New({});
    Napi::ObjectWrap<Phi4ReasoningParserWrapper>::Unwrap(obj)->set_parser(std::move(parser));
    return obj;
}

Llama3PythonicToolParserWrapper::Llama3PythonicToolParserWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<Llama3PythonicToolParserWrapper>(info) {
    Napi::Env env = info.Env();

    try {
        _parser = std::make_shared<ov::genai::Llama3PythonicToolParser>();
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    }
}

Napi::Function Llama3PythonicToolParserWrapper::get_class(Napi::Env env) {
    return DefineClass(env,
                       "Llama3PythonicToolParser",
                       {InstanceMethod("parse", &Llama3PythonicToolParserWrapper::parse)});
}

void Llama3PythonicToolParserWrapper::parse(const Napi::CallbackInfo& info) {
    parse_with_parser(info, _parser);
}

std::shared_ptr<ov::genai::Llama3PythonicToolParser> Llama3PythonicToolParserWrapper::get_parser() {
    return _parser;
}

void Llama3PythonicToolParserWrapper::set_parser(std::shared_ptr<ov::genai::Llama3PythonicToolParser> parser) {
    _parser = std::move(parser);
}

Napi::Object Llama3PythonicToolParserWrapper::wrap(Napi::Env env,
                                                   std::shared_ptr<ov::genai::Llama3PythonicToolParser> parser) {
    const auto& prototype = env.GetInstanceData<AddonData>()->llama3_pythonic_tool_parser;
    OPENVINO_ASSERT(prototype, "Invalid pointer to Llama3PythonicToolParser prototype.");
    Napi::Object obj = prototype.Value().As<Napi::Function>().New({});
    Napi::ObjectWrap<Llama3PythonicToolParserWrapper>::Unwrap(obj)->set_parser(std::move(parser));
    return obj;
}

Llama3JsonToolParserWrapper::Llama3JsonToolParserWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<Llama3JsonToolParserWrapper>(info) {
    Napi::Env env = info.Env();

    try {
        _parser = std::make_shared<ov::genai::Llama3JsonToolParser>();
    } catch (const std::exception& e) {
        Napi::Error::New(env, e.what()).ThrowAsJavaScriptException();
    }
}

Napi::Function Llama3JsonToolParserWrapper::get_class(Napi::Env env) {
    return DefineClass(env, "Llama3JsonToolParser", {InstanceMethod("parse", &Llama3JsonToolParserWrapper::parse)});
}

void Llama3JsonToolParserWrapper::parse(const Napi::CallbackInfo& info) {
    parse_with_parser(info, _parser);
}

std::shared_ptr<ov::genai::Llama3JsonToolParser> Llama3JsonToolParserWrapper::get_parser() {
    return _parser;
}

void Llama3JsonToolParserWrapper::set_parser(std::shared_ptr<ov::genai::Llama3JsonToolParser> parser) {
    _parser = std::move(parser);
}

Napi::Object Llama3JsonToolParserWrapper::wrap(Napi::Env env, std::shared_ptr<ov::genai::Llama3JsonToolParser> parser) {
    const auto& prototype = env.GetInstanceData<AddonData>()->llama3_json_tool_parser;
    OPENVINO_ASSERT(prototype, "Invalid pointer to Llama3JsonToolParser prototype.");
    Napi::Object obj = prototype.Value().As<Napi::Function>().New({});
    Napi::ObjectWrap<Llama3JsonToolParserWrapper>::Unwrap(obj)->set_parser(std::move(parser));
    return obj;
}
