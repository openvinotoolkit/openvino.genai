// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

#include "openvino/genai/parsers.hpp"

class JSParser : public ov::genai::Parser {
public:
    JSParser(Napi::Env env, Napi::Object jsParser);
    ~JSParser() override {
        parser_tsfn.Release();
        js_parser_ref.Unref();
    };
    void parse(ov::genai::JsonContainer& message) override;
    Napi::Object get_js_object(Napi::Env env) const;

private:
    Napi::ThreadSafeFunction parser_tsfn;
    Napi::ObjectReference js_parser_ref;
};

class ReasoningParserWrapper : public Napi::ObjectWrap<ReasoningParserWrapper> {
public:
    ReasoningParserWrapper(const Napi::CallbackInfo& info);

    static Napi::Function get_class(Napi::Env env);
    static Napi::Object wrap(Napi::Env env, std::shared_ptr<ov::genai::ReasoningParser> parser);

    void parse(const Napi::CallbackInfo& info);

    std::shared_ptr<ov::genai::ReasoningParser> get_parser();
    void set_parser(std::shared_ptr<ov::genai::ReasoningParser> parser);

private:
    std::shared_ptr<ov::genai::ReasoningParser> _parser;
};

class DeepSeekR1ReasoningParserWrapper : public Napi::ObjectWrap<DeepSeekR1ReasoningParserWrapper> {
public:
    DeepSeekR1ReasoningParserWrapper(const Napi::CallbackInfo& info);

    static Napi::Function get_class(Napi::Env env);
    static Napi::Object wrap(Napi::Env env, std::shared_ptr<ov::genai::DeepSeekR1ReasoningParser> parser);

    void parse(const Napi::CallbackInfo& info);

    std::shared_ptr<ov::genai::DeepSeekR1ReasoningParser> get_parser();
    void set_parser(std::shared_ptr<ov::genai::DeepSeekR1ReasoningParser> parser);

private:
    std::shared_ptr<ov::genai::DeepSeekR1ReasoningParser> _parser;
};

class Phi4ReasoningParserWrapper : public Napi::ObjectWrap<Phi4ReasoningParserWrapper> {
public:
    Phi4ReasoningParserWrapper(const Napi::CallbackInfo& info);

    static Napi::Function get_class(Napi::Env env);
    static Napi::Object wrap(Napi::Env env, std::shared_ptr<ov::genai::Phi4ReasoningParser> parser);

    void parse(const Napi::CallbackInfo& info);

    std::shared_ptr<ov::genai::Phi4ReasoningParser> get_parser();
    void set_parser(std::shared_ptr<ov::genai::Phi4ReasoningParser> parser);

private:
    std::shared_ptr<ov::genai::Phi4ReasoningParser> _parser;
};

class Llama3PythonicToolParserWrapper : public Napi::ObjectWrap<Llama3PythonicToolParserWrapper> {
public:
    Llama3PythonicToolParserWrapper(const Napi::CallbackInfo& info);

    static Napi::Function get_class(Napi::Env env);
    static Napi::Object wrap(Napi::Env env, std::shared_ptr<ov::genai::Llama3PythonicToolParser> parser);

    void parse(const Napi::CallbackInfo& info);

    std::shared_ptr<ov::genai::Llama3PythonicToolParser> get_parser();
    void set_parser(std::shared_ptr<ov::genai::Llama3PythonicToolParser> parser);

private:
    std::shared_ptr<ov::genai::Llama3PythonicToolParser> _parser;
};

class Llama3JsonToolParserWrapper : public Napi::ObjectWrap<Llama3JsonToolParserWrapper> {
public:
    Llama3JsonToolParserWrapper(const Napi::CallbackInfo& info);

    static Napi::Function get_class(Napi::Env env);
    static Napi::Object wrap(Napi::Env env, std::shared_ptr<ov::genai::Llama3JsonToolParser> parser);

    void parse(const Napi::CallbackInfo& info);

    std::shared_ptr<ov::genai::Llama3JsonToolParser> get_parser();
    void set_parser(std::shared_ptr<ov::genai::Llama3JsonToolParser> parser);

private:
    std::shared_ptr<ov::genai::Llama3JsonToolParser> _parser;
};
