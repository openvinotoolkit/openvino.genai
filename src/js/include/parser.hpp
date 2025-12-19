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

private:
    Napi::ThreadSafeFunction parser_tsfn;
    Napi::ObjectReference js_parser_ref;
};

class ReasoningParserWrapper : public Napi::ObjectWrap<ReasoningParserWrapper> {
public:
    ReasoningParserWrapper(const Napi::CallbackInfo& info);

    static Napi::Function get_class(Napi::Env env);

    void parse(const Napi::CallbackInfo& info);

    std::shared_ptr<ov::genai::ReasoningParser> get_parser();

private:
    std::shared_ptr<ov::genai::ReasoningParser> _parser;
};
