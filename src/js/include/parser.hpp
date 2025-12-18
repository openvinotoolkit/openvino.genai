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
