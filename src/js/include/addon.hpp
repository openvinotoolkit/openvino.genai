// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

typedef Napi::Function (*Prototype)(Napi::Env);

struct AddonData {
    Napi::FunctionReference core;
    Napi::FunctionReference vlm_pipeline;
    Napi::FunctionReference text_rerank_pipeline;
    Napi::FunctionReference tokenizer;
    Napi::FunctionReference perf_metrics;
    Napi::FunctionReference vlm_perf_metrics;
    Napi::FunctionReference chat_history;
    Napi::FunctionReference reasoning_parser;
    Napi::FunctionReference deepseek_r1_reasoning_parser;
    Napi::FunctionReference phi4_reasoning_parser;
    Napi::FunctionReference llama3_pythonic_tool_parser;
    Napi::FunctionReference llama3_json_tool_parser;
    Napi::ObjectReference openvino_addon;
};

void init_class(Napi::Env env,
                Napi::Object exports,
                std::string class_name,
                Prototype func,
                Napi::FunctionReference& reference);

Napi::Object init_module(Napi::Env env, Napi::Object exports);
