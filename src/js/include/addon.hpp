// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <napi.h>

typedef Napi::Function (*Prototype)(Napi::Env);

struct AddonData {
    Napi::FunctionReference core;
    Napi::FunctionReference tokenizer;
    Napi::FunctionReference perf_metrics;
};

void init_class(Napi::Env env,
                Napi::Object exports,
                std::string class_name,
                Prototype func,
                Napi::FunctionReference& reference);

Napi::Object init_module(Napi::Env env, Napi::Object exports);
