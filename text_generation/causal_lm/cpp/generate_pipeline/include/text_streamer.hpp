// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "llm_tokenizer.hpp"

class StreamerBase {
public:
    virtual void put(int64_t token) = 0;

    virtual void end() = 0;
};

class TextCallbackStreamer: public StreamerBase {
    Tokenizer tokenizer;
    std::vector<int64_t> token_cache;
    size_t print_len = 0;
    std::function<void (std::string)> m_callback = [](std::string words){ ;};
    
public:
    TextCallbackStreamer() = default;
    TextCallbackStreamer(const Tokenizer& tokenizer);
    TextCallbackStreamer(const Tokenizer& tokenizer, std::function<void (std::string)> callback);
    void set_tokenizer(Tokenizer tokenizer);
    void set_callback(std::function<void (std::string)> callback);
    
    void put(int64_t token) override;
    void end() override;  
};
