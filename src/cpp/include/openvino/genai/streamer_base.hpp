// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/tokenizer.hpp"

namespace ov {

/** 
 * @brief base class for streamers. In order to use inherit from from this class and inplement put, and methods
 * 
 * @param m_tokenizer tokenizer
*/
class StreamerBase {
public:
    Tokenizer m_tokenizer;
    StreamerBase(Tokenizer tokenizer): m_tokenizer(tokenizer) {};
    StreamerBase() = default;
    
    /// @brief put is called every time new token is decoded
    virtual void put(int64_t token) = 0;
    
    /// @brief end is called at the end of generation. It can be used to flush cache if your own streamer has one
    virtual void end() = 0;
};

} // namespace ov
