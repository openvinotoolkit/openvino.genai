// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "openvino/genai/text_streamer.hpp"

namespace ov {
namespace genai {

struct DeltaToolCall; // Forward declaration, define as needed

struct DeltaMessage {
    std::optional<std::string> role;
    std::optional<std::string> content;
    std::optional<std::string> reasoning_content;
    // std::vector<DeltaToolCall> tool_calls;

    DeltaMessage()
        : role(std::nullopt),
          content(std::nullopt),
          reasoning_content(std::nullopt) {}
};

class TextParserStreamer : public ov::genai::TextStreamer {
public:
    TextParserStreamer(const Tokenizer& tokenizer);

    StreamingStatus write(const DeltaMessage& message);

    ov::genai::CallbackTypeVariant write(std::string message);
};

class ReasoningParserBase {
public:
    ReasoningParserBase() = default;

    void parse(const std::string& text);
};

class ToolCallingParserBase {
public:
    ToolCallingParserBase() = default;

    void parse(const std::string& text);
};

}  // namespace genai
}  // namespace ov
