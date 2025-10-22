// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/parsers.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "openvino/genai/llm_pipeline.hpp"

using namespace ov::genai;

TEST(ParserTest, test_llama3_parser_1) {
    std::string prompt = R"(What's the weather in New York today?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n[get_weather(location="New York, NY", unit="celsius")]<|eom_id|>)";
    // By default content should keep original values.
    
    JsonContainer expected;
    expected["content"] = prompt;
    expected["tool_calls"] = JsonContainer::array();
    expected["tool_calls"].push_back(JsonContainer({
        {"name", "get_weather"},
        {"arguments", JsonContainer{
            {"location", "New York, NY"},
            {"unit", "celsius"}
        }}
    }));


    std::shared_ptr<Llama3PythonicToolParser> parser = std::make_shared<Llama3PythonicToolParser>();
    JsonContainer input;
    input["content"] = prompt;
    parser->parse(input);
    
    ASSERT_TRUE(expected == input);
}

TEST(ParserTest, test_reasoning_parser_1) {
    std::string prompt = R"("<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜User｜>What is 2 + 1?<｜Assistant｜><think>\nI need to determine the sum of 2 and 1.\n\nFirst, I'll identify the two numbers involved in the addition: 2 and 1.\n\nNext, I'll perform the addition by combining these two numbers.\n\nFinally, I'll state the result of the addition, which is 3.\n</think>\n\n**Solution:**\n\nTo find the sum of 2 and 1, )";
    
    JsonContainer expected;
    expected["content"] = R"("<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜User｜>What is 2 + 1?<｜Assistant｜>\n\n**Solution:**\n\nTo find the sum of 2 and 1, )";
    expected["reasoning_content"] = R"(\nI need to determine the sum of 2 and 1.\n\nFirst, I'll identify the two numbers involved in the addition: 2 and 1.\n\nNext, I'll perform the addition by combining these two numbers.\n\nFinally, I'll state the result of the addition, which is 3.\n)";

    std::shared_ptr<ReasoningParser> parser = std::make_shared<ReasoningParser>(
        /*expect_open_tag*/ true,
        /*keep_original_content*/ false
    );
    JsonContainer input;
    input["content"] = prompt;
    parser->parse(input);

    ASSERT_EQ(input, expected);
}

TEST(ParserTest, test_reasoning_parser_2) {
    std::string prompt = R"("<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜User｜>What is 2 + 1?<｜Assistant｜><think>\nI need to determine the sum of 2 and 1.\n\nFirst, I'll identify the two numbers involved in the addition: 2 and 1.\n\nNext, I'll perform the addition by combining these two numbers.\n\nFinally, I'll state the result of the addition, which is 3.\n</think>\n\n**Solution:**\n\nTo find the sum of 2 and 1, )";
    
    JsonContainer expected;
    expected["content"] = prompt;
    expected["reasoning_content"] = R"(\nI need to determine the sum of 2 and 1.\n\nFirst, I'll identify the two numbers involved in the addition: 2 and 1.\n\nNext, I'll perform the addition by combining these two numbers.\n\nFinally, I'll state the result of the addition, which is 3.\n)";

    std::shared_ptr<ReasoningParser> parser = std::make_shared<ReasoningParser>(
        /*expect_open_tag*/ true,
        /*keep_original_content*/ true
    );
    JsonContainer input;
    input["content"] = prompt;
    parser->parse(input);

    ASSERT_EQ(input, expected);
}



class DeepSeekR1ReasoningParserTest : public ::testing::Test {
protected:
    ov::genai::DeepSeekR1ReasoningIncrementalParser parser;
    JsonContainer msg;
};

TEST_F(DeepSeekR1ReasoningParserTest, ReasoningContentAccumulatesAcrossCalls) {
    std::vector<std::string> input_stream = {
        "<｜begin▁of▁sentence｜>", "First", ",", " I", " recognize", " that", " the", " question", " is", " asking", 
        " for", " the", " sum", " of", " ", "2", " and", " ", "1", ".\n\n", "I", " know", " that", " addition", 
        " involves", " combining", " two", " numbers", " to", " find", " their", " total", ".\n\n", "Starting", 
        " with", " ", "2", ",", " I", " add", " ", "1", " to", " it", ".\n\n", "2", " plus", " ", "1", " equals", 
        " ", "3", ".\n", "</think>", "\n\n", "**", "Solution", ":", "**\n\n", "To", " find", " the", " sum", 
        " of", " ", "2", " and", " ", "1", " follow", " these", " simple", " steps", ":\n\n", "1", ".", " **", 
        "Start", " with", " the", " number", " ", "2", ".", "**\n", "2", ".", " **", "Add", " ", "1", " to", 
        " it", ".", "**\n", "   \n", "  ", " \\", "[\n", "  "
    };
    
    std::string ref_res = "First, I recognize that the question is asking for the sum of 2 and 1.\n\nI know that addition involves combining two numbers to find their total.\n\nStarting with 2, I add 1 to it.\n\n2 plus 1 equals 3.\n";
    
    JsonContainer msg;
    
    for (int i = 1; i < input_stream.size(); i++) {
        std::string delta_text = input_stream[i];
        delta_text = parser.parse(msg, delta_text);
    }
    ASSERT_EQ(msg["reasoning_content"], ref_res);
}

TEST(ParserTest, test_custom_parser) {
    // Define a small custom parser derived from Parser
    class CustomParser : public ov::genai::Parser {
    public:
        void parse(ov::genai::JsonContainer& msg) override {
            // extract "content"
            if (!msg.contains("content"))
                return;

            auto content_opt = msg["content"].as_string();
            if (!content_opt.has_value())
                return;

            const std::string& content = content_opt.value();

            // find text between <think> and </think>
            std::size_t start = content.find("<think>");
            std::size_t end   = content.find("</think>");
            if (start != std::string::npos && end != std::string::npos && end > start) {
                std::string think_text = content.substr(start + 7, end - (start + 7));
                // trim leading/trailing whitespace
                auto l = think_text.find_first_not_of(" \n\r\t");
                auto r = think_text.find_last_not_of(" \n\r\t");
                if (l != std::string::npos && r != std::string::npos)
                    think_text = think_text.substr(l, r - l + 1);
                msg["reasoning_content"] = think_text;
            }
        }
    };

    CustomParser parser;

    ov::genai::JsonContainer msg;
    msg["content"] = "<think>This is reasoning.</think> And this is the answer";

    parser.parse(msg);

    ASSERT_TRUE(msg.contains("reasoning_content"));
    ASSERT_EQ(msg["reasoning_content"].get_string(), "This is reasoning.");
}

TEST(ParserTest, CustomParser_AccumulatesBetweenStartStop) {
    using namespace ov::genai;

    // Custom incremental parser: mirrors the Python logic
    class CustomParser : public IncrementalParser {
    public:
        bool main_part_started = false;

        std::string parse(JsonContainer& msg,
                          std::string& delta_text,
                          const std::optional<std::vector<int64_t>>& /*delta_tokens*/ = std::nullopt) override {
            // Ensure fields exist (Python test used dict defaults)
            if (!msg.contains("content")) {
                msg.to_empty_object();
                msg["content"] = "";
            }
            if (!msg.contains("reasoning_content")) {
                msg["reasoning_content"] = "";
            }

            if (!main_part_started && delta_text == "<think>") {
                main_part_started = true;
            } else if (main_part_started && delta_text == "</think>") {
                main_part_started = false;
            } else {
                if (main_part_started) {
                    // Append delta into reasoning_content
                    auto cur = msg["reasoning_content"].as_string().value_or("");
                    cur += delta_text;
                    msg["reasoning_content"] = cur;
                }
            }
            // Return delta_text (same as Python)
            return delta_text;
        }

        void reset() override {
            main_part_started = false;
        }

        // Virtual dtor for safety
        ~CustomParser() override = default;
    };

    class CustomStreamer : public ov::genai::TextParserStreamer {
    public:
        using TextParserStreamer::write;
        // Forwarding constructor to base class
        CustomStreamer(ov::genai::Tokenizer& tok, const std::vector<std::shared_ptr<IncrementalParser>>& parsers)
            : ov::genai::TextParserStreamer(tok, parsers) {}

        JsonContainer final_msg;
        StreamingStatus write(JsonContainer& message) override {
            final_msg = message;
            return StreamingStatus::RUNNING;
        }
    };

    Tokenizer tok;
    std::shared_ptr<IncrementalParser> parser = std::make_shared<CustomParser>();
    CustomStreamer streamer(tok, {parser});
    
    
    // Same stream as in the Python example
    std::vector<std::string> stream_string = {"<think>", " ", "world", " ", "</think>", "!"};

    for (size_t i = 0; i < stream_string.size(); ++i) {
        streamer.write(stream_string[i]);
    }
    
    JsonContainer msg = streamer.get_parsed_message();
    ASSERT_TRUE(msg.contains("reasoning_content"));
    ASSERT_EQ(msg["reasoning_content"].get_string(), " world ");
}
