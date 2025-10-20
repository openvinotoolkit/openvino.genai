// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/parsers.hpp"
#include "nlohmann/json.hpp"
#include "openvino/genai/llm_pipeline.hpp"

using namespace ov::genai;

TEST(ParserTest, test_llama32_parser_1) {
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


    std::shared_ptr<Llama32PythonicToolParser> parser = std::make_shared<Llama32PythonicToolParser>();
    JsonContainer input;
    input["content"] = prompt;
    parser->parse(input);
    
    ASSERT_TRUE(expected == input);
}

TEST(ParserTest, test_llama32_parser_2) {
    std::string prompt = R"(What's the weather in New York today?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n[get_weather(location="New York, NY", unit="celsius")]<|eom_id|>)";
    
    JsonContainer expected;
    expected["content"] = std::string(R"(What's the weather in New York today?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<|eom_id|>)");
    expected["tool_calls"] = JsonContainer::array();
    expected["tool_calls"].push_back(JsonContainer(ov::AnyMap({
        {"name", "get_weather"},
        {"arguments", ov::AnyMap{
            {"location", "New York, NY"},
            {"unit", "celsius"}
        }}
    })));

    std::shared_ptr<Llama32PythonicToolParser> parser = std::make_shared<Llama32PythonicToolParser>(/*keep_original_content*/ false);
    JsonContainer input;
    input["content"] = prompt;
    parser->parse(input);

    ASSERT_EQ(input, expected);
}

TEST(ParserTest, test_reasoning_parser_1) {
    std::string prompt = R"("<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜User｜>What is 2 + 1?<｜Assistant｜><think>\nI need to determine the sum of 2 and 1.\n\nFirst, I'll identify the two numbers involved in the addition: 2 and 1.\n\nNext, I'll perform the addition by combining these two numbers.\n\nFinally, I'll state the result of the addition, which is 3.\n</think>\n\n**Solution:**\n\nTo find the sum of 2 and 1, )";
    
    JsonContainer expected;
    expected["content"] = std::string(R"("<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜><｜User｜>What is 2 + 1?<｜Assistant｜>\n\n**Solution:**\n\nTo find the sum of 2 and 1, )");
    expected["reasoning_content"] = std::string(R"(\nI need to determine the sum of 2 and 1.\n\nFirst, I'll identify the two numbers involved in the addition: 2 and 1.\n\nNext, I'll perform the addition by combining these two numbers.\n\nFinally, I'll state the result of the addition, which is 3.\n)");

    std::shared_ptr<BaseReasoningParser> parser = std::make_shared<BaseReasoningParser>(
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
    expected["reasoning_content"] = std::string(R"(\nI need to determine the sum of 2 and 1.\n\nFirst, I'll identify the two numbers involved in the addition: 2 and 1.\n\nNext, I'll perform the addition by combining these two numbers.\n\nFinally, I'll state the result of the addition, which is 3.\n)");

    std::shared_ptr<BaseReasoningParser> parser = std::make_shared<BaseReasoningParser>(
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
    ov::genai::DeepSeekR1ReasoningParser parser;
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
        std::string previous_text = input_stream[i - 1];
        std::string delta_text = input_stream[i];
        delta_text = parser.parse(msg, previous_text, delta_text);
    }
    ASSERT_EQ(msg["reasoning_content"], ref_res);
}

// TODO: add tests when streamer is called directly instead of manual subsequent calling of parsers.
