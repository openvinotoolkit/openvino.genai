// Copyright (C) 2024-2026 Intel Corporation
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

namespace {

JsonContainer parse_qwen3_coder(const std::string& content, const JsonContainer& tools = JsonContainer::array()) {
    JsonContainer message;
    message["content"] = content;
    Qwen3CoderToolParser(tools).parse(message);
    return message;
}

JsonContainer weather_tools() {
    return JsonContainer::from_json_string(
        R"([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
        "location": {"type": "string"}, "days": {"type": "integer"}, "detailed": {"type": "boolean"},
        "units": {"type": ["string", "null"]}, "options": {"type": "object"}}}}}])");
}

}  // namespace

TEST(ParserTest, test_qwen3_coder_tool_parser_single_call) {
    const std::string content =
        "I'll check.\n<tool_call>\n<function=get_weather>\n<parameter=location>\nNew York, NY\n</parameter>\n"
        "<parameter=unit>\ncelsius\n</parameter>\n</function>\n</tool_call>";
    JsonContainer message = parse_qwen3_coder(content);

    JsonContainer expected;
    expected["content"] = content;  // content is kept as is
    expected["tool_calls"] = JsonContainer::array();
    expected["tool_calls"].push_back(JsonContainer(
        {{"name", "get_weather"}, {"arguments", JsonContainer{{"location", "New York, NY"}, {"unit", "celsius"}}}}));
    ASSERT_EQ(message, expected);
}

TEST(ParserTest, test_qwen3_coder_tool_parser_several_calls_and_multiline_value) {
    const std::string content =
        "<tool_call>\n<function=write>\n<parameter=text>\nline 1\nline 2\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>";
    JsonContainer calls = parse_qwen3_coder(content)["tool_calls"];
    ASSERT_EQ(calls.size(), 2);
    ASSERT_EQ(calls[0]["name"].get_string(), "write");
    ASSERT_EQ(calls[0]["arguments"]["text"].get_string(), "line 1\nline 2");
    ASSERT_EQ(calls[1]["name"].get_string(), "get_weather");
    ASSERT_EQ(calls[1]["arguments"]["location"].get_string(), "Paris");
}

TEST(ParserTest, test_qwen3_coder_tool_parser_types_from_tools) {
    const std::string content =
        "<tool_call>\n<function=get_weather>\n<parameter=location>\n007\n</parameter>\n<parameter=days>\n3\n</"
        "parameter>\n"
        "<parameter=detailed>\ntrue\n</parameter>\n<parameter=units>\n42\n</parameter>\n"
        "<parameter=options>\n{\"lang\": "
        "\"en\"}\n</parameter>\n<parameter=unknown>\n5\n</parameter>\n</function>\n</tool_call>";
    JsonContainer args = parse_qwen3_coder(content, weather_tools())["tool_calls"][0]["arguments"];
    ASSERT_EQ(args["location"], JsonContainer("007"));  // declared string: never coerced
    ASSERT_EQ(args["days"], JsonContainer(int64_t{3}));
    ASSERT_EQ(args["detailed"], JsonContainer(true));
    ASSERT_EQ(args["units"], JsonContainer("42"));  // ["string", "null"] counts as string
    ASSERT_EQ(args["options"], JsonContainer::from_json_string(R"({"lang": "en"})"));
    ASSERT_EQ(args["unknown"], JsonContainer("5"));  // not in the schema: string

    // Without tools every value is a string; a value that is not valid JSON stays a string.
    ASSERT_EQ(parse_qwen3_coder(content)["tool_calls"][0]["arguments"]["days"], JsonContainer("3"));
    const std::string bad = "<tool_call><function=get_weather><parameter=days>three</parameter></function></tool_call>";
    ASSERT_EQ(parse_qwen3_coder(bad, weather_tools())["tool_calls"][0]["arguments"]["days"], JsonContainer("three"));
}

TEST(ParserTest, test_qwen3_coder_tool_parser_truncated_and_malformed) {
    // Cut off by max_new_tokens inside a value: what arrived is kept.
    JsonContainer cut = parse_qwen3_coder("<tool_call>\n<function=get_weather>\n<parameter=location>\nNew Yo");
    ASSERT_EQ(cut["tool_calls"][0]["arguments"]["location"].get_string(), "New Yo");
    // A missing </parameter> ends the value at the next parameter.
    JsonContainer skipped =
        parse_qwen3_coder("<tool_call><function=f><parameter=a>1<parameter=b>2</parameter></function></tool_call>");
    ASSERT_EQ(skipped["tool_calls"][0]["arguments"], JsonContainer({{"a", "1"}, {"b", "2"}}));
    // No function, empty name, no parameters, no markup at all.
    ASSERT_FALSE(parse_qwen3_coder("<tool_call>{\"name\": \"f\"}</tool_call>").contains("tool_calls"));
    ASSERT_FALSE(parse_qwen3_coder("<tool_call><function=></function></tool_call>").contains("tool_calls"));
    ASSERT_EQ(parse_qwen3_coder("<tool_call><function=ping></function></tool_call>")["tool_calls"][0]["arguments"],
              JsonContainer::object());
    ASSERT_FALSE(parse_qwen3_coder("Just an answer, mentioning <tool_call> in prose").contains("tool_calls"));
    // Garbage tool definitions are ignored rather than thrown on.
    JsonContainer junk = JsonContainer::from_json_string(
        R"([null, "x", {"type": "function"}, {"function": {"name": "f", "parameters": null}}])");
    ASSERT_EQ(parse_qwen3_coder("<tool_call><function=f><parameter=a>1</parameter></function></tool_call>",
                                junk)["tool_calls"][0]["arguments"]["a"],
              JsonContainer("1"));
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
    JsonContainer accumulated_msg;
    for (int i = 1; i < input_stream.size(); i++) {
        std::string delta_text = input_stream[i];
        delta_text = parser.parse(msg, delta_text);
        accumulated_msg.concatenate(msg);
    }
    ASSERT_EQ(accumulated_msg["reasoning_content"], ref_res);
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
