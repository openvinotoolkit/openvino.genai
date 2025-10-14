#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai


def streamer(subword):
    print(subword, end='', flush=True)
    # Return flag corresponds whether generation should be stopped.
    return openvino_genai.StreamingStatus.RUNNING

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', help='Path to the model directory')
    parser.add_argument('device', nargs='?', default='CPU', help='Device to run the model on (default: CPU)')
    args = parser.parse_args()

    device = args.device
    pipe = openvino_genai.LLMPipeline(args.model_dir, device)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100

    pipe.start_chat()
    while True:
        try:
            prompt = input('question:\n')
        except EOFError:
            break
        pipe.generate(prompt, config, streamer)
        print('\n----------')
    pipe.finish_chat()


if '__main__' == __name__:
    main()

    pipe = openvino_genai.LLMPipeline(args.model_dir, device)

    prompt = "What is the weather in New York today?"
    res = pipe.generate(prompt, max_new_tokens=100, streamer=streamer)
    print(res.texts[0])

    res.parsed['tool_caling']

    class LlamaToolCallParser(ParserBase):
        def parse(self, parsed_data: ParsedData) -> ParsedData:
            # parsed_data 
            # process parsed_data 
            # e.g. extract tool calls, or other fields from content
            return new_parsed_output

    llama_parser = LlamaToolCallParser()
    res = pipe.generate(prompt, parsers=[llama_parser | "LLama3.2Pythonic"], max_new_tokens=100)

# At the beginning msg['original_content'] is filled with full text
msg = res.texts[i]
for parser in m_parsers:
    msg = parser.parse(msg)

# At the end msg is filled with all parsed fields
parsed_data = {
    'original_content': '<|system|>You are a helpful assistant... I will call the `get_weather` function with the location… \n\nfunctools[{"name": "get_weather", "arguments": {"location": "New York", "unit": "celsius"}}]<|end|>',
    'content': 'blah blah', 
    'reasoning_content': '', 
    'tool_calls': "[{\"name\":\"get_weather\",\"arguments\":{\"location\":\"New York, NY\",\"unit\":\"celsius\"}}]",
}

res.parsed: ParsedData