#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import requests
import argparse
import openvino_genai
import urllib.parse
import encrypted_model_causal_lm
import json
import json5

TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

PROMPT_REACT = """Answer the following questions as best as you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}"""


tools = [
    {
        "name_for_human": "get weather",
        "name_for_model": "get_weather",
        "description_for_model": 'Get the current weather in a given city name."',
        "parameters": [
            {
                "name": "city_name",
                "description": "City name",
                "required": True,
                "schema": {"type": "string"},
            }
        ],
    },
    {
        "name_for_human": "image generation",
        "name_for_model": "image_gen",
        "description_for_model": "AI painting (image generation) service, input text description, and return the image URL drawn based on text information.",
        "parameters": [
            {
                "name": "prompt",
                "description": "describe the image",
                "required": True,
                "schema": {"type": "string"},
            }
        ],
    },
]

def build_input_text(tokenizer, chat_history, list_of_tool_info) -> str:
    tools_text = []
    for tool_info in list_of_tool_info:
        tool = TOOL_DESC.format(
            name_for_model=tool_info["name_for_model"],
            name_for_human=tool_info["name_for_human"],
            description_for_model=tool_info["description_for_model"],
            parameters=json.dumps(tool_info["parameters"], ensure_ascii=False),
        )
        if tool_info.get("args_format", "json") == "json":
            tool += " Format the arguments as a JSON object."
        elif tool_info["args_format"] == "code":
            tool += " Enclose the code within triple backticks (`) at the beginning and end of the code."
        else:
            raise NotImplementedError
        tools_text.append(tool)

    tools_text = "\n\n".join(tools_text)
    tools_name_text = ", ".join([tool_info["name_for_model"] for tool_info in list_of_tool_info])

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i, (query, response) in enumerate(chat_history):
        if list_of_tool_info:
            if (len(chat_history) == 1) or (i == len(chat_history) - 2):
                query = PROMPT_REACT.format(
                    tools_text=tools_text,
                    tools_name_text=tools_name_text,
                    query=query,
                )
        if query:
            messages.append({"role": "user", "content": query})
        if response:
            messages.append({"role": "assistant", "content": response})

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    return prompt


def parse_latest_tool_call(text):
    tool_name, tool_args = "", ""
    i = text.rfind("\nAction:")
    j = text.rfind("\nAction Input:")
    k = text.rfind("\nObservation:")
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + "\nObservation:"  # Add it back.
        k = text.rfind("\nObservation:")
        tool_name = text[i + len("\nAction:") : j].strip()
        tool_args = text[j + len("\nAction Input:") : k].strip()
        text = text[:k]
    return tool_name, tool_args, text


def call_tool(tool_name: str, tool_args: str) -> str:
    if tool_name == "get_weather":
        try:
            city_name = json5.loads(tool_args)["city_name"]
        except KeyError:
            print("Error: 'city_name' key not found in tool_args")  # Debug print
            raise
        key_selection = {
            "current_condition": [
                "temp_C",
                "FeelsLikeC",
                "humidity",
                "weatherDesc",
                "observation_time",
            ],
        }
        resp = requests.get(f"https://wttr.in/{city_name}?format=j1")
        resp.raise_for_status()
        resp = resp.json()
        ret = {k: {_v: resp[k][0][_v] for _v in v} for k, v in key_selection.items()}
        return str(ret)
    elif tool_name == "image_gen":
        tool_args = tool_args.replace("(", "").replace(")", "")
        prompt = json5.loads(tool_args)["prompt"]
        prompt = urllib.parse.quote(prompt)
        return json.dumps(
            {"image_url": f"https://image.pollinations.ai/prompt/{prompt}"},
            ensure_ascii=False,
        )
    else:
        raise NotImplementedError


def llm_with_tool(llm_pipe, llm_config, tokenizer, prompt: str, history, list_of_tool_info):
    chat_history = [(x["user"], x["bot"]) for x in history] + [(prompt, "")]
    planning_prompt = build_input_text(tokenizer, chat_history, list_of_tool_info)

    text = ""
    while True:
        output = llm_pipe.generate(planning_prompt + text, llm_config, streamer)
        action, action_input, output = parse_latest_tool_call(output)
        if action:
            print("Action:", action)
            print("Action_input:", action_input)
            observation = call_tool(action, action_input)
            output += f"\nObservation: = {observation}\nThought:"
            observation = f"{observation}\nThought:"
            print(observation)
            text += output
        else:
            text += output
            break

    new_history = []
    new_history.extend(history)
    new_history.append({"user": prompt, "bot": text})
    return text, new_history


def streamer(subword):
    print(subword, end='', flush=True)
    # Return flag corresponds whether generation should be stopped.
    return openvino_genai.StreamingStatus.RUNNING

 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    args = parser.parse_args()

    device = 'GPU'
    llm_model_path = args.model_dir

    model, weights = encrypted_model_causal_lm.decrypt_model(llm_model_path, 'openvino_model.xml', 'openvino_model.bin')
    tokenizer = encrypted_model_causal_lm.read_tokenizer(llm_model_path)

    llm_pipe = openvino_genai.LLMPipeline(model, weights, tokenizer, device)
    llm_config = openvino_genai.GenerationConfig()
    llm_config.max_new_tokens = 256
    llm_pipe.start_chat()

    history = []
    query = "get the weather in London, and create a picture of Big Ben based on the weather information"
    response, history = llm_with_tool(llm_pipe, llm_config, tokenizer, prompt=query, history=history, list_of_tool_info=tools)
    llm_pipe.finish_chat()


if '__main__' == __name__:
    main()

