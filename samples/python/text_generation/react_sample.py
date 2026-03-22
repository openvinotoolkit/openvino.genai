#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import requests
import argparse
import openvino_genai
import urllib.parse
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
        "description_for_model": "Get the current weather in a given city name.",
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
        "name_for_human": "generate image",
        "name_for_model": "generate_image",
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
            if (len(chat_history) == 1):
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

def parse_first_tool_call(text):
    tool_name, tool_args = "", ""
    i = text.find("\nAction:")
    j = text.find("\nAction Input:")
    k = text.find("\nObservation:")
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is omitted by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + "\nObservation:"  # Add it back.
        k = text.find("\nObservation:")
        tool_name = text[i + len("\nAction:") : j].strip()
        tool_args = text[j + len("\nAction Input:") : k].strip()
        text = text[:k]
    return tool_name, tool_args, text

def _tool_error(tool_name: str, message: str) -> str:
    return json.dumps({"tool": tool_name, "error": message})

def _strip_wrapping_parentheses(text: str) -> str:
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == "(" and stripped[-1] == ")":
        return stripped[1:-1].strip()
    return stripped

def _parse_tool_args(tool_name: str, tool_args: str):
    try:
        parsed_args = json5.loads(tool_args)
    except Exception as ex:
        return None, _tool_error(tool_name, f"Failed to parse tool arguments: {ex}")

    if not isinstance(parsed_args, dict):
        return None, _tool_error(tool_name, "Tool arguments must be a JSON object")

    return parsed_args, None

def call_tool(tool_name: str, tool_args: str) -> str:
    if tool_name == "get_weather":
        parsed_args, parse_error = _parse_tool_args(tool_name, tool_args)
        if parse_error:
            return parse_error

        city_name = parsed_args.get("city_name")
        if not isinstance(city_name, str) or not city_name.strip():
            return _tool_error(tool_name, "Missing or invalid required argument: city_name")

        key_selection = {
            "current_condition": [
                "temp_C",
                "FeelsLikeC",
                "humidity",
                "weatherDesc",
                "observation_time",
            ],
        }

        try:
            response = requests.get(f"https://wttr.in/{city_name}?format=j1", timeout=10)
            response.raise_for_status()
        except requests.RequestException as ex:
            return _tool_error(tool_name, f"Weather request failed: {ex}")

        try:
            response_payload = response.json()
        except ValueError as ex:
            return _tool_error(tool_name, f"Weather response is not valid JSON: {ex}")

        current_condition = response_payload.get("current_condition")
        if not isinstance(current_condition, list) or not current_condition or not isinstance(current_condition[0], dict):
            return _tool_error(tool_name, "Weather response has unexpected structure")

        weather = current_condition[0]
        ret = {
            k: {_v: weather.get(_v, "") for _v in v}
            for k, v in key_selection.items()
        }
        return json.dumps(ret, ensure_ascii=False)
    elif tool_name == "generate_image":
        tool_args = _strip_wrapping_parentheses(tool_args)
        parsed_args, parse_error = _parse_tool_args(tool_name, tool_args)
        if parse_error:
            return parse_error

        prompt = parsed_args.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return _tool_error(tool_name, "Missing or invalid required argument: prompt")

        prompt = urllib.parse.quote(prompt)
        return json.dumps(
            {"image_url": f"https://image.pollinations.ai/prompt/{prompt}"},
            ensure_ascii=False,
        )
    else:
        return _tool_error(tool_name, "Unsupported tool")

def llm_with_tool(llm_pipe, prompt, history, list_of_tool_info, max_steps=8):
    chat_history = [(x["user"], x["bot"]) for x in history] + [(prompt, "")]
    planning_prompt = build_input_text(llm_pipe.get_tokenizer(), chat_history, list_of_tool_info)

    text = ""
    for _ in range(max_steps):
        # llm pipe output based planning_prompt and the text (previous output)
        llm_config = llm_pipe.get_generation_config()
        try:
            output = llm_pipe.generate(planning_prompt + text, llm_config, streamer)
        except Exception as ex:
            output = f"\nObservation: = {_tool_error('llm_generate', f'LLM generation failed: {ex}')}\nThought:"
            text += output
            break

        # parse the output to get action
        action, action_input, output = parse_first_tool_call(output)
        if action:
            try:
                observation = call_tool(action, action_input)
            except Exception as ex:
                observation = _tool_error(action, f"Unexpected tool execution failure: {ex}")

            observation_txt = f"\nObservation: = {observation}\nThought:"
            print("\n\n- Getting information from the tool API -", observation_txt, "\n")
            output += observation_txt
            text += output
        else:
            text += output
            break
    else:
        text += f"\nObservation: = {_tool_error('llm_with_tool', 'Reached max tool-calling steps')}\n"

    history.append({"user": prompt, "bot": text})
    return text, history

def streamer(subword):
    print(subword, end='', flush=True)
    # Return flag corresponds whether generation should be stopped.
    return openvino_genai.StreamingStatus.RUNNING

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    args = parser.parse_args()

    device = 'CPU'  # GPU can be used as well
    llm_model_path = args.model_dir

    llm_pipe = openvino_genai.LLMPipeline(llm_model_path, device)
    llm_config = openvino_genai.GenerationConfig()
    llm_config.max_new_tokens = 256
    llm_pipe.set_generation_config(llm_config)

    history = []
    query = "get the weather in London, and create a picture of Big Ben based on the weather information"
    response, history = llm_with_tool(llm_pipe, prompt=query, history=history, list_of_tool_info=tools)

if '__main__' == __name__:
    main()
