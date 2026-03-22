#!/usr/bin/env python3
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

import openvino_genai
from openvino_genai import GenerationConfig, LLMPipeline, StructuredOutputConfig, ChatHistory

from openai_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    FunctionCall,
    ResponseMessage,
    ToolCall,
    Usage,
)

app = FastAPI(title="OpenVINO GenAI OpenAI-Compatible Server")

pipe = None
pipeline_lock = asyncio.Lock()


def build_tool_call_schema(tools):
    """Translate OpenAI tool definitions into a JSON Schema for StructuredOutputConfig."""
    any_of = []
    for tool in tools:
        func = tool.function
        func_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "const": func.name},
                "arguments": func.parameters if func.parameters else {"type": "object"},
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
        any_of.append(func_schema)

    return json.dumps({"anyOf": any_of})


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model pipeline not initialized")

    history = ChatHistory()
    for msg in request.messages:
        history.append({"role": msg.role, "content": msg.content or ""})

    config = GenerationConfig()
    config.max_new_tokens = request.max_tokens if request.max_tokens else 512

    if request.temperature is not None and request.temperature > 0.0:
        config.temperature = request.temperature
        config.do_sample = True
    else:
        config.do_sample = False

    if request.top_p is not None:
        config.top_p = request.top_p

    if request.tools:
        config.structured_output_config = StructuredOutputConfig(
            json_schema=build_tool_call_schema(request.tools)
        )

    req_id = f"chatcmpl-{uuid.uuid4().hex}"

    if request.stream:
        return StreamingResponse(
            stream_generate(req_id, request.model, history, config, request.tools),
            media_type="text/event-stream",
        )

    async with pipeline_lock:
        results = await asyncio.to_thread(pipe.generate, history, config)
    text = results.texts[0]

    message = build_response_message(text, request.tools)
    return ChatCompletionResponse(
        id=req_id,
        created=int(time.time()),
        model=request.model,
        choices=[Choice(index=0, message=message, finish_reason="stop")],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )


def build_response_message(text, tools):
    """Parse generated text into a ResponseMessage, detecting tool calls when applicable."""
    if tools:
        try:
            tool_output = json.loads(text)
            if isinstance(tool_output, dict) and "name" in tool_output and "arguments" in tool_output:
                return ResponseMessage(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            function=FunctionCall(
                                name=tool_output["name"],
                                arguments=json.dumps(tool_output["arguments"]),
                            ),
                        )
                    ],
                )
        except (json.JSONDecodeError, KeyError):
            pass

    return ResponseMessage(content=text)


async def stream_generate(req_id, model, history, config, tools):
    """Yield Server-Sent Events for streaming generation."""
    collected = []

    def streamer(subword):
        collected.append(subword)
        return openvino_genai.StreamingStatus.RUNNING

    async with pipeline_lock:
        await asyncio.to_thread(pipe.generate, history, config, streamer)

    # Emit all collected tokens as SSE chunks.
    for token in collected:
        resp = ChatCompletionResponse(
            id=req_id,
            object="chat.completion.chunk",
            model=model,
            created=int(time.time()),
            choices=[Choice(
                index=0,
                delta=ResponseMessage(content=token),
                finish_reason=None,
            )],
        )
        yield f"data: {resp.model_dump_json(exclude_none=True)}\n\n"

    # If tools were requested, re-emit the assembled result as a tool_calls chunk.
    if tools:
        full_text = "".join(collected)
        message = build_response_message(full_text, tools)
        if message.tool_calls:
            resp = ChatCompletionResponse(
                id=req_id,
                object="chat.completion.chunk",
                model=model,
                created=int(time.time()),
                choices=[Choice(
                    index=0,
                    delta=ResponseMessage(content=None, tool_calls=message.tool_calls),
                    finish_reason="stop",
                )],
            )
            yield f"data: {resp.model_dump_json(exclude_none=True)}\n\n"

    final = ChatCompletionResponse(
        id=req_id,
        object="chat.completion.chunk",
        model=model,
        created=int(time.time()),
        choices=[Choice(index=0, delta=ResponseMessage(), finish_reason="stop")],
    )
    yield f"data: {final.model_dump_json(exclude_none=True)}\n\n"
    yield "data: [DONE]\n\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to the model directory")
    parser.add_argument("--device", default="CPU", help="Device to run the model on (default: CPU)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    args = parser.parse_args()

    global pipe
    print(f"Loading model from {args.model_dir} on {args.device}...")
    pipe = LLMPipeline(args.model_dir, args.device)
    print("Model loaded. Starting server...")

    uvicorn.run(app, host=args.host, port=args.port)


if "__main__" == __name__:
    main()
