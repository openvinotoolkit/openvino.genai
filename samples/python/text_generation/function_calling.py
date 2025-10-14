#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Function Calling Sample
=======================

This sample demonstrates how to build a very small "agent loop" using
OpenVINO GenAI that:

1. Accepts a natural language user request.
2. Lets an LLM decide whether one (or multiple) functions ("tools") need to be
   called by emitting structured function call blocks.
3. Parses & executes these function calls locally (mock implementations here).
4. Feeds the tool results back to the model to obtain a final, grounded answer.

It uses structural tags to strictly constrain the format of function calls:

    <function="function_name">{"arg": "value", ...}</function>

Two example functions are provided:
  * get_weather(city, country, date)
  * get_currency_exchange(from_currency, to_currency, amount)

You can run with or without structural tags (fallback to free-form, may be less
reliable) to compare behaviors.

Example:
    python function_calling.py model_dir --prompt "What was the weather in Berlin yesterday and how many USD can I get for 50 EUR?"

NOTE:
  * For best results, use an instruction / chat-tuned model that is capable of
    function/tool calling (e.g. Llama / Phi / Qwen instruction variants).
  * Models vary in tool-calling propensity; adjust the system prompt or run
    without structural tags if you get no calls.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timedelta
from typing import Any, Callable, ClassVar, Dict, List, Tuple

from pydantic import BaseModel, Field
from openvino_genai import (
    LLMPipeline,
    GenerationConfig,
    StructuredOutputConfig,
    StructuralTagsConfig,
    StructuralTagItem,
    StreamingStatus,
)


# ---------------------------------------------------------------------------
# Tool (function) schemas
# ---------------------------------------------------------------------------
class ToolRequest(BaseModel):
    """Base class with helpers for name & string representation."""

    _name: ClassVar[str]

    @classmethod
    def get_name(cls) -> str:
        return cls._name

    @classmethod
    def string_representation(cls) -> str:
        return f'<function_name="{cls.get_name()}", arguments={list(cls.model_fields.keys())}>'


class WeatherRequest(ToolRequest):
    _name: ClassVar[str] = "get_weather"
    city: str = Field(description="City name")
    country: str = Field(description="Country name")
    date: str = Field(pattern=r"2\d\d\d-[0-1]\d-[0-3]\d", description="Date in YYYY-MM-DD format")


class CurrencyExchangeRequest(ToolRequest):
    _name: ClassVar[str] = "get_currency_exchange"
    from_currency: str = Field(description="Currency to convert from (ISO code)")
    to_currency: str = Field(description="Currency to convert to (ISO code)")
    amount: float = Field(ge=0.0, description="Amount to convert")


TOOLS: Dict[str, type[ToolRequest]] = {
    WeatherRequest.get_name(): WeatherRequest,
    CurrencyExchangeRequest.get_name(): CurrencyExchangeRequest,
}

# Prompt pack (id -> description, prompt)
PROMPT_PACK: Dict[int, Tuple[str, str]] = {
    1: ("Single city weather", "What is the weather in Dublin today?"),
    2: ("Weather + currency", "What is the weather in London today and how many USD for 75 EUR?"),
    3: ("Yesterday & conversion", "Tell me yesterday's weather in Paris and convert 200 EUR to JPY."),
    4: ("Compare cities", "Compare today's weather in Oslo and Helsinki, then convert 40 EUR to GBP."),
    5: ("Multi weather", "I need the weather for Rome today and for Zurich yesterday."),
    6: ("Currency only", "How many USD can I get for 100 EUR?"),
    7: ("Chained conversion", "First convert 50 EUR to GBP and then convert the result to JPY."),
    8: ("Implicit follow ups", "What's the weather in Tokyo today and in New York yesterday, and how many dollars for 300 euros?"),
}


# ---------------------------------------------------------------------------
# Mock tool implementations
# ---------------------------------------------------------------------------
def tool_get_weather(req: WeatherRequest) -> Dict[str, Any]:
    # Deterministic pseudo-random weather based on city (non-cryptographic)
    rng = random.Random(req.city.lower())  # nosec B311 - acceptable for mock data
    temperature_c = round(rng.uniform(-5, 30), 1)
    conditions = ["Sunny", "Cloudy", "Rain", "Fog", "Windy", "Snow"]
    condition = rng.choice(conditions)
    return {
        "city": req.city,
        "country": req.country,
        "date": req.date,
        "temperature_c": temperature_c,
        "condition": condition,
    }


def tool_get_currency_exchange(req: CurrencyExchangeRequest) -> Dict[str, Any]:
    # Static mock rates (relative to EUR)
    eur_rates = {
        "EUR": 1.0,
        "USD": 1.08,
        "GBP": 0.85,
        "JPY": 170.0,
        "CHF": 0.95,
    }
    if req.from_currency not in eur_rates or req.to_currency not in eur_rates:
        raise ValueError("Unsupported currency in mock tool")
    # Convert via EUR base
    amount_in_eur = req.amount / eur_rates[req.from_currency]
    converted = amount_in_eur * eur_rates[req.to_currency]
    return {
        "from_currency": req.from_currency,
        "to_currency": req.to_currency,
        "amount": req.amount,
        "rate": round(converted / req.amount, 6) if req.amount else None,
        "converted_amount": round(converted, 4),
    }


TOOL_IMPLS: Dict[str, Callable[[Any], Dict[str, Any]]] = {  # input validated separately
    WeatherRequest.get_name(): tool_get_weather,
    CurrencyExchangeRequest.get_name(): tool_get_currency_exchange,
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
FUNCTION_CALL_REGEX = re.compile(r'<function="([^"<>]+)">\s*({.*?})\s*</function>', re.DOTALL)


def parse_function_calls(text: str) -> List[dict[str, Any]]:
    calls: List[dict[str, Any]] = []
    for match in re.finditer(FUNCTION_CALL_REGEX, text):
        fn_name, json_str = match.groups()
        try:
            payload = json.loads(json_str)
            calls.append({"name": fn_name, "arguments": payload, "raw": match.group(0)})
        except json.JSONDecodeError:
            continue
    return calls


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
def build_system_prompt() -> str:
    today = datetime.today().strftime("%Y-%m-%d")
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    parts = [
        "You are a helpful assistant with access to tools.",
        f"Today is {today}. If user asks about 'yesterday' use the date {yesterday}.",
        "Decide if tools are needed; if so, call them. You MAY call multiple tools.",
        "TOOLS AVAILABLE:",
        "\n".join([t.string_representation() for t in TOOLS.values()]),
        "FORMAT STRICT FOR EACH CALL (one per block):",
        "<function=\"function_name\">{JSON arguments}</function>",
        ("If no tool is needed, answer normally. After tool results are provided, "
         "write a concise final answer for the user."),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def _prepare_config(use_structural_tags: bool) -> GenerationConfig:
    """Helper to create generation config with optional structural tags."""
    config = GenerationConfig()
    config.max_new_tokens = 320
    config.do_sample = True
    if use_structural_tags:
        config.structured_output_config = StructuredOutputConfig(
            structural_tags_config=StructuralTagsConfig(
                structural_tags=[
                    StructuralTagItem(
                        begin=f'<function="{name}">',
                        schema=json.dumps(tool.model_json_schema()),
                        end="</function>",
                    )
                    for name, tool in TOOLS.items()
                ],
                triggers=["<function="],
            )
        )
    return config


def _make_stream_collector() -> Tuple[Callable[[str], StreamingStatus], List[str]]:
    """Create a streamer callback that collects tokens and prints them live."""
    acc: List[str] = []

    def _streamer(subword: str) -> StreamingStatus:
        print(subword, end="", flush=True)
        acc.append(subword)
        return StreamingStatus.RUNNING

    return _streamer, acc


def _tool_cache_key(name: str, args: Dict[str, Any]) -> str:
    return name + "|" + json.dumps(args, sort_keys=True)


def _suggest_followups(answer: str, last_calls: List[dict]) -> List[str]:
    suggestions: List[str] = []
    tool_names = {c['name'] for c in last_calls}
    if 'get_weather' in tool_names:
        suggestions.append("Ask weather for another city tomorrow")
        suggestions.append("Compare temperatures between two cities")
    if 'get_currency_exchange' in tool_names:
        suggestions.append("Convert a different amount to another currency")
        suggestions.append("Ask for multi-step conversion chain")
    if not suggestions:
        suggestions = ["Ask for weather", "Ask for a currency conversion"]
    return suggestions[:3]


def run_single_question(
    pipe: LLMPipeline,
    question: str,
    use_structural_tags: bool,
    max_rounds: int,
    verbose: bool = False,
    stream: bool = True,
    tool_cache: Dict[str, Dict[str, Any]] | None = None,
) -> str:
    """Run a single user question through the tool-calling loop and return final answer.

    tool_cache: maps cache_key -> tool_result (raw dict) to avoid duplicate calls.
    """
    if tool_cache is None:
        tool_cache = {}
    pipe.start_chat(build_system_prompt())
    config = _prepare_config(use_structural_tags)

    streamer_cb, collected = (_make_stream_collector() if stream else (None, []))
    if streamer_cb:
        print("", end="")  # ensure we are at line start
    response = pipe.generate(question, config, streamer=streamer_cb) if streamer_cb else pipe.generate(question, config)
    if streamer_cb:
        print()  # newline after streaming
        response = "".join(collected)
    if verbose and not streamer_cb:
        print("[Model initial]\n" + response)

    # Follow tool calling rounds
    for round_idx in range(max_rounds):
        calls = parse_function_calls(response)
        if not calls:
            break
        tool_results = []
        for call in calls:
            name = call["name"]
            args = call["arguments"]
            tool_cls = TOOLS.get(name)
            impl = TOOL_IMPLS.get(name)
            if not tool_cls or not impl:
                continue
            cache_key = _tool_cache_key(name, args)
            if cache_key in tool_cache:
                cached_res = tool_cache[cache_key]
                tool_results.append({"name": name, "arguments": args, "result": cached_res, "cached": True})
                if verbose:
                    print(f"[Tool:CACHED] {name} -> {cached_res}")
                continue
            try:
                validated = tool_cls(**args)
                result = impl(validated)
                tool_cache[cache_key] = result
                tool_results.append({"name": name, "arguments": args, "result": result, "cached": False})
                if verbose:
                    print(f"[Tool] {name} -> {result}")
            except Exception as exc:  # pylint: disable=broad-except
                if verbose:
                    print(f"[Tool ERROR] {name}: {exc}")
        if not tool_results:
            break
        follow_up_prompt = (
            "Tool results available (JSON list). Write final answer for the user. "
            "If MORE calls are still REQUIRED, call them first, otherwise answer.\n"
            f"TOOL_RESULTS={json.dumps(tool_results, ensure_ascii=False)}"
        )
        streamer_cb2, collected2 = (_make_stream_collector() if stream else (None, []))
        response = (
            pipe.generate(follow_up_prompt, config, streamer=streamer_cb2)
            if streamer_cb2
            else pipe.generate(follow_up_prompt, config)
        )
        if streamer_cb2:
            print()
            response = "".join(collected2)
        if verbose and not stream:
            print(f"[Model follow-up {round_idx}]\n" + response)
    pipe.finish_chat()
    return response


def _print_prompt_pack():
    print("Available prompt IDs:")
    for pid, (desc, prompt) in PROMPT_PACK.items():
        print(f"  {pid}. {desc} -> {prompt}")


def run_chat_loop(pipe: LLMPipeline, use_structural_tags: bool, max_rounds: int, stream: bool, show_suggestions: bool) -> None:
    """Interactive chat loop.

    User enters messages. Assistant replies using function calling if needed.
    Type 'quit' / 'exit' / 'bye' / 'q' to stop.
    """
    print("=== OpenVINO GenAI Function Calling Chat ===")
    print("Type your questions. I can help with weather info and currency conversions.")
    print("Commands: :prompts (list), :id <n> (use prompt id), :cache (show cache), :clearcache, :help, quit")
    print("Type 'quit' to exit.\n")
    system_summary = build_system_prompt().split("\n")[0]
    print(f"OVGenAI: Hello, how can I help you today? I can assist you to {system_summary.lower()}")

    tool_cache: Dict[str, Dict[str, Any]] = {}
    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nOVGenAI: Goodbye!")
            break
        if user_input.lower() in {"quit", "exit", "bye", "q"}:
            print("OVGenAI: Goodbye!")
            break
        if user_input.startswith(":"):
            cmd_parts = user_input[1:].split()
            if not cmd_parts:
                continue
            cmd = cmd_parts[0].lower()
            if cmd == "prompts":
                _print_prompt_pack()
                continue
            if cmd == "id" and len(cmd_parts) > 1 and cmd_parts[1].isdigit():
                pid = int(cmd_parts[1])
                if pid in PROMPT_PACK:
                    user_input = PROMPT_PACK[pid][1]
                    print(f"[Using prompt {pid}: {PROMPT_PACK[pid][0]}]")
                else:
                    print("Unknown prompt id")
                    continue
            elif cmd == "cache":
                print(f"Cache entries: {len(tool_cache)}")
                for k, v in tool_cache.items():
                    print(f"  {k} -> {v}")
                continue
            elif cmd == "clearcache":
                tool_cache.clear()
                print("Cache cleared.")
                continue
            elif cmd == "help":
                print(":prompts, :id <n>, :cache, :clearcache, quit")
                continue
            # If unknown colon-command fall through to normal processing
        if not user_input:
            continue
        # Get answer (not verbose in chat mode)
        answer = run_single_question(
            pipe,
            user_input,
            use_structural_tags,
            max_rounds,
            verbose=False,
            stream=stream,
            tool_cache=tool_cache,
        )
        # Fallback: if model produced nothing, echo
        if not answer.strip():
            answer = f"You said: {user_input}"
        print(f"OVGenAI: {answer}\n")
        if show_suggestions:
            calls = parse_function_calls(answer)
            sugg = _suggest_followups(answer, calls)
            if sugg:
                print("Suggestions:")
                for s in sugg:
                    print(f"  - {s}")
                print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Function calling sample. Run without --prompt to enter interactive chat loop; "
            "with --prompt to run a single question."
        )
    )
    parser.add_argument("model_dir", help="Path to the model directory (OpenVINO IR or GGUF file).")
    parser.add_argument("--prompt", help="Single question to process instead of interactive chat loop.")
    parser.add_argument(
        "--max-rounds", type=int, default=2, help="Max tool-calling refinement rounds per question."
    )
    parser.add_argument(
        "--no-structural-tags", action="store_true", help="Disable structural tags enforcement."
    )
    parser.add_argument("--device", default="CPU", help="Inference device (CPU, GPU, etc.).")
    parser.add_argument("--verbose", action="store_true", help="Verbose logs (single-shot mode only).")
    parser.add_argument("--no-stream", action="store_true", help="Disable token streaming.")
    parser.add_argument("--prompt-id", type=int, help="Select a prompt from the prompt pack by id (single-shot mode).")
    parser.add_argument("--list-prompts", action="store_true", help="List available prompt pack items and exit.")
    parser.add_argument("--no-suggestions", action="store_true", help="Disable follow-up suggestions in chat mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = LLMPipeline(args.model_dir, args.device)
    use_structural = not args.no_structural_tags
    if args.list_prompts:
        _print_prompt_pack()
        return
    selected_prompt = args.prompt
    if args.prompt_id and not selected_prompt:
        if args.prompt_id in PROMPT_PACK:
            selected_prompt = PROMPT_PACK[args.prompt_id][1]
        else:
            print("Unknown prompt id")
            return
    stream = not args.no_stream
    if selected_prompt:
        answer = run_single_question(
            pipe,
            selected_prompt,
            use_structural,
            args.max_rounds,
            verbose=args.verbose,
            stream=stream,
            tool_cache={},
        )
        print(answer)
    else:
        run_chat_loop(pipe, use_structural, args.max_rounds, stream=stream, show_suggestions=not args.no_suggestions)


if __name__ == "__main__":
    main()
