#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import ast
import operator
from typing import Any

try:
    from .llm_wrapper import OpenVINOChatModel
except ImportError:
    from llm_wrapper import OpenVINOChatModel

def _resolve_langchain_agent_apis() -> dict[str, Any]:
    """Resolve LangChain APIs with compatibility fallback for 1.x and legacy agents."""
    try:
        from langchain.tools import tool
    except ImportError as import_error:
        raise ImportError(
            "LangChain is required for the agent sample. Install 'langchain' to run this script."
        ) from import_error

    try:
        from langchain.agents import create_agent
        return {
            "mode": "create_agent",
            "tool_decorator": tool,
            "create_agent": create_agent,
        }
    except ImportError:
        try:
            from langchain.agents import AgentType, initialize_agent
            from langchain.tools import Tool
        except ImportError as import_error:
            raise ImportError(
                "Compatible LangChain agent APIs were not found. Install a supported 'langchain' version."
            ) from import_error

        return {
            "mode": "initialize_agent",
            "tool_decorator": tool,
            "initialize_agent": initialize_agent,
            "AgentType": AgentType,
            "Tool": Tool,
        }


_ALLOWED_BINARY_OPERATORS: dict[type[ast.AST], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_ALLOWED_UNARY_OPERATORS: dict[type[ast.AST], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate_expression_node(node: ast.AST) -> float:
    """Evaluate a restricted arithmetic AST node."""
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINARY_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _evaluate_expression_node(node.left)
        right = _evaluate_expression_node(node.right)
        return float(_ALLOWED_BINARY_OPERATORS[op_type](left, right))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARY_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return float(_ALLOWED_UNARY_OPERATORS[op_type](_evaluate_expression_node(node.operand)))

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def safe_calculator(expression: str) -> str:
    """Safely evaluate basic arithmetic expressions.

    Supported operations: +, -, *, / and parentheses.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Expression must be a non-empty string")

    parsed = ast.parse(expression, mode="eval")
    result = _evaluate_expression_node(parsed.body)
    if result.is_integer():
        return str(int(result))
    return str(result)


def build_agent(model_path: str, device: str) -> tuple[Any, str]:
    """Build a zero-shot ReAct agent backed by OpenVINOChatModel."""
    apis = _resolve_langchain_agent_apis()

    llm = OpenVINOChatModel(
        model_path=model_path,
        device=device,
        max_new_tokens=256,
        temperature=0.1,
    )

    @apis["tool_decorator"]
    def calculator(expression: str) -> str:
        """Evaluate basic arithmetic expressions with +, -, *, / and parentheses."""
        return safe_calculator(expression)

    if apis["mode"] == "create_agent":
        agent = apis["create_agent"](
            model=llm,
            tools=[calculator],
            system_prompt=(
                "You are a helpful assistant. Use the calculator tool for arithmetic. "
                "When a calculation is requested, call the tool with a valid expression string."
            ),
        )
        return agent, "create_agent"

    calculator_tool = apis["Tool"](
        name="calculator",
        func=safe_calculator,
        description="Evaluate basic arithmetic expressions with +, -, *, / and parentheses.",
    )
    agent = apis["initialize_agent"](
        tools=[calculator_tool],
        llm=llm,
        agent=apis["AgentType"].ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    return agent, "initialize_agent"


def run_agent_query(agent: Any, question: str, mode: str) -> str:
    """Run one question through an initialized LangChain agent."""
    if mode == "create_agent":
        result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        messages = result.get("messages") if isinstance(result, dict) else None
        if isinstance(messages, list) and messages:
            last_message = messages[-1]
            content = getattr(last_message, "content", None)
            if isinstance(content, str):
                return content
            return str(content)
        return str(result)

    if hasattr(agent, "invoke"):
        result = agent.invoke({"input": question})
        if isinstance(result, dict) and "output" in result:
            return str(result["output"])
        return str(result)

    if hasattr(agent, "run"):
        return str(agent.run(question))

    raise RuntimeError("Agent object does not support invoke() or run()")


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain agent sample powered by OpenVINOChatModel")
    parser.add_argument("model_path", help="Path to OpenVINO model directory")
    parser.add_argument("--device", default="CPU", help="Inference device (default: CPU)")
    parser.add_argument("--question", default="What is 25 * 4 + 10?", help="Question to ask the agent")
    args = parser.parse_args()

    agent, mode = build_agent(model_path=args.model_path, device=args.device)
    answer = run_agent_query(agent, args.question, mode)
    print("\nFinal answer:")
    print(answer)


if __name__ == "__main__":
    main()
