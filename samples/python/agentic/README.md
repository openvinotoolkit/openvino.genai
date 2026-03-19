# OpenVINO GenAI Agentic Python Sample

This folder contains a LangChain-compatible local wrapper for `openvino_genai.LLMPipeline` and a runnable agent example.

## Contents

- `llm_wrapper.py`: `OpenVINOChatModel` implementation for LangChain chat model APIs.
- `example_agent.py`: End-to-end example with a safe calculator tool.
- `__init__.py`: Export for `OpenVINOChatModel`.

## Prerequisites

1. Install OpenVINO GenAI sample dependencies:

```sh
pip install --upgrade-strategy eager -r ../../deployment-requirements.txt
```

2. Install a converted OpenVINO LLM model directory (contains model + tokenizer files).

## Quick Start

Run the example agent:

```sh
python example_agent.py <MODEL_PATH>
```

Run with custom device and question:

```sh
python example_agent.py <MODEL_PATH> --device CPU --question "What is (25 * 4) + 10?"
```

Expected flow:

1. Agent receives a question.
2. Agent calls the calculator tool for arithmetic.
3. Agent returns a final answer.

## Wrapper Usage

Minimal wrapper usage in code:

```python
from llm_wrapper import OpenVINOChatModel
from langchain_core.messages import HumanMessage

llm = OpenVINOChatModel(model_path="<MODEL_PATH>", device="CPU")
result = llm.invoke([HumanMessage(content="Say hello")])
print(result.content)
```

Streaming usage:

```python
from llm_wrapper import OpenVINOChatModel
from langchain_core.messages import HumanMessage

llm = OpenVINOChatModel(model_path="<MODEL_PATH>", device="CPU")
for chunk in llm.stream([HumanMessage(content="Count from 1 to 5")]):
    print(chunk.content, end="", flush=True)
```

## Safe Calculator Tool

`example_agent.py` uses an AST-based arithmetic evaluator.

Allowed:

- Binary operators: `+`, `-`, `*`, `/`
- Unary operators: `+`, `-`
- Parentheses
- Integer/float constants

Disallowed:

- Function calls (`abs(1)`)
- Attribute access
- Variables
- Any executable code

## Current Limitations

- `OpenVINOChatModel` currently supports only `num_return_sequences=1`.
- Supported message types are `SystemMessage`, `HumanMessage`, and `AIMessage`.
- Message content must be `str`.
- Runtime kwargs are limited to explicitly whitelisted generation parameters in `llm_wrapper.py`.

## Troubleshooting

- `Model path does not exist`: Verify `<MODEL_PATH>` points to a valid OpenVINO model directory.
- `LangChain is required`: Install `langchain` and `langchain-core` from `deployment-requirements.txt`.
- `Unsupported generation kwargs`: Remove unsupported kwargs and use the supported set implemented in `_validate_kwargs`.
- `Unsupported message type`: Convert tool or custom message types to supported chat message types before calling the wrapper.
