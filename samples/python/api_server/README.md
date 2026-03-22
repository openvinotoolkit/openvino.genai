# OpenVINO GenAI OpenAI-Compatible API Server

This sample provides an OpenAI-compatible REST API server built with FastAPI. It wraps the OpenVINO GenAI `LLMPipeline` in a standard `/v1/chat/completions` endpoint, allowing integration with the `openai` Python package, AutoGen, CrewAI, and LangGraph.

When a request includes OpenAI `tools` definitions, the server automatically translates them into an OpenVINO GenAI `StructuredOutputConfig(json_schema=...)` to guarantee constrained generation.

## Prerequisites

Install the sample dependencies:
```bash
pip install -r ../../deployment-requirements.txt
```

## Running the Server

```bash
python openai_api_server.py /path/to/openvino_model --device CPU --port 8000
```

## Usage Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="ov-local"
)

response = client.chat.completions.create(
    model="openvino-local",
    messages=[{"role": "user", "content": "What's the weather in Seattle?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    }]
)

print(response.choices[0].message)
```

## How It Works

1. Incoming `tools` definitions are merged into a single JSON Schema (`anyOf`) that constrains model output to valid tool calls.
2. The model generates output natively constrained by the schema via `StructuredOutputConfig`.
3. The generated JSON is repacked as a standard `tool_calls` response in the Chat Completion payload.
