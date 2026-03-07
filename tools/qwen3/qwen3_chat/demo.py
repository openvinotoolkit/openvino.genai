from typing import Any

import torch

from .generate import init_history, generate_response

DEMO_PROMPTS = [
    "Hello, who are you?",
    "What is 2+2? Answer in one sentence.",
    "Tell me a one-sentence joke.",
]


def run_demo(model: Any, processor: Any) -> None:
    print("=== Demo mode: text-only smoke test ===\n")

    for prompt in DEMO_PROMPTS:
        history = init_history()
        history.append(
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        )

        with torch.no_grad():
            text, _ = generate_response(
                model,
                processor,
                history,
                enable_audio=False,
                speaker=None,
            )

        print(f"You:  {prompt}")
        print(f"Qwen: {text}\n")

    print("=== Demo complete ===")
