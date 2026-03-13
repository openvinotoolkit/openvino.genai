from typing import Any
import argparse

import torch

from .model import load_model
from .generate import init_history, generate_response
from .io_utils import parse_user_input, save_audio

HELP_TEXT = """Commands:
  /image <path>  - Attach an image file (use quotes for paths with spaces)
  /audio <path>  - Attach an audio file
  /video <path>  - Attach a video file
  /clear         - Clear chat history
  /help          - Show this help
  /quit          - Exit chat
"""


def _handle_command(
    user_input: str,
    history: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    if user_input in ("/quit", "/exit"):
        print("Goodbye!")
        return "quit", history
    if user_input == "/help":
        print(HELP_TEXT)
        return "continue", history
    if user_input == "/clear":
        print("Chat history cleared.\n")
        return "continue", init_history()
    return None, history


def chat_loop(
    model: Any,
    processor: Any,
    enable_audio: bool,
    speaker: str | None,
    output_dir: str,
) -> None:
    history = init_history()
    turn = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        action, history = _handle_command(user_input, history)
        if action == "quit":
            break
        if action == "continue":
            continue

        content = parse_user_input(user_input)
        if not content:
            continue

        history.append({"role": "user", "content": content})

        print("Qwen: ", end="", flush=True)
        with torch.inference_mode():
            text, audio = generate_response(
                model,
                processor,
                history,
                enable_audio,
                speaker,
            )

        if audio is not None:
            turn += 1
            path = save_audio(audio, output_dir, turn)
            print(f"[Audio saved to {path}]")

        history.append(
            {"role": "assistant", "content": [{"type": "text", "text": text}]},
        )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-Omni CLI Chat")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no_audio", action="store_true")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--speaker", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    enable_audio = not args.no_audio

    print(f"Loading model from {args.model_path}...")
    model, processor = load_model(args.model_path, args.device, enable_audio)
    print("Model loaded.\n")

    if args.demo:
        from .demo import run_demo

        run_demo(model, processor)
        return

    print("Type /help for commands.\n")
    chat_loop(model, processor, enable_audio, args.speaker, args.output_dir)


if __name__ == "__main__":
    main()
