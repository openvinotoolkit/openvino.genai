from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from openvino import save_model
from openvino_genai import Tokenizer
from openvino_tokenizers import convert_tokenizer
from transformers import AutoTokenizer


DEFAULT_CHECKPOINT = "google/gemma-3-1b-it"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS_FILE = REPO_ROOT / "tools" / "who_what_benchmark" / "whowhatbench" / "prompts" / "text_long_prompts.yaml"
DEFAULT_CONVERTED_TOKENIZER_DIR = Path(__file__).resolve().parent / "gemma-3-1b-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate WWB long prompts against tokenizer chat-template handling without loading any model. "
            "The script compares the Hugging Face tokenizer with the tokenizer converted by openvino_tokenizers, "
            "and optionally with an exported OpenVINO tokenizer directory."
        )
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Hugging Face tokenizer source.")
    parser.add_argument(
        "--ov-tokenizer-dir",
        type=Path,
        default=DEFAULT_CONVERTED_TOKENIZER_DIR,
        help=(
            "Directory with OpenVINO tokenizer files. If it is empty or missing, the script converts the HF tokenizer "
            "and saves openvino_tokenizer.xml and openvino_detokenizer.xml there."
        ),
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=DEFAULT_PROMPTS_FILE,
        help="WWB prompt file to validate. Defaults to the internal long-prompt dataset.",
    )
    parser.add_argument("--language", default="en", help="Prompt language key from the WWB YAML file.")
    parser.add_argument(
        "--prompt-index",
        type=int,
        nargs="*",
        default=None,
        help="Zero-based prompt indices to validate. If omitted, validate all prompts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Validate only the first N prompts when --prompt-index is not provided.",
    )
    parser.add_argument(
        "--show-text-on-match",
        action="store_true",
        help="Print templated prompt text even when no mismatch is found.",
    )
    return parser.parse_args()


def load_prompts(prompts_file: Path, language: str, prompt_indices: list[int] | None, limit: int | None) -> list[tuple[int, str]]:
    prompt_data = yaml.safe_load(prompts_file.read_text(encoding="utf-8"))
    prompts = prompt_data[language]["prompts"]

    if prompt_indices is not None:
        return [(index, prompts[index]) for index in prompt_indices]
    if limit is not None:
        return list(enumerate(prompts[:limit]))
    return list(enumerate(prompts))


def encode_hf(tokenizer, text: str) -> list[int]:
    return tokenizer([text], return_tensors="np").input_ids[0].tolist()


def encode_ov(tokenizer: Tokenizer, text: str) -> list[int]:
    return np.asarray(tokenizer.encode(text).input_ids.data).reshape(-1).tolist()


def get_string_diff(left: str, right: str, context: int = 120) -> str:
    if left == right:
        return "identical"

    mismatch_index = next(
        index for index, (left_char, right_char) in enumerate(zip(left, right)) if left_char != right_char
    )
    start = max(0, mismatch_index - context)
    end = mismatch_index + context
    left_excerpt = left[start:end].replace("\n", "\\n")
    right_excerpt = right[start:end].replace("\n", "\\n")
    return (
        f"first mismatch at char {mismatch_index}\n"
        f"hf : {left_excerpt}\n"
        f"ov : {right_excerpt}"
    )


def get_token_diff(reference: list[int], candidate: list[int], context: int = 12) -> str:
    if reference == candidate:
        return "identical"

    limit = min(len(reference), len(candidate))
    mismatch_index = next((index for index in range(limit) if reference[index] != candidate[index]), limit)
    start = max(0, mismatch_index - context)
    end = min(max(len(reference), len(candidate)), mismatch_index + context)
    return (
        f"first mismatch at token {mismatch_index}; lengths: hf={len(reference)}, other={len(candidate)}\n"
        f"hf : {reference[start:end]}\n"
        f"ov : {candidate[start:end]}"
    )


def print_templated_text(label: str, text: str) -> None:
    print(f"{label} templated prompt:")
    print("```text")
    print(text)
    print("```")


def is_directory_empty(path: Path) -> bool:
    return not any(path.iterdir())


def get_or_create_ov_tokenizer(hf_tokenizer, output_dir: Path) -> Tokenizer:
    tokenizer_xml = output_dir / "openvino_tokenizer.xml"
    detokenizer_xml = output_dir / "openvino_detokenizer.xml"

    output_dir.mkdir(parents=True, exist_ok=True)

    if tokenizer_xml.exists():
        print(f"Reusing converted OpenVINO tokenizer from: {output_dir}")
        return Tokenizer(str(output_dir))

    if not is_directory_empty(output_dir):
        raise RuntimeError(
            f"Directory exists and is not empty, but {tokenizer_xml.name} is missing: {output_dir}"
        )

    print(f"Converting Hugging Face tokenizer and saving to: {output_dir}")
    tokenizer_model, detokenizer_model = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
    save_model(tokenizer_model, str(tokenizer_xml))
    save_model(detokenizer_model, str(detokenizer_xml))
    return Tokenizer(str(output_dir))


def validate_backend(name: str, backend_tokenizer: Tokenizer, hf_tokenizer, prompts: list[tuple[int, str]], show_text_on_match: bool) -> dict[str, int]:
    summary = {
        "prompt_tokenization_failures": 0,
        "chat_template_failures": 0,
        "hf_chat_tokenization_failures": 0,
        "backend_chat_tokenization_failures": 0,
    }

    print(f"\n===== {name} =====")
    for prompt_index, prompt in prompts:
        chat_history = [{"role": "user", "content": prompt}]

        hf_prompt_ids = encode_hf(hf_tokenizer, prompt)
        backend_prompt_ids = encode_ov(backend_tokenizer, prompt)

        hf_chat = hf_tokenizer.apply_chat_template(chat_history, add_generation_prompt=True, tokenize=False)
        backend_chat = backend_tokenizer.apply_chat_template(chat_history, add_generation_prompt=True)

        hf_chat_ids = encode_hf(hf_tokenizer, hf_chat)
        backend_hf_chat_ids = encode_ov(backend_tokenizer, hf_chat)
        backend_chat_ids = encode_ov(backend_tokenizer, backend_chat)
        hf_backend_chat_ids = encode_hf(hf_tokenizer, backend_chat)

        prompt_tokenization_match = hf_prompt_ids == backend_prompt_ids
        chat_template_match = hf_chat == backend_chat
        hf_chat_tokenization_match = hf_chat_ids == backend_hf_chat_ids
        backend_chat_tokenization_match = hf_backend_chat_ids == backend_chat_ids

        if not prompt_tokenization_match:
            summary["prompt_tokenization_failures"] += 1
        if not chat_template_match:
            summary["chat_template_failures"] += 1
        if not hf_chat_tokenization_match:
            summary["hf_chat_tokenization_failures"] += 1
        if not backend_chat_tokenization_match:
            summary["backend_chat_tokenization_failures"] += 1

        passed = (
            prompt_tokenization_match
            and chat_template_match
            and hf_chat_tokenization_match
            and backend_chat_tokenization_match
        )

        status = "PASS" if passed else "FAIL"
        print(
            f"Prompt {prompt_index}: {status} | "
            f"prompt_ids={prompt_tokenization_match} | "
            f"chat_template={chat_template_match} | "
            f"encode_hf_chat={hf_chat_tokenization_match} | "
            f"encode_backend_chat={backend_chat_tokenization_match}"
        )

        if passed and show_text_on_match:
            print_templated_text("HF", hf_chat)
            print_templated_text(name, backend_chat)

        if not prompt_tokenization_match:
            print("  Raw prompt token mismatch:")
            print(f"  {get_token_diff(hf_prompt_ids, backend_prompt_ids)}")

        if not chat_template_match:
            print("  Chat template mismatch:")
            print(f"  {get_string_diff(hf_chat, backend_chat)}")

        if not hf_chat_tokenization_match:
            print("  Tokenization mismatch on HF chat-template string:")
            print(f"  {get_token_diff(hf_chat_ids, backend_hf_chat_ids)}")

        if not backend_chat_tokenization_match:
            print("  Tokenization mismatch on backend chat-template string:")
            print(f"  {get_token_diff(hf_backend_chat_ids, backend_chat_ids)}")

    return summary


def main() -> int:
    args = parse_args()

    print(f"Loading Hugging Face tokenizer from: {args.checkpoint}")
    hf_tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if hf_tokenizer.chat_template is None:
        raise RuntimeError(f"Tokenizer {args.checkpoint} has no chat_template")

    prompts = load_prompts(args.prompts_file, args.language, args.prompt_index, args.limit)
    print(f"Loaded {len(prompts)} prompt(s) from: {args.prompts_file}")

    backends: list[tuple[str, Tokenizer]] = [
        ("converted_openvino_tokenizer", get_or_create_ov_tokenizer(hf_tokenizer, args.ov_tokenizer_dir)),
    ]

    any_failure = False
    for name, backend_tokenizer in backends:
        summary = validate_backend(name, backend_tokenizer, hf_tokenizer, prompts, args.show_text_on_match)
        any_failure = any_failure or any(summary.values())
        print(f"Summary for {name}: {summary}")

    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())


