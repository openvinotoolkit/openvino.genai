# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Gemma3 GGUF acceptance against a pinned llama.cpp CPU oracle.

Build gguf_mmproj_oracle.cpp against REFERENCE_REVISION. No llama.cpp production dependency.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai as genai
from PIL import Image

REFERENCE_REVISION = "16fb7d9d326a3fe69a331ce5fbe7a679a1a281bb"


class Tokens(genai.StreamerBase):
    def __init__(self):
        super().__init__()
        self.tokens = []

    def write(self, tokens):
        self.tokens.extend(tokens if isinstance(tokens, list) else [tokens])
        return genai.StreamingStatus.RUNNING

    def end(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", type=Path)
    parser.add_argument("mmproj", type=Path)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--chat", action="store_true", help="Also compare a cached image chat follow-up")
    args = parser.parse_args()
    pipe = genai.VLMPipeline(str(args.language), "CPU", mmproj_path=str(args.mmproj),
                            INFERENCE_PRECISION_HINT="f32", DYNAMIC_QUANTIZATION_GROUP_SIZE=0,
                            INFERENCE_NUM_THREADS=4)
    tokenizer = pipe.get_tokenizer()
    if args.image:
        pixels = np.asarray(Image.open(args.image).convert("RGB"))
    else:
        pixels = np.zeros((64, 64, 3), np.uint8)
        pixels[..., 0] = 255
    cases = []
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        image_file = directory / "image.png"
        Image.fromarray(pixels).save(image_file)

        def compare(messages, tokens, text, modality, with_image):
            """Replay `tokens` through the oracle on the same history and score the agreement."""
            # GenAI expands image markers into embeddings. mtmd uses its own marker and adds
            # the same Gemma3 begin/end-image tokens around the reference encoder output.
            (directory / "prompt.txt").write_text(
                tokenizer.apply_chat_template(messages, add_generation_prompt=True))
            (directory / "history.txt").write_text(" ".join(map(str, tokens)))
            process = subprocess.run([str(args.oracle.resolve()), str(args.language.resolve()),
                str(args.mmproj.resolve()), str(image_file) if with_image else "-",
                str(directory / "prompt.txt"), str(directory / "history.txt")], capture_output=True, text=True)
            (args.report.parent / f"{args.report.stem}-{modality}.log").write_text(process.stderr)
            process.check_returncode()
            choices = next(line for line in process.stdout.splitlines() if line.startswith("CHOICES"))
            reference = list(map(int, choices.split()[1:]))
            assert len(reference) == len(tokens) and reference
            case = {"modality": modality, "text": text, "tokens": tokens,
                    "reference_choices_on_same_history": reference,
                    "first_token_matches": reference[0] == tokens[0],
                    "matching_choice_fraction": sum(a == b for a, b in zip(reference, tokens)) / len(reference)}
            cases.append(case)
            print(json.dumps(case), flush=True)
            return case

        for image in (False, True):
            prompt = "<start_of_image>\nDescribe the image." if image else "What is 2 plus 2?"
            stream = Tokens()
            kwargs = {"images": [ov.Tensor(pixels[None])]} if image else {}
            result = pipe.generate(prompt, max_new_tokens=20, do_sample=False, streamer=stream, **kwargs)
            compare([{"role": "user", "content": prompt.replace("<start_of_image>", "<__media__>")}],
                    stream.tokens, result.texts[0], "image" if image else "text", image)
        chat_reset_matches = True
        if args.chat:
            pipe.start_chat()
            first = Tokens()
            first_result = pipe.generate("<start_of_image>\nDescribe the image.",
                images=[ov.Tensor(pixels[None])], max_new_tokens=20, do_sample=False, streamer=first)
            chat_reset_matches = first.tokens == cases[1]["tokens"]
            followup = Tokens()
            followup_prompt = "What is shown?"
            followup_result = pipe.generate(followup_prompt, max_new_tokens=20,
                                            do_sample=False, streamer=followup)
            compare([{"role": "user", "content": "<__media__>\nDescribe the image."},
                     {"role": "assistant", "content": first_result.texts[0]},
                     {"role": "user", "content": followup_prompt}],
                    followup.tokens, followup_result.texts[0], "image_chat", True)
            pipe.finish_chat()
        # A second request must start with an empty cache.
        reset_stream = Tokens()
        pipe.generate("What is 2 plus 2?", max_new_tokens=20, do_sample=False, streamer=reset_stream)
        reset_matches = reset_stream.tokens == cases[0]["tokens"]
    report = {"language": str(args.language), "mmproj": str(args.mmproj),
              "reference_revision": REFERENCE_REVISION,
              "cases": cases, "request_reset_matches": reset_matches,
              "chat_checked": args.chat, "chat_reset_matches": chat_reset_matches,
              "passed": reset_matches and chat_reset_matches and all(c["first_token_matches"] and c["matching_choice_fraction"] >= .9 for c in cases)}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
