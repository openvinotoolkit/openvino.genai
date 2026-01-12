#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai
import whisper
from pathlib import Path
from dataclasses import dataclass
import tqdm
import time
import datasets
from itertools import zip_longest


def get_openai_lib_pipe(model_size="tiny"):
    model = whisper.load_model(model_size)
    generate_options = {
        "task": "transcribe",
        "language": "en",
        "beam_size": 1,
        "best_of": 1,
        "word_timestamps": True,
    }
    return model, generate_options


def get_genai_pipe(models_path: Path, device: str = "CPU"):
    pipeline = openvino_genai.WhisperPipeline(
        models_path,
        device=device,
        # need to pass word_timestamps=True to ctor kwargs to run model transformations before complilation
        word_timestamps=True,
    )

    generate_options = {
        "return_timestamps": True,
        "word_timestamps": True,
        "language": "<|en|>",
        "task": "transcribe",
    }

    return pipeline, generate_options


@dataclass
class WhisperWordTiming:
    word: str
    start_ts: float
    end_ts: float

    def to_dict(self):
        return {
            "word": self.word,
            "start_ts": float(self.start_ts),
            "end_ts": float(self.end_ts),
        }

    @staticmethod
    def from_dict(obj):
        return WhisperWordTiming(
            word=obj["word"],
            start_ts=float(obj["start_ts"]),
            end_ts=float(obj["end_ts"]),
        )


def from_openai_to_word_timing(openai_word_timestamps):
    word_timings = []
    for segment in openai_word_timestamps["segments"]:
        for word in segment["words"]:
            word_timing = WhisperWordTiming(
                word=word["word"],
                start_ts=word["start"],
                end_ts=word["end"],
            )
            word_timings.append(word_timing)
    return word_timings


def run_sample(sample, models_path: Path, device="CPU"):
    openai_pipe, openai_options = get_openai_lib_pipe()
    genai_pipe, genai_options = get_genai_pipe(models_path, device=device)

    genai_result = genai_pipe.generate(sample, **genai_options)
    genai_lib_word_timings = genai_result.words

    openai_lib_result = openai_pipe.transcribe(sample, **openai_options)
    openai_lib_word_timings = from_openai_to_word_timing(openai_lib_result)

    return {
        "openai": {"transcription": openai_lib_result["text"], "word_timings": openai_lib_word_timings},
        "genai": {"transcription": genai_result.texts[0], "word_timings": genai_lib_word_timings},
    }


def collect_libs_results(samples, models_path: Path, device="CPU"):
    openai_pipe, openai_options = get_openai_lib_pipe()
    genai_pipe, genai_options = get_genai_pipe(models_path, device=device)

    results = []

    infer_times = {
        "genai": [],
        "openai": [],
    }

    for sample in tqdm.tqdm(samples):
        genai_start = time.perf_counter()
        genai_result = genai_pipe.generate(sample, **genai_options)
        genai_lib_word_timings = genai_result.words
        infer_times["genai"].append(time.perf_counter() - genai_start)

        openai_start = time.perf_counter()
        openai_lib_result = openai_pipe.transcribe(sample, **openai_options)
        openai_lib_word_timings = from_openai_to_word_timing(openai_lib_result)
        infer_times["openai"].append(time.perf_counter() - openai_start)

        results.append(
            {
                "openai": {"transcription": openai_lib_result["text"], "word_timings": openai_lib_word_timings},
                "genai": {"transcription": genai_result.texts[0], "word_timings": genai_lib_word_timings},
            }
        )
    return (results, infer_times)


def compare_word_timings(word_timings1, word_timings2, tol=0.1):
    """
    Compare timings lengths length
    And compare timings only, skipping words text comparison
    """
    len1 = len(word_timings1)
    len2 = len(word_timings2)
    min_len = min(len1, len2)

    if len1 != len2:
        return False

    for i in range(min_len):
        w1 = word_timings1[i]
        w2 = word_timings2[i]
        start_diff = abs(w1.start_ts - w2.start_ts)
        end_diff = abs(w1.end_ts - w2.end_ts)
        if start_diff > tol or end_diff > tol:
            return False

    return True


def compare_genai_openai_results(results, tol=0.1):
    compare_results = {"timings_match": 0, "transcriptions_match": 0, "total": len(results)}
    for result in results:
        genai_word_timings = result["genai"]["word_timings"]
        openai_word_timings = result["openai"]["word_timings"]
        timings_matched = compare_word_timings(genai_word_timings, openai_word_timings, tol=tol)
        if timings_matched:
            compare_results["timings_match"] += 1

        transcriptions_matched = result["genai"]["transcription"] == result["openai"]["transcription"]
        if transcriptions_matched:
            compare_results["transcriptions_match"] += 1
    return compare_results


def print_word_timings(genai_word_timings, openai_word_timings):
    print(f"{'GenAI':^25} | {'OpenAI':^25}")
    for g_word, o_word in zip_longest(genai_word_timings, openai_word_timings):
        o_word_str = f"{o_word.word:10}  {o_word.start_ts:0.2f} - {o_word.end_ts:0.2f}" if o_word else " " * 25
        g_word_str = f"{g_word.word:10}  {g_word.start_ts:0.2f} - {g_word.end_ts:0.2f}" if g_word else " " * 25
        print(f"{g_word_str}   |   {o_word_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to the model directory")
    parser.add_argument("device", nargs="?", default="CPU", help="Device to run the model on (default: CPU)")
    args = parser.parse_args()

    ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    samples = [item["audio"]["array"] for item in ds]
    print(f"Samples in the dataset: {len(samples)}")

    sample_results = run_sample(samples[0], Path(args.model_dir), device=args.device)
    print_word_timings(sample_results["genai"]["word_timings"], sample_results["openai"]["word_timings"])

    results, infer_times = collect_libs_results(samples, Path(args.model_dir), device=args.device)
    print("\n\n")
    for lib_name, times in infer_times.items():
        avg_time = sum(times) / len(times)
        print(f"Average inference time for {lib_name}: {avg_time:.4f} seconds")

    compare_results = compare_genai_openai_results(results, tol=0.3)

    print("\nComparison results:")
    print(f"Total samples processed: {compare_results['total']}")
    print(f"Transcriptions match: {compare_results['transcriptions_match']}")
    print(f"Word timings match: {compare_results['timings_match']}")

    # for whisper-tiny model
    # 0.1 tolerance
    # Comparison results:
    # Total samples processed: 73
    # Transcriptions match: 50
    # Word timings match: 49

    # 0.2 tolerance
    # Total samples processed: 73
    # Transcriptions match: 50
    # Word timings match: 56

    # 0.3 tolerance
    # Total samples processed: 73
    # Transcriptions match: 50
    # Word timings match: 61

    # 14900 CPU
    # Average inference time for genai: 0.1867 seconds
    # Average inference time for openai: 0.3045 seconds


if "__main__" == __name__:
    main()
