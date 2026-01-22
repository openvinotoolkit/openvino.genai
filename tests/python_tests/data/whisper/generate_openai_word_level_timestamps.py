# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import whisper
import datasets
import json


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


if __name__ == "__main__":
    no_samples = 10
    ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").take(
        no_samples
    )
    samples = [i["audio"]["array"] for i in ds]
    openai_pipe, openai_options = get_openai_lib_pipe()

    results = []
    for sample in samples:
        openai_lib_result = openai_pipe.transcribe(sample, **openai_options)
        results.append(openai_lib_result)
    json.dump(results, open(f"librispeech_asr_dummy_{no_samples}_openai_whisper_tiny_results.json", "w"))
