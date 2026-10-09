#!/usr/bin/env python3
# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai
import openvino as ov
import librosa


def read_wav(filepath):
    raw_speech, samplerate = librosa.load(filepath, sr=16000)
    return ov.Tensor(raw_speech)


def get_config_for_cache():
    config_cache = dict()
    config_cache["CACHE_DIR"] = "asr_cache"
    return config_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="Path to the model directory")
    parser.add_argument("device", help="Device to run the model on")
    parser.add_argument("wav_file_path", nargs="+", help="Path to the WAV file")
    args = parser.parse_args()

    ov_config = {}
    if args.device == "NPU" or "GPU" in args.device:  # need to handle cases like "GPU", "GPU.0" and "GPU.1"
        # Cache compiled models on disk for GPU and NPU to save time on the
        # next run. It's not beneficial for CPU.
        ov_config = get_config_for_cache()

    scheduler_config = openvino_genai.SchedulerConfig()
    pipe = openvino_genai.ContinuousBatchingPipeline(args.model_dir, scheduler_config, args.device, ov_config)

    audios = [read_wav(wav_file) for wav_file in args.wav_file_path]

    handles: list[openvino_genai.GenerationHandle] = []
    history = openvino_genai.ChatHistory(
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio"}]},
        ]
    )
    tokenizer = pipe.get_tokenizer()
    prompt = tokenizer.apply_chat_template(history, add_generation_prompt=True)
    prompts = [prompt] * len(audios)
    generation_config = openvino_genai.GenerationConfig()

    for request_id, (audio, prompt) in enumerate(zip(audios, prompts)):
        handle = pipe.add_request(
            request_id, prompt, images=[], videos=[], audios=[audio], generation_config=generation_config
        )
        handles.append(handle)

    while pipe.has_non_finished_requests():
        pipe.step()

    for idx, handle in enumerate(handles):
        handle_result = handle.read_all()[0]
        text_result = tokenizer.decode(handle_result.generated_ids)
        print(f"Result for audio {idx}: {text_result}")


if "__main__" == __name__:
    main()
