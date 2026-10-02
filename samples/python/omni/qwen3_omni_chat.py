#!/usr/bin/env python3
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-Omni multimodal chat sample.

Preview: the Qwen3-Omni API (OmniPipeline and related types) is a preview feature
and is subject to change in future releases.

Demonstrates text + image + audio + video -> text + speech output using the ChatHistory API.

Usage:
    python qwen3_omni_chat.py <MODEL_DIR> <IMAGE_FILE_OR_DIR> <VIDEO_FILE> [--audio WAV ...]
"""

import argparse
import itertools
from pathlib import Path

import cv2
import librosa
import numpy as np
import openvino_genai
import soundfile as sf
from openvino import Tensor
from PIL import Image


def streamer(subword: str) -> openvino_genai.StreamingStatus:
    """Stream text tokens to stdout."""
    print(subword, end="", flush=True)
    return openvino_genai.StreamingStatus.RUNNING


def read_image(path: str) -> Tensor:
    pic = Image.open(path).convert("RGB")
    image_data = np.array(pic)
    return Tensor(image_data)


def read_images(path: str) -> list[Tensor]:
    entry = Path(path)
    if entry.is_dir():
        return [read_image(str(file)) for file in sorted(entry.iterdir())]
    return [read_image(path)]


def load_audio(audio_path: str, target_sr: int = 16000) -> Tensor:
    """Load audio from WAV file and convert to float32 mono tensor at target_sr."""
    # librosa.load downmixes to mono, resamples, and returns float32 in one call.
    audio_data, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    return Tensor(audio_data)


def read_video(path: str, num_frames: int = 8) -> tuple[Tensor, openvino_genai.VideoMetadata]:
    """Load a video as an [N, H, W, 3] uint8 tensor and pick num_frames evenly spaced indices."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open the video file: {path}")

    # OpenCV reports 0 or -1 when a container/codec doesn't expose a frame count.
    total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_num_frames <= 0:
        cap.release()
        raise RuntimeError(f"Could not determine the frame count of {path}. The container or codec may not expose it.")

    # A short video can't yield num_frames distinct indices; sampling fewer here instead of emitting duplicates.
    sampled_frames = min(num_frames, total_num_frames)
    step = total_num_frames / sampled_frames
    indices = [min(int(i * step), total_num_frames - 1) for i in range(sampled_frames)]

    video_metadata = openvino_genai.VideoMetadata()
    video_metadata.fps = cap.get(cv2.CAP_PROP_FPS)
    # Passing frame indices selects those frames within the pipeline and skips model-specific sampling.
    # Leave frames_indices empty to apply model-specific sampling (e.g. for Qwen3-VL).
    video_metadata.frames_indices = indices

    frames = []
    # Stop at the reported count: a container that under-reports it would otherwise make this list grow without limit.
    while len(frames) < total_num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV decodes to BGR; GenAI video inputs are RGB.
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    has_extra_frame = cap.read()[0]
    cap.release()

    if len(frames) != total_num_frames:
        raise RuntimeError(f"Frame count mismatch: expected {total_num_frames}, got {len(frames)}")
    if has_extra_frame:
        raise RuntimeError(f"Video has more frames than its container reports ({total_num_frames}): {path}")

    return Tensor(np.array(frames)), video_metadata


# Qwen3-Omni speech output is 24kHz mono PCM.
SPEECH_SAMPLE_RATE = 24000


def save_speech(decoded_results: openvino_genai.OmniDecodedResults, file_name: str) -> None:
    """Save the first speech waveform to a WAV file. Speech output is optional (talker mode)."""
    if not decoded_results.speech_result.waveforms:
        return
    waveform = np.array(decoded_results.speech_result.waveforms[0].data).reshape(-1)
    sf.write(file_name, waveform, samplerate=SPEECH_SAMPLE_RATE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-Omni multimodal chat")
    parser.add_argument("model_dir", help="Path to the OpenVINO model directory")
    parser.add_argument(
        "image_dir",
        help="Image file or directory with images. Refer to them in the prompt as "
        "<ov_genai_image_0>, <ov_genai_image_1>, ... An untagged prompt gets them prepended.",
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument(
        "--audio",
        action="append",
        default=[],
        metavar="WAV",
        help="Input audio WAV file. Repeat to pass several; refer to them in the prompt as "
        "<ov_genai_audio_0>, <ov_genai_audio_1>, ... An untagged prompt gets them prepended.",
    )
    args = parser.parse_args()

    rgbs = read_images(args.image_dir)

    pipe = openvino_genai.OmniPipeline(args.model_dir, "CPU")

    # Two configs: text_config drives the thinker text decode; talker_speech_config drives the
    # talker + speech output. Speech output is hardcoded on here to show the multimodal path.
    # Set talker_speech_config.return_audio = False to get text-only responses.
    text_config = openvino_genai.GenerationConfig()
    text_config.max_new_tokens = 256

    talker_speech_config = openvino_genai.OmniTalkerSpeechConfig(args.model_dir)
    talker_speech_config.return_audio = True
    # Leaving speaker empty selects the model's default voice. Available voices vary by checkpoint
    # (e.g. MoE exposes "Ethan", "Chelsie", "Aiden", "Cherry"); the full list is in
    # talker_config.speaker_id of the model's config.json.

    video_tensor, video_metadata = read_video(args.video)
    videos = [video_tensor]
    videos_metadata = [video_metadata]
    audios = [load_audio(path) for path in args.audio]

    history = openvino_genai.ChatHistory()
    prompt = input("question:\n")
    history.append({"role": "user", "content": prompt})
    decoded_results = pipe.generate(
        history,
        images=rgbs,
        videos=videos,
        videos_metadata=videos_metadata,
        audios=audios,
        text_config=text_config,
        talker_speech_config=talker_speech_config,
        streamer=streamer,
    )
    history.append({"role": "assistant", "content": decoded_results.texts[0]})
    save_speech(decoded_results, "output_audio_0.wav")

    for turn in itertools.count(start=1):
        try:
            prompt = input("\n----------\nquestion:\n")
        except EOFError:
            break

        history.append({"role": "user", "content": prompt})
        # Media attaches to the turn it is supplied on, so later turns pass none at all and refer
        # back through the history rather than re-sending turn 1's tensors.
        decoded_results = pipe.generate(
            history,
            text_config=text_config,
            talker_speech_config=talker_speech_config,
            streamer=streamer,
        )
        history.append({"role": "assistant", "content": decoded_results.texts[0]})
        save_speech(decoded_results, f"output_audio_{turn}.wav")


if __name__ == "__main__":
    main()
