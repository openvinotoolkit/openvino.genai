#!/usr/bin/env python3
# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-Omni multimodal chat sample.

Demonstrates:
- Text + image input -> text output
- Text + audio input -> text output
- Text -> text + speech output
- Multi-turn chat with mixed modalities

Usage:
    # Text-only:
    python qwen3_omni_chat.py model_dir

    # With image:
    python qwen3_omni_chat.py model_dir --image cat.jpg

    # With audio input:
    python qwen3_omni_chat.py model_dir --audio recording.wav

    # With speech output:
    python qwen3_omni_chat.py model_dir --enable-speech --speaker f245 --output-wav output.wav

    # With streaming speech output (audio chunks arrive during generation):
    python qwen3_omni_chat.py model_dir --enable-speech --stream-audio --audio-chunk-frames 5

    # Full omni (audio input + speech output):
    python qwen3_omni_chat.py model_dir --audio recording.wav --enable-speech
"""

import argparse
import sys

import numpy as np
import openvino as ov
import openvino_genai as ov_genai


def load_image(image_path: str) -> ov.Tensor:
    """Load image from file and convert to OpenVINO tensor."""
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image, dtype=np.uint8)
    # Add batch dimension if needed: [H, W, C] -> [1, H, W, C]
    if image_array.ndim == 3:
        image_array = np.expand_dims(image_array, axis=0)
    return ov.Tensor(image_array)


def load_audio(audio_path: str, target_sr: int = 16000) -> ov.Tensor:
    """Load audio from WAV file and convert to float32 tensor at target sample rate."""
    try:
        import soundfile as sf
    except ImportError:
        print("Error: soundfile package required for audio input. Install with: pip install soundfile")
        sys.exit(1)

    audio_data, sample_rate = sf.read(audio_path, dtype="float32")

    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    if sample_rate != target_sr:
        try:
            import librosa

            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
        except ImportError:
            print(
                f"Warning: Audio sample rate is {sample_rate}Hz, expected {target_sr}Hz. "
                f"Install librosa for resampling: pip install librosa"
            )

    return ov.Tensor(audio_data.astype(np.float32))


def save_wav(tensor: ov.Tensor, output_path: str, sample_rate: int = 24000) -> None:
    """Save waveform tensor to WAV file."""
    try:
        import soundfile as sf
    except ImportError:
        print("Error: soundfile package required. Install with: pip install soundfile")
        return

    waveform = np.array(tensor.data).flatten()
    sf.write(output_path, waveform, sample_rate)
    print(f"Speech saved to: {output_path}")


def streamer_callback(text: str) -> bool:
    """Stream text tokens to stdout."""
    print(text, end="", flush=True)
    return False  # Don't stop generation


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-Omni multimodal chat")
    parser.add_argument("model_dir", type=str, help="Path to OpenVINO model directory")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--audio", type=str, default=None, help="Path to input audio WAV file")
    parser.add_argument("--enable-speech", action="store_true", help="Enable speech output generation")
    parser.add_argument("--speaker", type=str, default="m02", help="Speaker name for speech (e.g., m02, f245)")
    parser.add_argument("--output-wav", type=str, default="output.wav", help="Output WAV file path")
    parser.add_argument(
        "--stream-audio", action="store_true", help="Enable audio streaming (receive chunks during generation)"
    )
    parser.add_argument(
        "--audio-chunk-frames",
        type=int,
        default=5,
        help="Number of codec frames per streaming chunk (default 5 = ~400ms)",
    )
    parser.add_argument("--device", type=str, default="CPU", help="Inference device")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--system-prompt", type=str, default="", help="System prompt for speech language control")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_dir}")
    pipe = ov_genai.VLMPipeline(args.model_dir, args.device)

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens

    if args.enable_speech:
        config.return_audio = True
        config.speaker = args.speaker

    audio_chunks: list = []

    def audio_streamer_callback(audio_chunk: ov.Tensor) -> ov_genai.StreamingStatus:
        """Receive audio chunks during speech generation."""
        chunk_data = np.array(audio_chunk.data).flatten()
        audio_chunks.append(chunk_data)
        total_samples = sum(len(c) for c in audio_chunks)
        duration_ms = total_samples / 24  # 24kHz = 24 samples/ms
        print(f"\r  [audio: {len(audio_chunks)} chunks, {duration_ms:.0f}ms]", end="", flush=True)
        return ov_genai.StreamingStatus.RUNNING

    generate_kwargs: dict = {}

    if args.stream_audio and args.enable_speech:
        generate_kwargs["audio_streamer"] = audio_streamer_callback
        config.audio_chunk_frames = args.audio_chunk_frames

    if args.image:
        print(f"Loading image: {args.image}")
        image_tensor = load_image(args.image)
        generate_kwargs["images"] = [image_tensor]

    if args.audio:
        print(f"Loading audio: {args.audio}")
        audio_tensor = load_audio(args.audio)
        generate_kwargs["audios"] = [audio_tensor]

    pipe.start_chat(args.system_prompt)
    print("\nQwen3-Omni Chat (type 'quit' to exit)")
    print("-" * 40)

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if prompt.lower() in ("quit", "exit", "q"):
            break

        if not prompt:
            continue

        print("Assistant: ", end="", flush=True)
        if args.enable_speech:
            # Speech generation happens after text, can take minutes on CPU
            print("[speech will be generated after text completes]", file=sys.stderr, flush=True)
        result = pipe.generate(
            prompt,
            generation_config=config,
            streamer=streamer_callback,
            **generate_kwargs,
        )
        print()  # Newline after streamed output

        # Save speech output if available
        if args.enable_speech:
            if args.stream_audio and audio_chunks:
                # Concatenate streamed chunks and save
                print()  # Newline after streaming progress
                waveform = np.concatenate(audio_chunks)
                try:
                    import soundfile as sf

                    sf.write(args.output_wav, waveform, 24000)
                    print(f"Streamed speech saved to: {args.output_wav} ({len(audio_chunks)} chunks)")
                except ImportError:
                    print("Error: soundfile package required. Install with: pip install soundfile")
                audio_chunks.clear()
            elif result.speech_outputs:
                save_wav(result.speech_outputs[0], args.output_wav)

        # Clear media after first turn (subsequent turns are text-only)
        generate_kwargs.pop("images", None)
        generate_kwargs.pop("audios", None)

    pipe.finish_chat()
    print("\nChat ended.")


if __name__ == "__main__":
    main()
