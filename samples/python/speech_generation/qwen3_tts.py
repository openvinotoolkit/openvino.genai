#!/usr/bin/env python3
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import openvino as ov
import openvino_genai
import soundfile as sf


def _load_speaker_embedding(file_path, shape):
    data = np.fromfile(file_path, dtype="<f4")
    if data.size == 0:
        raise RuntimeError(f"Speaker embedding file is empty: {file_path}")
    return ov.Tensor(data.reshape(shape))


def _save_speaker_embedding(file_path, tensor):
    np.array(tensor.data, dtype="<f4").reshape(-1).tofile(file_path)


def _load_reference_codes(file_path):
    # Layout: [int64 rank][int64 dim_0] ... [int64 dim_{rank-1}][int64 payload ...]
    with open(file_path, "rb") as f:
        rank = int(np.frombuffer(f.read(8), dtype="<i8")[0])
        if rank <= 0 or rank > 8:
            raise RuntimeError(f"Invalid reference codes file: {file_path}")
        dims = np.frombuffer(f.read(8 * rank), dtype="<i8")
        payload = np.frombuffer(f.read(), dtype="<i8").astype(np.int64)
    shape = tuple(int(d) for d in dims)
    return ov.Tensor(np.ascontiguousarray(payload.reshape(shape)))


def _save_reference_codes(file_path, tensor):
    codes = np.ascontiguousarray(np.array(tensor.data), dtype="<i8")
    with open(file_path, "wb") as f:
        f.write(np.array([codes.ndim], dtype="<i8").tobytes())
        f.write(np.array(codes.shape, dtype="<i8").tobytes())
        f.write(codes.tobytes())


def _load_ref_audio(file_path):
    audio, sr = sf.read(file_path, dtype="float32", always_2d=False)
    if int(sr) != 24000:
        raise RuntimeError(
            f"Reference audio sample rate is {int(sr)} Hz, but 24000 Hz is required. "
            "OV GenAI does not resample reference audio."
        )
    if audio.ndim == 2:
        audio = audio.mean(axis=1).astype(np.float32)
    return ov.Tensor(np.ascontiguousarray(audio, dtype=np.float32))


def _write_audio_and_perf(result, output_file_path):
    assert len(result.speeches) == 1, "Expected only one waveform for the requested input text"
    speech_data = np.array(result.speeches[0].data).reshape(-1)
    sf.write(output_file_path, speech_data, samplerate=result.output_sample_rate)
    print(f'[Info] Text successfully converted to audio file "{output_file_path}".')

    perf_metrics = result.perf_metrics
    if perf_metrics.m_evaluated:
        print("\n\n=== Performance Summary ===")
        print("Throughput              : ", perf_metrics.throughput.mean, " samples/sec.")
        print("Total Generation Time   : ", perf_metrics.generate_duration.mean / 1000.0, " sec.")


def _pipeline_kwargs_for_device(device):
    device_upper = device.upper()
    kwargs = {}

    # Persist compiled artifacts on accelerators to reduce recompilation costs.
    if "NPU" in device_upper or "GPU" in device_upper:
        kwargs["CACHE_DIR"] = "qwen3_tts_cache_dir"

    return kwargs


def _build_parser():
    parser = argparse.ArgumentParser(description="Qwen3-TTS sample (Base, CustomVoice, VoiceDesign)")
    subparsers = parser.add_subparsers(dest="variant", required=True)

    base = subparsers.add_parser("base", help="Qwen3-TTS Base voice cloning")
    base.add_argument("model_dir", help="Path to the Qwen3-TTS Base OpenVINO model directory")
    base.add_argument("text", help="Target text to synthesize")
    base.add_argument("--ref_audio_wav_path", default=None, help="Reference audio WAV (mono/stereo, 24000 Hz)")
    base.add_argument("--speaker_embedding_file_path", default=None, help="Pre-saved speaker embedding (.bin)")
    base.add_argument("--ref_text", default="", help="Reference transcript; enables ICL mode when set")
    base.add_argument("--ref_codec_ids_file_path", default=None, help="Pre-saved reference codec ids (.bin)")
    base.add_argument("--save_speaker_embedding_file_path", default=None, help="Where to save the speaker embedding")
    base.add_argument("--save_ref_codec_ids_file_path", default=None, help="Where to save the reference codec ids")
    base.add_argument("--language", default="", help="Optional language (for example: english). Omit for auto.")
    base.add_argument("--device", default="CPU", help="Device to run the model on (default: CPU)")
    base.add_argument("--max_new_tokens", type=int, default=None, help="Optional cap for generated tokens")
    base.add_argument("--output_wav_path", default="output_audio.wav", help="Output WAV path")

    customvoice = subparsers.add_parser("customvoice", help="Qwen3-TTS CustomVoice generation")
    customvoice.add_argument("model_dir", help="Path to the Qwen3-TTS CustomVoice OpenVINO model directory")
    customvoice.add_argument("text", help="Input text to synthesize")
    customvoice.add_argument("--speaker", required=True, help="Built-in speaker name (for example: ryan)")
    customvoice.add_argument("--language", default="", help="Optional language (for example: english). Omit for auto.")
    customvoice.add_argument("--instruct", default="", help="Optional natural-language style instruction")
    customvoice.add_argument("--device", default="CPU", help="Device to run the model on (default: CPU)")
    customvoice.add_argument("--max_new_tokens", type=int, default=None, help="Optional cap for generated tokens")
    customvoice.add_argument("--output_wav_path", default="output_audio.wav", help="Output WAV path")

    voice_design = subparsers.add_parser("voice-design", help="Qwen3-TTS VoiceDesign generation")
    voice_design.add_argument("model_dir", help="Path to the Qwen3-TTS VoiceDesign OpenVINO model directory")
    voice_design.add_argument("text", help="Input text to synthesize")
    voice_design.add_argument("--instruct", required=True, help="Natural-language description of the target voice")
    voice_design.add_argument("--language", default="", help="Optional language (for example: english). Omit for auto.")
    voice_design.add_argument("--device", default="CPU", help="Device to run the model on (default: CPU)")
    voice_design.add_argument("--max_new_tokens", type=int, default=None, help="Optional cap for generated tokens")
    voice_design.add_argument("--output_wav_path", default="output_audio.wav", help="Output WAV path")

    return parser


def _run_base(args):
    if not args.ref_audio_wav_path and not args.speaker_embedding_file_path:
        raise RuntimeError("Qwen3-TTS Base requires --ref_audio_wav_path or --speaker_embedding_file_path.")

    if args.ref_codec_ids_file_path and not args.ref_text.strip():
        raise RuntimeError("--ref_text is required when --ref_codec_ids_file_path is provided (ICL mode).")

    pipe = openvino_genai.Text2SpeechPipeline(
        args.model_dir,
        args.device,
        **_pipeline_kwargs_for_device(args.device),
    )
    generation_properties = {}
    if args.language:
        generation_properties["language"] = args.language
    if args.ref_audio_wav_path:
        generation_properties["ref_audio"] = _load_ref_audio(args.ref_audio_wav_path)
    if args.ref_text.strip():
        generation_properties["ref_text"] = args.ref_text
    if args.ref_codec_ids_file_path:
        generation_properties["ref_codec_ids"] = _load_reference_codes(args.ref_codec_ids_file_path)
    if args.max_new_tokens is not None:
        generation_properties["max_new_tokens"] = args.max_new_tokens

    icl_mode = bool(args.ref_text.strip())
    print(f"[Info] Qwen3-TTS Base voice clone ({'ICL' if icl_mode else 'x-vector'} mode).")

    speaker_embedding = None
    if args.speaker_embedding_file_path:
        speaker_embedding = _load_speaker_embedding(
            args.speaker_embedding_file_path, pipe.get_speaker_embedding_shape()
        )

    result = pipe.generate(args.text, speaker_embedding, **generation_properties)
    _write_audio_and_perf(result, args.output_wav_path)

    if args.save_speaker_embedding_file_path:
        if not result.speaker_embedding:
            raise RuntimeError("No speaker embedding was produced to save. Provide --ref_audio_wav_path.")
        _save_speaker_embedding(args.save_speaker_embedding_file_path, result.speaker_embedding)
        print(f'[Info] Saved speaker embedding to "{args.save_speaker_embedding_file_path}".')
    if args.save_ref_codec_ids_file_path:
        if not result.ref_codec_ids:
            raise RuntimeError(
                "No reference codes were produced to save. ICL mode (--ref_text) with --ref_audio_wav_path is required."
            )
        _save_reference_codes(args.save_ref_codec_ids_file_path, result.ref_codec_ids)
        print(f'[Info] Saved reference codes to "{args.save_ref_codec_ids_file_path}".')


def _run_customvoice(args):
    pipe = openvino_genai.Text2SpeechPipeline(
        args.model_dir,
        args.device,
        **_pipeline_kwargs_for_device(args.device),
    )
    generation_properties = {"speaker": args.speaker}
    if args.language:
        generation_properties["language"] = args.language
    if args.instruct:
        generation_properties["instruct"] = args.instruct
    if args.max_new_tokens is not None:
        generation_properties["max_new_tokens"] = args.max_new_tokens

    result = pipe.generate(args.text, None, **generation_properties)
    _write_audio_and_perf(result, args.output_wav_path)


def _run_voice_design(args):
    pipe = openvino_genai.Text2SpeechPipeline(
        args.model_dir,
        args.device,
        **_pipeline_kwargs_for_device(args.device),
    )
    generation_properties = {"instruct": args.instruct}
    if args.language:
        generation_properties["language"] = args.language
    if args.max_new_tokens is not None:
        generation_properties["max_new_tokens"] = args.max_new_tokens

    result = pipe.generate(args.text, None, **generation_properties)
    _write_audio_and_perf(result, args.output_wav_path)


def main():
    args = _build_parser().parse_args()
    if args.variant == "base":
        _run_base(args)
    elif args.variant == "customvoice":
        _run_customvoice(args)
    else:
        _run_voice_design(args)


if "__main__" == __name__:
    main()
