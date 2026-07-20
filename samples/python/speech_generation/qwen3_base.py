#!/usr/bin/env python3
# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Qwen3-TTS Base (voice clone) sample.
#
# Base models clone a voice from a short reference recording. There are two
# cloning modes, selected automatically from the inputs you provide:
#
#   1. x-vector mode (fast, identity only)
#        Provide reference audio (or a pre-saved speaker embedding). Only the
#        speaker embedding is used. --ref_text is NOT required.
#
#   2. ICL mode (in-context learning, higher fidelity)
#        Provide reference audio AND its transcript (--ref_text). The pipeline
#        additionally conditions on the reference speech codes.
#
# Reusing a reference prompt
# --------------------------
# Extracting the speaker embedding and reference codes from audio is the
# expensive part of cloning. To avoid recomputing them for every generation,
# clone once from reference audio and save the artifacts returned on the result
# (speaker_embedding and voice_clone_ref_codec_ids) with
# --save_speaker_embedding_file_path / --save_ref_codec_ids_file_path. Later runs
# can pass them back via --speaker_embedding_file_path and
# --ref_codec_ids_file_path to skip the reference-audio encoder entirely.

import argparse

import numpy as np
import openvino as ov
import openvino_genai
import soundfile as sf


def _load_speaker_embedding(file_path, shape):
    """Load a flat float32 binary and reshape it to the model's expected shape."""
    data = np.fromfile(file_path, dtype="<f4")
    if data.size == 0:
        raise RuntimeError(f"Speaker embedding file is empty: {file_path}")
    return ov.Tensor(data.reshape(shape))


def _save_speaker_embedding(file_path, tensor):
    """Save a speaker embedding tensor as a flat float32 binary."""
    np.array(tensor.data, dtype="<f4").reshape(-1).tofile(file_path)


def _load_reference_codes(file_path):
    """Load reference codec ids from the sample's self-describing binary format.

    Layout: [int64 rank][int64 dim_0] ... [int64 dim_{rank-1}][int64 payload ...]
    (little-endian). Matches the C++ qwen3_base sample so files are interchangeable.
    """
    with open(file_path, "rb") as f:
        rank = int(np.frombuffer(f.read(8), dtype="<i8")[0])
        if rank <= 0 or rank > 8:
            raise RuntimeError(f"Invalid reference codes file: {file_path}")
        dims = np.frombuffer(f.read(8 * rank), dtype="<i8")
        payload = np.frombuffer(f.read(), dtype="<i8").astype(np.int64)
    shape = tuple(int(d) for d in dims)
    return ov.Tensor(np.ascontiguousarray(payload.reshape(shape)))


def _save_reference_codes(file_path, tensor):
    """Save reference codec ids in the sample's self-describing binary format."""
    codes = np.ascontiguousarray(np.array(tensor.data), dtype="<i8")
    with open(file_path, "wb") as f:
        f.write(np.array([codes.ndim], dtype="<i8").tobytes())
        f.write(np.array(codes.shape, dtype="<i8").tobytes())
        f.write(codes.tobytes())


def _load_ref_audio(file_path):
    """Read a WAV into a mono float32 waveform tensor. Requires 24000 Hz (no resampling)."""
    audio, sr = sf.read(file_path, dtype="float32", always_2d=False)
    if int(sr) != 24000:
        raise RuntimeError(
            f"Reference audio sample rate is {int(sr)} Hz, but 24000 Hz is required. "
            "OV GenAI does not resample reference audio."
        )
    if audio.ndim == 2:
        audio = audio.mean(axis=1).astype(np.float32)
    return ov.Tensor(np.ascontiguousarray(audio, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Base voice cloning")
    parser.add_argument("model_dir", help="Path to the Qwen3-TTS Base OpenVINO model directory")
    parser.add_argument("text", help="Target text to synthesize")
    parser.add_argument("--ref_audio_wav_path", default=None, help="Reference audio WAV (mono/stereo, 24000 Hz)")
    parser.add_argument("--speaker_embedding_file_path", default=None, help="Pre-saved speaker embedding (.bin)")
    parser.add_argument("--ref_text", default="", help="Reference transcript; enables ICL mode when set")
    parser.add_argument("--ref_codec_ids_file_path", default=None, help="Pre-saved reference codec ids (.bin)")
    parser.add_argument("--save_speaker_embedding_file_path", default=None, help="Where to save the speaker embedding")
    parser.add_argument("--save_ref_codec_ids_file_path", default=None, help="Where to save the reference codec ids")
    parser.add_argument("--language", default="", help="Optional language (for example: english). Omit for auto.")
    parser.add_argument("--device", default="CPU", help="Device to run the model on (default: CPU)")
    args = parser.parse_args()

    # A Base clone always needs a voice source: either reference audio (from which the
    # pipeline extracts the identity) or a pre-saved speaker embedding.
    if not args.ref_audio_wav_path and not args.speaker_embedding_file_path:
        raise RuntimeError(
            "Qwen3-TTS Base requires --ref_audio_wav_path or --speaker_embedding_file_path."
        )

    # Pre-saved reference codes are only meaningful in ICL mode, which requires the transcript.
    if args.ref_codec_ids_file_path and not args.ref_text.strip():
        raise RuntimeError("--ref_text is required when --ref_codec_ids_file_path is provided (ICL mode).")

    pipe = openvino_genai.Text2SpeechPipeline(args.model_dir, args.device)

    generation_properties = {}
    if args.language:
        generation_properties["language"] = args.language

    # Reference audio: the pipeline internally derives the speaker embedding and,
    # in ICL mode, the reference codes from this waveform.
    if args.ref_audio_wav_path:
        generation_properties["voice_clone_ref_audio"] = _load_ref_audio(args.ref_audio_wav_path)

    # ICL mode is enabled by providing the reference transcript. Reference codes are
    # either supplied directly (reuse) or extracted from the reference audio.
    if args.ref_text.strip():
        generation_properties["voice_clone_ref_text"] = args.ref_text
    if args.ref_codec_ids_file_path:
        generation_properties["voice_clone_ref_codec_ids"] = _load_reference_codes(args.ref_codec_ids_file_path)

    icl_mode = bool(args.ref_text.strip())
    print(f"[Info] Qwen3-TTS Base voice clone ({'ICL' if icl_mode else 'x-vector'} mode).")

    # A pre-saved speaker embedding is passed as the pipeline's speaker_embedding
    # argument; otherwise the pipeline extracts it from the reference audio.
    speaker_embedding = None
    if args.speaker_embedding_file_path:
        speaker_embedding = _load_speaker_embedding(
            args.speaker_embedding_file_path, pipe.get_speaker_embedding_shape()
        )

    result = pipe.generate(args.text, speaker_embedding, **generation_properties)

    assert len(result.speeches) == 1, "Expected only one waveform for the requested input text"
    speech_data = np.array(result.speeches[0].data).reshape(-1)
    output_file_name = "output_audio.wav"
    sf.write(output_file_name, speech_data, samplerate=result.output_sample_rate)
    print(f'[Info] Text successfully converted to audio file "{output_file_name}".')

    # Persist the resolved clone artifacts for reuse. Passing them back on a later run
    # (via --speaker_embedding_file_path / --ref_codec_ids_file_path) skips the
    # reference-audio encoding step.
    if args.save_speaker_embedding_file_path:
        if not result.speaker_embedding:
            raise RuntimeError("No speaker embedding was produced to save. Provide --ref_audio_wav_path.")
        _save_speaker_embedding(args.save_speaker_embedding_file_path, result.speaker_embedding)
        print(f'[Info] Saved speaker embedding to "{args.save_speaker_embedding_file_path}".')
    if args.save_ref_codec_ids_file_path:
        if not result.voice_clone_ref_codec_ids:
            raise RuntimeError(
                "No reference codes were produced to save. ICL mode (--ref_text) with "
                "--ref_audio_wav_path is required."
            )
        _save_reference_codes(args.save_ref_codec_ids_file_path, result.voice_clone_ref_codec_ids)
        print(f'[Info] Saved reference codes to "{args.save_ref_codec_ids_file_path}".')

    perf_metrics = result.perf_metrics
    if perf_metrics.m_evaluated:
        print("\n\n=== Performance Summary ===")
        print("Throughput              : ", perf_metrics.throughput.mean, " samples/sec.")
        print("Total Generation Time   : ", perf_metrics.generate_duration.mean / 1000.0, " sec.")


if "__main__" == __name__:
    main()
