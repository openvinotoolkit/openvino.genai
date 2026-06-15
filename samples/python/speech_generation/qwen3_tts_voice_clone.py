#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse

import numpy as np
import openvino as ov
import openvino_genai
import soundfile as sf


def _load_embedding(file_path: str, expected_shape):
    data = np.fromfile(file_path, dtype=np.float32)
    if data.size == 0:
        raise RuntimeError(f"Speaker embedding file is empty: {file_path}")
    return ov.Tensor(data.reshape(expected_shape))


def _load_ref_code_tensor(file_path: str):
    ref_code = np.load(file_path)
    num_code_groups = 16

    # Normalize ref_code to [T, G] or [1, T, G].
    if ref_code.ndim == 3 and ref_code.shape[1] == num_code_groups and ref_code.shape[-1] != num_code_groups:
        ref_code = np.transpose(ref_code, (0, 2, 1))
    elif ref_code.ndim == 2 and ref_code.shape[0] == num_code_groups and ref_code.shape[-1] != num_code_groups:
        ref_code = ref_code.T

    if ref_code.ndim == 2:
        # [T, G]
        pass
    elif ref_code.ndim == 3 and ref_code.shape[0] == 1:
        # [1, T, G]
        pass
    else:
        raise RuntimeError(
            f"Expected ref_code shape [T, G] or [1, T, G], got {ref_code.shape}"
        )

    if ref_code.dtype != np.int64:
        ref_code = ref_code.astype(np.int64)

    return ov.Tensor(ref_code)


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS Base voice cloning via OV GenAI generate(...) properties"
    )
    parser.add_argument("model_dir", help="Path to Qwen3-TTS Base OpenVINO model directory")
    parser.add_argument("text", help="Target text to synthesize")
    parser.add_argument(
        "--ref_speaker_embedding_file_path",
        required=True,
        help="Path to float32 speaker embedding .bin file",
    )
    parser.add_argument(
        "--ref_code_file_path",
        required=False,
        default=None,
        help="Path to reference codec ids .npy file ([T, G] or [1, T, G]); required for ICL mode",
    )
    parser.add_argument(
        "--ref_text",
        default="",
        help="Reference transcript used for ICL mode",
    )
    parser.add_argument(
        "--x_vector_only_mode",
        action="store_true",
        help="Use embedding-only voice cloning (no ICL prompt conditioning)",
    )
    parser.add_argument("--language", default="english", help="Generation language")
    parser.add_argument("--device", default="CPU", help="OpenVINO device (default: CPU)")
    parser.add_argument("--output", default="output_audio.wav", help="Output wav path")
    args = parser.parse_args()

    pipe = openvino_genai.Text2SpeechPipeline(args.model_dir, args.device)

    speaker_embedding = _load_embedding(
        args.ref_speaker_embedding_file_path, pipe.get_speaker_embedding_shape()
    )

    generation_properties = {
        "language": args.language,
        "qwen_x_vector_only_mode": bool(args.x_vector_only_mode),
    }

    if not args.x_vector_only_mode:
        if not args.ref_text.strip():
            raise RuntimeError("--ref_text is required when --x_vector_only_mode is not set")
        if not args.ref_code_file_path:
            raise RuntimeError("--ref_code_file_path is required when --x_vector_only_mode is not set")
        generation_properties["qwen_ref_text"] = args.ref_text
        generation_properties["qwen_ref_code"] = _load_ref_code_tensor(args.ref_code_file_path)

    result = pipe.generate(args.text, speaker_embedding, **generation_properties)

    if len(result.speeches) != 1:
        raise RuntimeError(f"Expected one output waveform, got {len(result.speeches)}")

    speech_data = np.array(result.speeches[0].data).reshape(-1)
    sf.write(args.output, speech_data, samplerate=result.output_sample_rate)

    print(f"[Info] Saved: {args.output}")
    print(f"[Info] Sample rate: {result.output_sample_rate}")


if __name__ == "__main__":
    main()
