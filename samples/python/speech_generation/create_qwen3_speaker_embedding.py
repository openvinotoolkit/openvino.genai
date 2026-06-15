#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

import numpy as np
import openvino as ov
import soundfile as sf


def _load_qwen3_speaker_encoder_config(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model config: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    speaker_cfg = cfg.get("speaker_encoder_config")
    if not isinstance(speaker_cfg, dict):
        raise RuntimeError(
            "Model config does not contain 'speaker_encoder_config'. "
            "Use a Qwen3-TTS Base OpenVINO model directory."
        )

    return {
        "sample_rate": int(speaker_cfg.get("sample_rate", 24000)),
        "mel_dim": int(speaker_cfg.get("mel_dim", 128)),
        "enc_dim": int(speaker_cfg.get("enc_dim", 1024)),
        "num_code_groups": int(cfg.get("talker_config", {}).get("num_code_groups", 16)),
    }


def _load_qwen3_speech_tokenizer_config(model_dir: Path) -> dict:
    cfg_path = model_dir / "speech_tokenizer" / "config.json"
    if not cfg_path.exists():
        return {"input_sample_rate": 24000}

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    return {
        "input_sample_rate": int(cfg.get("input_sample_rate", 24000)),
    }


def _extract_log_mel(audio: np.ndarray, sample_rate: int, mel_dim: int) -> np.ndarray:
    # Mirrors the helper extraction settings used for Qwen3-TTS speaker encoder.
    n_fft = 1024
    hop_size = 256
    win_size = 1024
    fmin = 0
    fmax = 12000

    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("This utility requires librosa for mel feature extraction.") from exc

    # Mirror the reflect-padding applied in the Python helper before STFT:
    #   padding = (n_fft - hop_size) // 2  =>  (1024 - 256) // 2 = 384 samples
    padding = (n_fft - hop_size) // 2
    audio_padded = np.pad(audio.astype(np.float32), (padding, padding), mode="reflect")

    mel = librosa.feature.melspectrogram(
        y=audio_padded,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        n_mels=mel_dim,
        fmin=fmin,
        fmax=fmax,
        center=False,
        power=1.0,
    )

    # Dynamic range compression in line with model helper.
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))

    # Model expects [batch, time, mel_dim]
    return mel.T[np.newaxis, :, :].astype(np.float32)


def _to_single_channel(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    if audio.ndim == 2:
        # soundfile returns [frames, channels]
        return np.mean(audio, axis=1).astype(np.float32)
    raise RuntimeError(f"Unsupported audio shape: {audio.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Qwen3-TTS Base speaker embedding (.bin) from reference audio."
    )
    parser.add_argument("model_dir", help="Path to converted Qwen3-TTS Base OpenVINO model directory")
    parser.add_argument("ref_audio", help="Reference audio path (wav/flac/ogg/...) used for voice identity")
    parser.add_argument(
        "--ref_text",
        default="",
        help=(
            "Optional reference transcript. Currently stored only in metadata and not used for embedding extraction."
        ),
    )
    parser.add_argument(
        "--output",
        default="qwen_speaker_embedding.bin",
        help="Output speaker embedding binary file (float32)"
    )
    parser.add_argument(
        "--metadata_output",
        default="",
        help="Optional metadata JSON path (stores ref_audio/ref_text/model info)",
    )
    parser.add_argument(
        "--ref_code_output",
        default="",
        help="Optional .npy output path for speech-tokenizer reference codes used by ICL mode",
    )
    parser.add_argument("--device", default="CPU", help="OpenVINO device for speaker encoder (default: CPU)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    ref_audio_path = Path(args.ref_audio)
    output_path = Path(args.output)

    if not ref_audio_path.exists():
        raise FileNotFoundError(f"Reference audio was not found: {ref_audio_path}")

    speaker_encoder_xml = model_dir / "openvino_speaker_encoder_model.xml"
    if not speaker_encoder_xml.exists():
        raise FileNotFoundError(
            f"Missing speaker encoder model: {speaker_encoder_xml}. "
            "Use a converted Qwen3-TTS Base OpenVINO model."
        )

    cfg = _load_qwen3_speaker_encoder_config(model_dir)
    speech_tok_cfg = _load_qwen3_speech_tokenizer_config(model_dir)
    model_sr = cfg["sample_rate"]
    mel_dim = cfg["mel_dim"]
    enc_dim = cfg["enc_dim"]
    num_code_groups = cfg["num_code_groups"]
    speech_tokenizer_sr = speech_tok_cfg["input_sample_rate"]

    audio, sr = sf.read(str(ref_audio_path), dtype="float32", always_2d=False)
    audio = _to_single_channel(np.asarray(audio, dtype=np.float32))
    if int(sr) != model_sr:
        raise RuntimeError(
            f"Reference audio sample rate is {int(sr)} Hz, but model expects {model_sr} Hz. "
            f"Please provide reference audio already resampled to {model_sr} Hz."
        )

    mels = _extract_log_mel(audio, sample_rate=model_sr, mel_dim=mel_dim)

    core = ov.Core()
    compiled = core.compile_model(str(speaker_encoder_xml), args.device)
    request = compiled.create_infer_request()

    # Most exports use a single input; use index-based I/O for robustness.
    request.set_input_tensor(0, ov.Tensor(mels))
    request.infer()
    embedding = np.array(request.get_output_tensor(0).data, dtype=np.float32).reshape(-1)

    if embedding.size != enc_dim:
        raise RuntimeError(
            f"Unexpected embedding size: got {embedding.size}, expected {enc_dim}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    embedding.tofile(str(output_path))

    print("[Info] Qwen3 speaker embedding created")
    print(f"[Info] Saved: {output_path}")
    print(f"[Info] Embedding shape (flattened): [{embedding.size}]")
    print("[Info] Compatible tensor forms for Base sample: [D], [1,D], [1,1,D]")

    if args.ref_code_output:
        if int(sr) != speech_tokenizer_sr:
            raise RuntimeError(
                f"Reference audio sample rate is {int(sr)} Hz, but speech tokenizer expects {speech_tokenizer_sr} Hz. "
                f"Please provide reference audio already resampled to {speech_tokenizer_sr} Hz."
            )

        speech_tokenizer_encoder_xml = model_dir / "speech_tokenizer" / "openvino_speech_tokenizer_encoder_model.xml"
        if not speech_tokenizer_encoder_xml.exists():
            raise FileNotFoundError(
                f"Missing speech tokenizer encoder model: {speech_tokenizer_encoder_xml}. "
                "Cannot export ref_code for ICL mode from this model directory."
            )

        tok_compiled = core.compile_model(str(speech_tokenizer_encoder_xml), args.device)
        tok_request = tok_compiled.create_infer_request()
        # Speech tokenizer encoder expects [batch, channels, time] => [1, 1, T].
        audio_tensor = ov.Tensor(np.asarray(audio, dtype=np.float32).reshape(1, 1, -1))
        tok_request.set_input_tensor(0, audio_tensor)
        tok_request.infer()
        ref_code = np.array(tok_request.get_output_tensor(0).data, dtype=np.int64)

        # Normalize ref_code layout to [1, T, G] where G == num_code_groups.
        if ref_code.ndim == 3:
            if ref_code.shape[-1] == num_code_groups:
                # Already [1, T, G].
                pass
            elif ref_code.shape[1] == num_code_groups:
                # Convert [1, G, T] -> [1, T, G].
                ref_code = np.transpose(ref_code, (0, 2, 1))
            else:
                raise RuntimeError(
                    f"Unexpected ref_code shape {ref_code.shape}; cannot locate num_code_groups={num_code_groups}."
                )
        elif ref_code.ndim == 2:
            if ref_code.shape[-1] == num_code_groups:
                # Already [T, G].
                pass
            elif ref_code.shape[0] == num_code_groups:
                # Convert [G, T] -> [T, G].
                ref_code = ref_code.T
            else:
                raise RuntimeError(
                    f"Unexpected ref_code shape {ref_code.shape}; cannot locate num_code_groups={num_code_groups}."
                )
        else:
            raise RuntimeError(f"Unexpected ref_code rank: {ref_code.ndim}; expected 2 or 3")

        # Ensure C-order so np.save writes fortran_order=False.
        ref_code = np.ascontiguousarray(ref_code, dtype=np.int64)

        ref_code_path = Path(args.ref_code_output)
        ref_code_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(ref_code_path, ref_code)
        print(f"[Info] Saved ref_code: {ref_code_path}")
        print(f"[Info] ref_code shape: {list(ref_code.shape)}")

    if args.metadata_output:
        metadata_path = Path(args.metadata_output)
        metadata = {
            "model_dir": str(model_dir),
            "ref_audio": str(ref_audio_path),
            "ref_text": args.ref_text,
            "speaker_encoder_sample_rate": model_sr,
            "speaker_encoder_mel_dim": mel_dim,
            "speaker_embedding_dim": enc_dim,
            "speech_tokenizer_input_sample_rate": speech_tokenizer_sr,
            "ref_code_output": args.ref_code_output,
            "output_embedding": str(output_path),
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=True, indent=2)
        print(f"[Info] Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
