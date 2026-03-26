# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import math
import re
import string
from dataclasses import asdict, dataclass
from typing import Any, Optional

import librosa
import numpy as np
import soundfile as sf


DEFAULT_SR = 16000
DEFAULT_WHISPER_MODEL = "base.en"


@dataclass
class Scores:
    # Speaker = same voice?
    speaker: Optional[float]
    # Content = same words?
    content: Optional[float]
    # Duration = similar overall utterance length / pacing?
    duration: Optional[float]
    # Acoustic = similar overall spectral character / bandwidth?
    acoustic: Optional[float]
    overall: Optional[float]


@dataclass
class ScoringConfig:
    # SpeechBrain verification score -> normalized speaker score.
    # These are heuristic anchors, not universal constants.
    speaker_bad: float = 0.10
    speaker_good: float = 0.98

    # Content score uses normalized WER/CER and is intentionally somewhat forgiving,
    # since ASR can disagree on punctuation, hyphens, split words, etc.
    content_wer_bad: float = 0.35
    content_cer_bad: float = 0.15
    content_wer_weight: float = 0.60
    content_cer_weight: float = 0.40

    # Combine target-vs-expected correctness with target-vs-reference parity.
    # Reference parity gets more weight so expected-text formatting quirks do not dominate.
    content_expected_weight: float = 0.25
    content_reference_weight: float = 0.75

    # Duration = relative difference in overall clip length.
    # ~1% is very close; ~20% is a clearly noticeable drift.
    duration_diff_good: float = 0.01
    duration_diff_bad: float = 0.20

    # Acoustic = compare coarse spectral centroid / rolloff summaries.
    acoustic_centroid_diff_good: float = 0.05
    acoustic_centroid_diff_bad: float = 0.35
    acoustic_rolloff_diff_good: float = 0.05
    acoustic_rolloff_diff_bad: float = 0.35
    acoustic_centroid_weight: float = 0.50
    acoustic_rolloff_weight: float = 0.50

    # Overall weighting: content matters most, then speaker, then acoustic, then duration.
    overall_content_weight: float = 0.35
    overall_speaker_weight: float = 0.30
    overall_acoustic_weight: float = 0.25
    overall_duration_weight: float = 0.10


def normalize_text(text: str) -> str:
    """Normalize text for forgiving transcript comparison."""
    text = text.lower().strip()
    text = re.sub(r"[-‐‑‒–—]+", " ", text)  # treat hyphens/dashes as separators
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_float(x: Any) -> Optional[float]:
    try:
        x = float(x)
        return None if (math.isnan(x) or math.isinf(x)) else x
    except Exception:
        return None


def linear_distance_score(x: Optional[float], good: float, bad: float) -> Optional[float]:
    """Lower-is-better metric -> 0..1 score."""
    if x is None:
        return None
    if x <= good:
        return 1.0
    if x >= bad:
        return 0.0
    return 1.0 - (x - good) / (bad - good)


def linear_similarity_score(x: Optional[float], bad: float, good: float) -> Optional[float]:
    """Higher-is-better metric -> 0..1 score."""
    if x is None:
        return None
    if x <= bad:
        return 0.0
    if x >= good:
        return 1.0
    return (x - bad) / (good - bad)


def weighted_mean(items: list[tuple[Optional[float], float]]) -> Optional[float]:
    vals = [(v, w) for v, w in items if v is not None and w > 0]
    if not vals:
        return None
    total_w = sum(w for _, w in vals)
    return sum(v * w for v, w in vals) / total_w


def format_float(x: Optional[float], digits: int = 3) -> str:
    return "None" if x is None else f"{x:.{digits}f}"


def _log(verbose: bool, msg: str = "") -> None:
    if verbose:
        print(msg)


def load_audio_mono(path: str, sr: int) -> tuple[np.ndarray, int]:
    audio, file_sr = sf.read(path, always_2d=False)
    if getattr(audio, "ndim", 1) == 2:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        file_sr = sr
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio, file_sr


def duration_s(audio: np.ndarray, sr: int) -> float:
    return float(len(audio) / sr)


def relative_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return abs(a - b) / max(abs(b), 1e-8)


class TTSSimilarityEvaluator:
    def __init__(
        self,
        sample_rate: int = DEFAULT_SR,
        whisper_model: str = DEFAULT_WHISPER_MODEL,
        whisper_device: str = "auto",
        whisper_compute_type: str = "default",
        scoring_config: Optional[ScoringConfig] = None,
    ):
        self.sample_rate = sample_rate
        self.whisper_model_name = whisper_model
        self.whisper_device = whisper_device
        self.whisper_compute_type = whisper_compute_type
        self.cfg = scoring_config or ScoringConfig()
        self._whisper = None
        self._speaker_model = None

    @property
    def whisper(self):
        if self._whisper is None:
            from faster_whisper import WhisperModel

            self._whisper = WhisperModel(
                self.whisper_model_name,
                device=self.whisper_device,
                compute_type=self.whisper_compute_type,
            )
        return self._whisper

    @property
    def speaker_model(self):
        if self._speaker_model is None:
            import torchaudio

            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: []
            if not hasattr(torchaudio, "set_audio_backend"):
                torchaudio.set_audio_backend = lambda _backend: None

            from speechbrain.inference.speaker import SpeakerRecognition

            self._speaker_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
            )
            self._speaker_model.eval()
        return self._speaker_model

    def transcribe(self, path: str, language: Optional[str]) -> str:
        segments, _info = self.whisper.transcribe(path, language=language, vad_filter=False)
        return " ".join(seg.text.strip() for seg in segments).strip()

    def compute_speaker_similarity(
        self, target_path: str, reference_path: str
    ) -> tuple[Optional[float], Optional[int], Optional[str]]:
        try:
            import torch

            ref_tensor, _ = load_audio_mono(reference_path, 16000)
            tgt_tensor, _ = load_audio_mono(target_path, 16000)
            ref_tensor = torch.tensor(ref_tensor, dtype=torch.float32).unsqueeze(0)
            tgt_tensor = torch.tensor(tgt_tensor, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                emb_ref = self.speaker_model.encode_batch(ref_tensor).reshape(1, -1)
                emb_tgt = self.speaker_model.encode_batch(tgt_tensor).reshape(1, -1)
                score = float(torch.nn.functional.cosine_similarity(emb_ref, emb_tgt, dim=1).item())
            pred = int(score >= 0.25)
            return score, pred, None
        except Exception as e:
            return None, None, f"compute_speaker_similarity: {e}"

    def evaluate(
        self,
        target_path: str,
        reference_path: str,
        expected_text: Optional[str] = None,
        language: Optional[str] = None,
        verbose: bool = False,
    ) -> Scores:
        from jiwer import cer as jiwer_cer
        from jiwer import wer as jiwer_wer

        if not verbose:
            import logging

            logging.getLogger("faster_whisper").setLevel(logging.ERROR)

        target_audio, sr_t = load_audio_mono(target_path, self.sample_rate)
        reference_audio, sr_r = load_audio_mono(reference_path, self.sample_rate)
        if not (sr_t == sr_r == self.sample_rate):
            raise ValueError(
                f"Sample-rate mismatch after load: target={sr_t}, reference={sr_r}, requested={self.sample_rate}"
            )

        target_duration = duration_s(target_audio, self.sample_rate)
        reference_duration = duration_s(reference_audio, self.sample_rate)

        _log(verbose, "=== TTS Similarity Evaluation ===")
        _log(verbose, f"Target:    {target_path}")
        _log(verbose, f"Reference: {reference_path}")
        _log(verbose, f"Sample rate: {self.sample_rate}")
        _log(
            verbose,
            f"Durations: target={format_float(target_duration)}s, reference={format_float(reference_duration)}s",
        )
        _log(verbose)

        # Speaker
        speaker_raw, speaker_pred, speaker_error = self.compute_speaker_similarity(target_path, reference_path)
        speaker_score = linear_similarity_score(speaker_raw, self.cfg.speaker_bad, self.cfg.speaker_good)
        _log(verbose, "--- Speaker ---")
        _log(verbose, f"Verification score: {speaker_raw}")
        _log(verbose, f"Same-speaker prediction: {speaker_pred}")
        if speaker_error:
            _log(verbose, f"Note: {speaker_error}")
        _log(verbose)

        # Content
        target_tx = self.transcribe(target_path, language)
        reference_tx = self.transcribe(reference_path, language)
        norm_tgt = normalize_text(target_tx)
        norm_ref = normalize_text(reference_tx)
        norm_exp = normalize_text(expected_text) if expected_text is not None else None

        wer_ref_norm = safe_float(jiwer_wer(norm_ref, norm_tgt))
        cer_ref_norm = safe_float(jiwer_cer(norm_ref, norm_tgt))
        ref_content = weighted_mean(
            [
                (linear_distance_score(wer_ref_norm, 0.0, self.cfg.content_wer_bad), self.cfg.content_wer_weight),
                (linear_distance_score(cer_ref_norm, 0.0, self.cfg.content_cer_bad), self.cfg.content_cer_weight),
            ]
        )

        exp_content = None
        wer_exp_norm = cer_exp_norm = None
        if norm_exp is not None:
            wer_exp_norm = safe_float(jiwer_wer(norm_exp, norm_tgt))
            cer_exp_norm = safe_float(jiwer_cer(norm_exp, norm_tgt))
            exp_content = weighted_mean(
                [
                    (linear_distance_score(wer_exp_norm, 0.0, self.cfg.content_wer_bad), self.cfg.content_wer_weight),
                    (linear_distance_score(cer_exp_norm, 0.0, self.cfg.content_cer_bad), self.cfg.content_cer_weight),
                ]
            )

        content_score = (
            weighted_mean(
                [
                    (exp_content, self.cfg.content_expected_weight),
                    (ref_content, self.cfg.content_reference_weight),
                ]
            )
            if exp_content is not None
            else ref_content
        )

        _log(verbose, "--- Content ---")
        _log(verbose, f"Reference transcript: {reference_tx}")
        _log(verbose, f"Target transcript:    {target_tx}")
        if expected_text is not None:
            _log(verbose, f"Expected text:        {expected_text}")
        _log(verbose, f"Normalized WER/CER (target vs reference): {wer_ref_norm}, {cer_ref_norm}")
        if exp_content is not None:
            _log(verbose, f"Normalized WER/CER (target vs expected):  {wer_exp_norm}, {cer_exp_norm}")
        _log(verbose, f"Normalized transcripts match: {norm_tgt == norm_ref}")
        _log(verbose)

        # Duration
        duration_diff = abs(target_duration - reference_duration) / max(reference_duration, 1e-8)
        duration_score = linear_distance_score(duration_diff, self.cfg.duration_diff_good, self.cfg.duration_diff_bad)

        _log(verbose, "--- Duration ---")
        _log(verbose, f"Target duration (s):    {target_duration}")
        _log(verbose, f"Reference duration (s): {reference_duration}")
        _log(verbose, f"Relative duration diff: {duration_diff}")
        _log(verbose)

        # Acoustic-lite
        centroid_tgt = safe_float(np.mean(librosa.feature.spectral_centroid(y=target_audio, sr=self.sample_rate)))
        centroid_ref = safe_float(np.mean(librosa.feature.spectral_centroid(y=reference_audio, sr=self.sample_rate)))
        rolloff_tgt = safe_float(np.mean(librosa.feature.spectral_rolloff(y=target_audio, sr=self.sample_rate)))
        rolloff_ref = safe_float(np.mean(librosa.feature.spectral_rolloff(y=reference_audio, sr=self.sample_rate)))

        centroid_diff = relative_diff(centroid_tgt, centroid_ref)
        rolloff_diff = relative_diff(rolloff_tgt, rolloff_ref)
        acoustic_score = weighted_mean(
            [
                (
                    linear_distance_score(
                        centroid_diff,
                        self.cfg.acoustic_centroid_diff_good,
                        self.cfg.acoustic_centroid_diff_bad,
                    ),
                    self.cfg.acoustic_centroid_weight,
                ),
                (
                    linear_distance_score(
                        rolloff_diff,
                        self.cfg.acoustic_rolloff_diff_good,
                        self.cfg.acoustic_rolloff_diff_bad,
                    ),
                    self.cfg.acoustic_rolloff_weight,
                ),
            ]
        )

        _log(verbose, "--- Acoustic ---")
        _log(verbose, f"Mean spectral centroid (tgt/ref): {centroid_tgt}, {centroid_ref}")
        _log(verbose, f"Mean spectral rolloff  (tgt/ref): {rolloff_tgt}, {rolloff_ref}")
        _log(verbose, f"Relative centroid diff:           {centroid_diff}")
        _log(verbose, f"Relative rolloff diff:            {rolloff_diff}")
        _log(verbose)

        valid = [content_score, speaker_score, duration_score, acoustic_score]
        if any(v is None for v in valid):
            overall = None
        else:
            overall = (
                content_score * self.cfg.overall_content_weight
                + speaker_score * self.cfg.overall_speaker_weight
                + acoustic_score * self.cfg.overall_acoustic_weight
                + duration_score * self.cfg.overall_duration_weight
            )

        scores = Scores(
            speaker=speaker_score,
            content=content_score,
            duration=duration_score,
            acoustic=acoustic_score,
            overall=overall,
        )

        _log(verbose, "--- Scores ---")
        _log(verbose, f"Speaker:  {format_float(scores.speaker)}")
        _log(verbose, f"Content:  {format_float(scores.content)}")
        _log(verbose, f"Duration: {format_float(scores.duration)}")
        _log(verbose, f"Acoustic: {format_float(scores.acoustic)}")
        _log(verbose, f"Overall:  {format_float(scores.overall)}")

        return scores


def evaluate_tts_similarity(
    target_path: str,
    reference_path: str,
    expected_text: Optional[str] = None,
    language: Optional[str] = None,
    sample_rate: int = DEFAULT_SR,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    whisper_device: str = "auto",
    whisper_compute_type: str = "default",
    scoring_config: Optional[ScoringConfig] = None,
    verbose: bool = False,
) -> Scores:
    evaluator = TTSSimilarityEvaluator(
        sample_rate=sample_rate,
        whisper_model=whisper_model,
        whisper_device=whisper_device,
        whisper_compute_type=whisper_compute_type,
        scoring_config=scoring_config,
    )
    return evaluator.evaluate(target_path, reference_path, expected_text, language, verbose=verbose)


def evaluate_tts_similarity_dict(**kwargs) -> dict[str, Any]:
    return asdict(evaluate_tts_similarity(**kwargs))


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate similarity between a TTS target wav and a reference wav.")
    p.add_argument("--target", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--expected-text", default=None)
    p.add_argument("--language", default=None)
    p.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    p.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    p.add_argument("--whisper-device", default="auto")
    p.add_argument("--whisper-compute-type", default="default")
    p.add_argument("--json-out", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    scores = evaluate_tts_similarity(
        target_path=args.target,
        reference_path=args.reference,
        expected_text=args.expected_text,
        language=args.language,
        sample_rate=args.sample_rate,
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
        whisper_compute_type=args.whisper_compute_type,
        verbose=args.verbose,
    )

    if not args.verbose:
        print(json.dumps(asdict(scores), indent=2))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(asdict(scores), f, indent=2, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
