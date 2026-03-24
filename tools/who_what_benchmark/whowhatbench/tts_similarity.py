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
from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer

DEFAULT_SR = 22050
DEFAULT_WHISPER_MODEL = "small"


# ---- Public result / config -------------------------------------------------

@dataclass
class Scores:
    # Speaker = same voice?
    speaker: Optional[float]
    # Content = same words?
    content: Optional[float]
    # Prosody = same delivery?
    prosody: Optional[float]
    # Acoustic = same overall sound?
    acoustic: Optional[float]
    overall: Optional[float]


@dataclass
class ScoringConfig:
    # SpeechBrain verification score -> normalized speaker score.
    # These are heuristic anchors, not universal constants.
    speaker_bad: float = 0.10
    speaker_good: float = 0.75

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

    # Prosody = pitch contour + energy contour + speaking-rate difference.
    prosody_f0_good: float = 0.05
    prosody_f0_bad: float = 0.30
    prosody_rms_good: float = 0.03
    prosody_rms_bad: float = 0.20
    prosody_rate_good: float = 0.05
    prosody_rate_bad: float = 0.25
    prosody_f0_weight: float = 0.50
    prosody_rms_weight: float = 0.30
    prosody_rate_weight: float = 0.20

    # Acoustic = coarse spectral similarity proxies.
    acoustic_mfcc_good: float = 20.0
    acoustic_mfcc_bad: float = 100.0
    acoustic_logmel_good: float = 20.0
    acoustic_logmel_bad: float = 100.0
    acoustic_mcd_good: float = 120.0
    acoustic_mcd_bad: float = 600.0
    acoustic_mfcc_weight: float = 0.45
    acoustic_logmel_weight: float = 0.45
    acoustic_mcd_weight: float = 0.10

    # Overall weighting: content matters most, then speaker, then prosody, then acoustic.
    overall_content_weight: float = 0.35
    overall_speaker_weight: float = 0.30
    overall_prosody_weight: float = 0.20
    overall_acoustic_weight: float = 0.15


# ---- Small helpers ----------------------------------------------------------

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


# ---- Audio / model utilities ------------------------------------------------

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


def dtw_mean_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> Optional[float]:
    if seq_a.size == 0 or seq_b.size == 0:
        return None
    D, _ = librosa.sequence.dtw(X=seq_a, Y=seq_b, metric="euclidean")
    if D.size == 0:
        return None
    path_len = max(1, D.shape[0] + D.shape[1])
    return float(D[-1, -1] / path_len)


def voiced_f0_stats(audio: np.ndarray, sr: int) -> tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
    """
    Extract pitch-related summaries:
      - f0: frame-by-frame pitch contour (Hz)
      - voiced_ratio: fraction of frames classified as voiced
      - median_hz: typical voiced pitch (robust summary)
    """
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            sr=sr,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            frame_length=2048,
            hop_length=256,
        )
        if f0 is None or voiced_flag is None:
            return None, None, None
        voiced_ratio = float(np.mean(voiced_flag.astype(np.float32)))
        voiced_vals = f0[np.isfinite(f0)]
        median_hz = float(np.median(voiced_vals)) if voiced_vals.size else None
        return f0, voiced_ratio, median_hz
    except Exception:
        return None, None, None


# ---- Core evaluator ---------------------------------------------------------

class TTSSimilarityEvaluator:
    """
    Lean evaluator that returns just the final 4 category scores (+ overall).

    Verbose mode prints intermediate metrics as they are computed instead of
    storing a large result structure.
    """

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
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
            )
        return self._speaker_model

    def transcribe(self, path: str, language: Optional[str]) -> str:
        segments, _info = self.whisper.transcribe(path, language=language, vad_filter=False)
        return " ".join(seg.text.strip() for seg in segments).strip()

    def compute_speaker_similarity(self, target_path: str, reference_path: str) -> tuple[Optional[float], Optional[int], Optional[str]]:
        try:
            import torch

            verification = self.speaker_model
            def _load_wav_tensor(path: str, target_sr: int = 16000):
                wav, sr = sf.read(path, always_2d=False)
                if getattr(wav, "ndim", 1) == 2:
                    wav = np.mean(wav, axis=1)
                wav = np.asarray(wav, dtype=np.float32)
                if sr != target_sr:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
                peak = np.max(np.abs(wav)) if wav.size else 0.0
                if peak > 1.0:
                    wav = wav / peak
                return torch.tensor(wav, dtype=torch.float32).unsqueeze(0)

            ref_tensor = _load_wav_tensor(reference_path)
            tgt_tensor = _load_wav_tensor(target_path)
            emb_ref = verification.encode_batch(ref_tensor).reshape(1, -1)
            emb_tgt = verification.encode_batch(tgt_tensor).reshape(1, -1)
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
        target_audio, sr_t = load_audio_mono(target_path, self.sample_rate)
        reference_audio, sr_r = load_audio_mono(reference_path, self.sample_rate)
        if not (sr_t == sr_r == self.sample_rate):
            raise ValueError(
                f"Sample-rate mismatch after load: target={sr_t}, reference={sr_r}, requested={self.sample_rate}"
            )

        _log(verbose, "=== TTS Similarity Evaluation ===")
        _log(verbose, f"Target:    {target_path}")
        _log(verbose, f"Reference: {reference_path}")
        _log(verbose, f"Sample rate: {self.sample_rate}")
        _log(verbose, f"Durations: target={format_float(duration_s(target_audio, self.sample_rate))}s, "
                      f"reference={format_float(duration_s(reference_audio, self.sample_rate))}s")
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
        ref_content = weighted_mean([
            (linear_distance_score(wer_ref_norm, 0.0, self.cfg.content_wer_bad), self.cfg.content_wer_weight),
            (linear_distance_score(cer_ref_norm, 0.0, self.cfg.content_cer_bad), self.cfg.content_cer_weight),
        ])

        exp_content = None
        wer_exp_norm = cer_exp_norm = None
        if norm_exp is not None:
            wer_exp_norm = safe_float(jiwer_wer(norm_exp, norm_tgt))
            cer_exp_norm = safe_float(jiwer_cer(norm_exp, norm_tgt))
            exp_content = weighted_mean([
                (linear_distance_score(wer_exp_norm, 0.0, self.cfg.content_wer_bad), self.cfg.content_wer_weight),
                (linear_distance_score(cer_exp_norm, 0.0, self.cfg.content_cer_bad), self.cfg.content_cer_weight),
            ])

        content_score = weighted_mean([
            (exp_content, self.cfg.content_expected_weight),
            (ref_content, self.cfg.content_reference_weight),
        ]) if exp_content is not None else ref_content

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

        # Prosody
        tgt_f0, tgt_voiced, tgt_med = voiced_f0_stats(target_audio, self.sample_rate)
        ref_f0, ref_voiced, ref_med = voiced_f0_stats(reference_audio, self.sample_rate)

        f0_dtw = None
        if tgt_f0 is not None and ref_f0 is not None:
            tgt_log = np.nan_to_num(np.log(np.where(np.isfinite(tgt_f0), tgt_f0, np.nan)), nan=0.0)[None, :]
            ref_log = np.nan_to_num(np.log(np.where(np.isfinite(ref_f0), ref_f0, np.nan)), nan=0.0)[None, :]
            f0_dtw = dtw_mean_distance(tgt_log, ref_log)

        rms_tgt = librosa.feature.rms(y=target_audio, frame_length=2048, hop_length=256)
        rms_ref = librosa.feature.rms(y=reference_audio, frame_length=2048, hop_length=256)
        rms_dtw = dtw_mean_distance(rms_tgt, rms_ref)

        tgt_rate = len(target_tx.strip()) / max(duration_s(target_audio, self.sample_rate), 1e-8) if target_tx.strip() else None
        ref_rate = len(reference_tx.strip()) / max(duration_s(reference_audio, self.sample_rate), 1e-8) if reference_tx.strip() else None
        rate_diff = None if (tgt_rate is None or ref_rate is None) else abs(tgt_rate - ref_rate) / max(ref_rate, 1e-8)

        prosody_score = weighted_mean([
            (linear_distance_score(f0_dtw, self.cfg.prosody_f0_good, self.cfg.prosody_f0_bad), self.cfg.prosody_f0_weight),
            (linear_distance_score(rms_dtw, self.cfg.prosody_rms_good, self.cfg.prosody_rms_bad), self.cfg.prosody_rms_weight),
            (linear_distance_score(rate_diff, self.cfg.prosody_rate_good, self.cfg.prosody_rate_bad), self.cfg.prosody_rate_weight),
        ])

        _log(verbose, "--- Prosody ---")
        _log(verbose, f"F0 DTW distance:         {f0_dtw}")
        _log(verbose, f"Voiced ratio (tgt/ref):  {tgt_voiced}, {ref_voiced}")
        _log(verbose, f"Median F0 Hz (tgt/ref):  {tgt_med}, {ref_med}")
        _log(verbose, f"RMS DTW distance:        {rms_dtw}")
        _log(verbose, f"Speaking rate diff:      {rate_diff}")
        _log(verbose)

        # Acoustic
        n_fft, hop, n_mels, n_mfcc = 1024, 256, 80, 13
        tgt_mfcc = librosa.feature.mfcc(y=target_audio, sr=self.sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop)
        ref_mfcc = librosa.feature.mfcc(y=reference_audio, sr=self.sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop)
        tgt_mel = librosa.power_to_db(
            librosa.feature.melspectrogram(y=target_audio, sr=self.sample_rate, n_fft=n_fft, hop_length=hop, n_mels=n_mels),
            ref=np.max,
        )
        ref_mel = librosa.power_to_db(
            librosa.feature.melspectrogram(y=reference_audio, sr=self.sample_rate, n_fft=n_fft, hop_length=hop, n_mels=n_mels),
            ref=np.max,
        )

        mfcc_dtw = dtw_mean_distance(tgt_mfcc, ref_mfcc)
        logmel_dtw = dtw_mean_distance(tgt_mel, ref_mel)

        mcd_like = None
        try:
            _, wp = librosa.sequence.dtw(X=tgt_mfcc, Y=ref_mfcc, metric="euclidean")
            if wp is not None and len(wp) > 0:
                dists = [float(np.sqrt(np.sum((tgt_mfcc[:, i] - ref_mfcc[:, j]) ** 2))) for i, j in wp]
                if dists:
                    mcd_like = float((10.0 / np.log(10.0)) * np.sqrt(2.0) * np.mean(dists))
        except Exception:
            pass

        acoustic_score = weighted_mean([
            (linear_distance_score(mfcc_dtw, self.cfg.acoustic_mfcc_good, self.cfg.acoustic_mfcc_bad), self.cfg.acoustic_mfcc_weight),
            (linear_distance_score(logmel_dtw, self.cfg.acoustic_logmel_good, self.cfg.acoustic_logmel_bad), self.cfg.acoustic_logmel_weight),
            (linear_distance_score(mcd_like, self.cfg.acoustic_mcd_good, self.cfg.acoustic_mcd_bad), self.cfg.acoustic_mcd_weight),
        ])

        _log(verbose, "--- Acoustic ---")
        _log(verbose, f"MFCC DTW mean distance:  {mfcc_dtw}")
        _log(verbose, f"Log-mel DTW distance:    {logmel_dtw}")
        _log(verbose, f"MCD-like distance:       {mcd_like}")
        _log(verbose)

        overall = weighted_mean([
            (content_score, self.cfg.overall_content_weight),
            (speaker_score, self.cfg.overall_speaker_weight),
            (prosody_score, self.cfg.overall_prosody_weight),
            (acoustic_score, self.cfg.overall_acoustic_weight),
        ])

        scores = Scores(
            speaker=speaker_score,
            content=content_score,
            prosody=prosody_score,
            acoustic=acoustic_score,
            overall=overall,
        )

        return scores


# ---- Convenience wrappers / CLI --------------------------------------------

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
