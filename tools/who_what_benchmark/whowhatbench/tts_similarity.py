
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import string
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import soundfile as sf
from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer


DEFAULT_SR = 22050
DEFAULT_WHISPER_MODEL = "small"


@dataclass
class TranscriptResult:
    text: str
    language: Optional[str]


@dataclass
class ScoreBreakdown:
    speaker_score: Optional[float]
    content_score: Optional[float]
    prosody_score: Optional[float]
    acoustic_score: Optional[float]
    overall_score: Optional[float]


@dataclass
class SpeakerScoreConfig:
    # SpeechBrain verification score: higher means more likely same speaker.
    # 0.10 is a deliberately conservative "poor match" anchor, while 0.75
    # is used as a strong same-speaker anchor for this heuristic mapping.
    # These are not universal constants; they should be calibrated on your data.
    verification_bad: float = 0.10
    verification_good: float = 0.75


@dataclass
class ContentScoreConfig:
    # WER/CER are lower-is-better distances. We map 0.0 -> perfect, then
    # degrade toward 0 as errors grow.
    # WER bad=0.35: roughly 35% word error is a meaningful content miss.
    # CER bad=0.15: character drift above ~15% is also a strong mismatch.
    # WER gets most of the weight because word correctness usually matters more
    # than minor character/punctuation differences.
    wer_good: float = 0.0
    wer_bad: float = 0.35
    cer_good: float = 0.0
    cer_bad: float = 0.15
    wer_weight: float = 0.60
    cer_weight: float = 0.40

    # By default, if expected text is available, content scoring now combines
    # both target-vs-expected correctness and target-vs-reference parity.
    #
    # This is a better default for TTS parity testing because:
    #   - target-vs-expected catches cases where the target says the wrong text
    #   - target-vs-reference catches cases where target and reference drift apart
    #
    # Reference parity gets more weight by default because if target and
    # reference ASR transcripts match, small expected-text formatting differences
    # (such as hyphens, punctuation, etc.) should not dominate the score.
    prefer_expected_text: bool = True
    combine_expected_and_reference: bool = True
    expected_weight: float = 0.25
    reference_weight: float = 0.75


@dataclass
class ProsodyScoreConfig:
    # F0 DTW distance compares pitch contour shape.
    # good=0.05: very similar melody/intonation contour.
    # bad=0.30: noticeably different pitch contour / delivery.
    f0_dtw_good: float = 0.05
    f0_dtw_bad: float = 0.30

    # RMS DTW distance compares loudness/energy envelope.
    # good=0.03: close emphasis contour.
    # bad=0.20: clearly different stress / energy pattern.
    rms_dtw_good: float = 0.03
    rms_dtw_bad: float = 0.20

    # Relative speaking-rate difference.
    # good=0.05: within ~5% pace difference is effectively the same tempo.
    # bad=0.25: ~25% difference is an obvious faster/slower delivery.
    speaking_rate_diff_good: float = 0.05
    speaking_rate_diff_bad: float = 0.25

    # F0 carries the most prosody weight, then energy envelope, then pace.
    f0_weight: float = 0.50
    rms_weight: float = 0.30
    rate_weight: float = 0.20


@dataclass
class AcousticScoreConfig:
    # MFCC / log-mel DTW are lower-is-better spectral similarity proxies.
    # good=20 and bad=100 are heuristic anchors chosen so clearly close pairs
    # score high while obviously different pairs score low. They are intended
    # as starting points, not dataset-independent truths.
    mfcc_good: float = 20.0
    mfcc_bad: float = 100.0
    logmel_good: float = 20.0
    logmel_bad: float = 100.0

    # This MCD-like value is a rough proxy rather than a canonical MCD metric,
    # so it gets a lighter weight. 120 ~= fairly close, 600 ~= very different.
    mcd_good: float = 120.0
    mcd_bad: float = 600.0

    mfcc_weight: float = 0.45
    logmel_weight: float = 0.45
    mcd_weight: float = 0.10


@dataclass
class OverallScoreConfig:
    # Overall weights reflect a "behavioral parity" view:
    #  - content matters most (wrong words are usually most severe)
    #  - speaker is next (same text but wrong voice is also severe)
    #  - prosody matters, but mild drift can be acceptable
    #  - acoustic matters, but these metrics are the least semantically direct
    content_weight: float = 0.35
    speaker_weight: float = 0.30
    prosody_weight: float = 0.20
    acoustic_weight: float = 0.15


@dataclass
class ScoringConfig:
    speaker: SpeakerScoreConfig = field(default_factory=SpeakerScoreConfig)
    content: ContentScoreConfig = field(default_factory=ContentScoreConfig)
    prosody: ProsodyScoreConfig = field(default_factory=ProsodyScoreConfig)
    acoustic: AcousticScoreConfig = field(default_factory=AcousticScoreConfig)
    overall: OverallScoreConfig = field(default_factory=OverallScoreConfig)


@dataclass
class EvalResult:
    target_path: str
    reference_path: str
    analysis_sample_rate: int
    target_duration_s: float
    reference_duration_s: float

    speaker_verification_score: Optional[float]
    speaker_same_prediction: Optional[int]

    reference_transcript: str
    target_transcript: str

    normalized_reference_transcript: str
    normalized_target_transcript: str
    normalized_transcripts_match: bool

    wer_target_vs_reference: Optional[float]
    cer_target_vs_reference: Optional[float]
    wer_target_vs_reference_normalized: Optional[float]
    cer_target_vs_reference_normalized: Optional[float]

    expected_text: Optional[str]
    normalized_expected_text: Optional[str]
    wer_target_vs_expected: Optional[float]
    cer_target_vs_expected: Optional[float]
    wer_target_vs_expected_normalized: Optional[float]
    cer_target_vs_expected_normalized: Optional[float]
    wer_reference_vs_expected: Optional[float]
    cer_reference_vs_expected: Optional[float]
    wer_reference_vs_expected_normalized: Optional[float]
    cer_reference_vs_expected_normalized: Optional[float]

    f0_dtw_distance: Optional[float]
    f0_voiced_ratio_target: Optional[float]
    f0_voiced_ratio_reference: Optional[float]
    f0_median_hz_target: Optional[float]
    f0_median_hz_reference: Optional[float]
    rms_dtw_distance: Optional[float]
    speaking_rate_chars_per_s_target: Optional[float]
    speaking_rate_chars_per_s_reference: Optional[float]
    speaking_rate_relative_diff: Optional[float]

    mfcc_dtw_mean_distance: Optional[float]
    logmel_dtw_mean_distance: Optional[float]
    mcd_like_distance: Optional[float]

    scores: ScoreBreakdown
    notes: List[str]


def normalize_text(text: str) -> str:
    # Treat hyphens/dashes as separators before punctuation stripping so
    # "well-designed" and "well designed" normalize to the same text.
    text = text.lower().strip()
    text = re.sub(r"[-‐‑‒–—]+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        xf = float(x)
    except Exception:
        return None
    if math.isnan(xf) or math.isinf(xf):
        return None
    return xf


def linear_distance_score(value: Optional[float], good: float, bad: float) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if value <= good:
        return 1.0
    if value >= bad:
        return 0.0
    return 1.0 - (value - good) / (bad - good)


def linear_similarity_score(value: Optional[float], bad: float, good: float) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if value <= bad:
        return 0.0
    if value >= good:
        return 1.0
    return (value - bad) / (good - bad)


def weighted_mean(items: Sequence[Tuple[Optional[float], float]]) -> Optional[float]:
    valid = [(v, w) for v, w in items if v is not None and w > 0]
    if not valid:
        return None
    total_w = sum(w for _, w in valid)
    return sum(v * w for v, w in valid) / total_w


def load_audio_mono(path: str, sr: int) -> Tuple[np.ndarray, int]:
    audio, file_sr = sf.read(path, always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    if file_sr != sr:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        file_sr = sr
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio))
    return audio, file_sr


def compute_duration_s(audio: np.ndarray, sr: int) -> float:
    return float(len(audio) / sr)


def transcribe_audio(path: str, model_name: str, language: Optional[str], device: str, compute_type: str) -> TranscriptResult:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = model.transcribe(path, language=language, vad_filter=False)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    detected_lang = getattr(info, "language", None)
    return TranscriptResult(text=text, language=detected_lang)


def compute_speaker_similarity(target_path: str, reference_path: str) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    try:
        import torchaudio
        import torch

        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: []
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda _backend: None

        from speechbrain.inference.speaker import SpeakerRecognition

        verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
        )
        try:
            score, prediction = verification.verify_files(reference_path, target_path)

            if hasattr(score, "item"):
                score = float(score.item())
            else:
                score = float(score)

            if hasattr(prediction, "item"):
                prediction = int(prediction.item())
            else:
                prediction = int(prediction)

            return score, prediction, None
        except Exception as verify_exc:
            def _load_wav_tensor(path: str, target_sr: int = 16000):
                wav, sr = sf.read(path, always_2d=False)
                if isinstance(wav, np.ndarray) and wav.ndim == 2:
                    wav = np.mean(wav, axis=1)
                wav = np.asarray(wav, dtype=np.float32)
                if sr != target_sr:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
                if np.max(np.abs(wav)) > 1.0:
                    wav = wav / np.max(np.abs(wav))
                return torch.tensor(wav, dtype=torch.float32).unsqueeze(0)

            ref_tensor = _load_wav_tensor(reference_path)
            tgt_tensor = _load_wav_tensor(target_path)

            emb_ref = verification.encode_batch(ref_tensor)
            emb_tgt = verification.encode_batch(tgt_tensor)

            emb_ref_flat = emb_ref.reshape(1, -1)
            emb_tgt_flat = emb_tgt.reshape(1, -1)
            score = float(torch.nn.functional.cosine_similarity(emb_ref_flat, emb_tgt_flat, dim=1).item())
            prediction = int(score >= 0.25)

            return score, prediction, (
                f"compute_speaker_similarity: verify_files failed ({verify_exc}); "
                "used manual waveform embedding fallback"
            )
    except Exception as e:
        return None, None, f"compute_speaker_similarity: Exception: {e}"


def dtw_mean_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> Optional[float]:
    if seq_a.size == 0 or seq_b.size == 0:
        return None
    D, _ = librosa.sequence.dtw(X=seq_a, Y=seq_b, metric="euclidean")
    if D.size == 0:
        return None
    path_len = max(1, D.shape[0] + D.shape[1])
    return float(D[-1, -1] / path_len)


def voiced_f0_stats(audio: np.ndarray, sr: int) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
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


def compute_prosody_metrics(
    target_audio: np.ndarray,
    reference_audio: np.ndarray,
    sr: int,
    target_text: str,
    reference_text: str,
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "f0_dtw_distance": None,
        "f0_voiced_ratio_target": None,
        "f0_voiced_ratio_reference": None,
        "f0_median_hz_target": None,
        "f0_median_hz_reference": None,
        "rms_dtw_distance": None,
        "speaking_rate_chars_per_s_target": None,
        "speaking_rate_chars_per_s_reference": None,
        "speaking_rate_relative_diff": None,
    }

    tgt_f0, tgt_voiced_ratio, tgt_med = voiced_f0_stats(target_audio, sr)
    ref_f0, ref_voiced_ratio, ref_med = voiced_f0_stats(reference_audio, sr)
    out["f0_voiced_ratio_target"] = tgt_voiced_ratio
    out["f0_voiced_ratio_reference"] = ref_voiced_ratio
    out["f0_median_hz_target"] = tgt_med
    out["f0_median_hz_reference"] = ref_med

    if tgt_f0 is not None and ref_f0 is not None:
        tgt_logf0 = np.log(np.where(np.isfinite(tgt_f0), tgt_f0, np.nan))
        ref_logf0 = np.log(np.where(np.isfinite(ref_f0), ref_f0, np.nan))
        tgt_logf0 = np.nan_to_num(tgt_logf0, nan=0.0, posinf=0.0, neginf=0.0)[None, :]
        ref_logf0 = np.nan_to_num(ref_logf0, nan=0.0, posinf=0.0, neginf=0.0)[None, :]
        out["f0_dtw_distance"] = dtw_mean_distance(tgt_logf0, ref_logf0)

    rms_tgt = librosa.feature.rms(y=target_audio, frame_length=2048, hop_length=256)
    rms_ref = librosa.feature.rms(y=reference_audio, frame_length=2048, hop_length=256)
    out["rms_dtw_distance"] = dtw_mean_distance(rms_tgt, rms_ref)

    tgt_dur = max(compute_duration_s(target_audio, sr), 1e-8)
    ref_dur = max(compute_duration_s(reference_audio, sr), 1e-8)
    out["speaking_rate_chars_per_s_target"] = len(target_text.strip()) / tgt_dur if target_text.strip() else None
    out["speaking_rate_chars_per_s_reference"] = len(reference_text.strip()) / ref_dur if reference_text.strip() else None
    if out["speaking_rate_chars_per_s_target"] is not None and out["speaking_rate_chars_per_s_reference"] is not None:
        ref_rate = max(out["speaking_rate_chars_per_s_reference"], 1e-8)
        out["speaking_rate_relative_diff"] = abs(out["speaking_rate_chars_per_s_target"] - out["speaking_rate_chars_per_s_reference"]) / ref_rate

    return out


def compute_acoustic_metrics(target_audio: np.ndarray, reference_audio: np.ndarray, sr: int) -> Dict[str, Optional[float]]:
    n_fft = 1024
    hop = 256
    n_mels = 80
    n_mfcc = 13

    tgt_mfcc = librosa.feature.mfcc(y=target_audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop)
    ref_mfcc = librosa.feature.mfcc(y=reference_audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop)

    tgt_mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=target_audio, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels),
        ref=np.max,
    )
    ref_mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=reference_audio, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels),
        ref=np.max,
    )

    mfcc_dtw = dtw_mean_distance(tgt_mfcc, ref_mfcc)
    logmel_dtw = dtw_mean_distance(tgt_mel, ref_mel)

    mcd_like = None
    try:
        _, wp = librosa.sequence.dtw(X=tgt_mfcc, Y=ref_mfcc, metric="euclidean")
        if wp is not None and len(wp) > 0:
            distances = []
            for i, j in wp:
                diff = tgt_mfcc[:, i] - ref_mfcc[:, j]
                distances.append(float(np.sqrt(np.sum(diff * diff))))
            if distances:
                mcd_like = float((10.0 / np.log(10.0)) * np.sqrt(2.0) * np.mean(distances))
    except Exception:
        mcd_like = None

    return {
        "mfcc_dtw_mean_distance": mfcc_dtw,
        "logmel_dtw_mean_distance": logmel_dtw,
        "mcd_like_distance": mcd_like,
    }


def compute_text_metrics(target_text: str, reference_text: str, expected_text: Optional[str]) -> Dict[str, Optional[float]]:
    norm_target = normalize_text(target_text)
    norm_ref = normalize_text(reference_text)
    norm_expected = normalize_text(expected_text) if expected_text is not None else None

    out: Dict[str, Optional[float]] = {
        "normalized_reference_transcript": norm_ref,
        "normalized_target_transcript": norm_target,
        "normalized_transcripts_match": norm_target == norm_ref,
        "wer_target_vs_reference": safe_float(jiwer_wer(reference_text, target_text)),
        "cer_target_vs_reference": safe_float(jiwer_cer(reference_text, target_text)),
        "wer_target_vs_reference_normalized": safe_float(jiwer_wer(norm_ref, norm_target)),
        "cer_target_vs_reference_normalized": safe_float(jiwer_cer(norm_ref, norm_target)),
        "normalized_expected_text": norm_expected,
        "wer_target_vs_expected": None,
        "cer_target_vs_expected": None,
        "wer_target_vs_expected_normalized": None,
        "cer_target_vs_expected_normalized": None,
        "wer_reference_vs_expected": None,
        "cer_reference_vs_expected": None,
        "wer_reference_vs_expected_normalized": None,
        "cer_reference_vs_expected_normalized": None,
    }

    if expected_text is not None:
        out["wer_target_vs_expected"] = safe_float(jiwer_wer(expected_text, target_text))
        out["cer_target_vs_expected"] = safe_float(jiwer_cer(expected_text, target_text))
        out["wer_target_vs_expected_normalized"] = safe_float(jiwer_wer(norm_expected, norm_target))
        out["cer_target_vs_expected_normalized"] = safe_float(jiwer_cer(norm_expected, norm_target))
        out["wer_reference_vs_expected"] = safe_float(jiwer_wer(expected_text, reference_text))
        out["cer_reference_vs_expected"] = safe_float(jiwer_cer(expected_text, reference_text))
        out["wer_reference_vs_expected_normalized"] = safe_float(jiwer_wer(norm_expected, norm_ref))
        out["cer_reference_vs_expected_normalized"] = safe_float(jiwer_cer(norm_expected, norm_ref))

    return out


def compute_category_scores(metrics: Dict[str, Any], config: Optional[ScoringConfig] = None) -> ScoreBreakdown:
    if config is None:
        config = ScoringConfig()

    speaker_score = linear_similarity_score(
        metrics.get("speaker_verification_score"),
        bad=config.speaker.verification_bad,
        good=config.speaker.verification_good,
    )

    wer_expected_norm = metrics.get("wer_target_vs_expected_normalized")
    cer_expected_norm = metrics.get("cer_target_vs_expected_normalized")
    wer_ref_norm = metrics.get("wer_target_vs_reference_normalized")
    cer_ref_norm = metrics.get("cer_target_vs_reference_normalized")

    expected_content_score = weighted_mean([
        (
            linear_distance_score(
                wer_expected_norm,
                good=config.content.wer_good,
                bad=config.content.wer_bad,
            ),
            config.content.wer_weight,
        ),
        (
            linear_distance_score(
                cer_expected_norm,
                good=config.content.cer_good,
                bad=config.content.cer_bad,
            ),
            config.content.cer_weight,
        ),
    ])

    reference_content_score = weighted_mean([
        (
            linear_distance_score(
                wer_ref_norm,
                good=config.content.wer_good,
                bad=config.content.wer_bad,
            ),
            config.content.wer_weight,
        ),
        (
            linear_distance_score(
                cer_ref_norm,
                good=config.content.cer_good,
                bad=config.content.cer_bad,
            ),
            config.content.cer_weight,
        ),
    ])

    if config.content.combine_expected_and_reference:
        content_score = weighted_mean([
            (expected_content_score, config.content.expected_weight),
            (reference_content_score, config.content.reference_weight),
        ])
    elif config.content.prefer_expected_text and (wer_expected_norm is not None or cer_expected_norm is not None):
        content_score = expected_content_score
    else:
        content_score = reference_content_score

    prosody_score = weighted_mean([
        (
            linear_distance_score(
                metrics.get("f0_dtw_distance"),
                good=config.prosody.f0_dtw_good,
                bad=config.prosody.f0_dtw_bad,
            ),
            config.prosody.f0_weight,
        ),
        (
            linear_distance_score(
                metrics.get("rms_dtw_distance"),
                good=config.prosody.rms_dtw_good,
                bad=config.prosody.rms_dtw_bad,
            ),
            config.prosody.rms_weight,
        ),
        (
            linear_distance_score(
                metrics.get("speaking_rate_relative_diff"),
                good=config.prosody.speaking_rate_diff_good,
                bad=config.prosody.speaking_rate_diff_bad,
            ),
            config.prosody.rate_weight,
        ),
    ])

    acoustic_score = weighted_mean([
        (
            linear_distance_score(
                metrics.get("mfcc_dtw_mean_distance"),
                good=config.acoustic.mfcc_good,
                bad=config.acoustic.mfcc_bad,
            ),
            config.acoustic.mfcc_weight,
        ),
        (
            linear_distance_score(
                metrics.get("logmel_dtw_mean_distance"),
                good=config.acoustic.logmel_good,
                bad=config.acoustic.logmel_bad,
            ),
            config.acoustic.logmel_weight,
        ),
        (
            linear_distance_score(
                metrics.get("mcd_like_distance"),
                good=config.acoustic.mcd_good,
                bad=config.acoustic.mcd_bad,
            ),
            config.acoustic.mcd_weight,
        ),
    ])

    overall_score = weighted_mean([
        (content_score, config.overall.content_weight),
        (speaker_score, config.overall.speaker_weight),
        (prosody_score, config.overall.prosody_weight),
        (acoustic_score, config.overall.acoustic_weight),
    ])

    return ScoreBreakdown(
        speaker_score=speaker_score,
        content_score=content_score,
        prosody_score=prosody_score,
        acoustic_score=acoustic_score,
        overall_score=overall_score,
    )


def qualitative_label(score: Optional[float]) -> str:
    if score is None:
        return "N/A"
    if score >= 0.90:
        return "excellent"
    if score >= 0.80:
        return "good"
    if score >= 0.65:
        return "fair"
    return "poor"


def build_notes(metrics: Dict[str, Any], speaker_error: Optional[str], scoring_config: Optional[ScoringConfig] = None) -> List[str]:
    if scoring_config is None:
        scoring_config = ScoringConfig()

    notes: List[str] = []

    if metrics.get("normalized_transcripts_match"):
        notes.append("Target/reference transcripts match after normalization, so spectral distances are more interpretable.")
    else:
        notes.append("Target/reference transcripts differ after normalization, so MFCC/log-mel distances should be interpreted cautiously.")

    if metrics.get("expected_text") is not None:
        if scoring_config.content.combine_expected_and_reference:
            notes.append(
                "Content score combines normalized target-vs-expected correctness and "
                "target-vs-reference parity; reference parity is weighted more heavily by default."
            )
        else:
            notes.append("Content score is based primarily on normalized WER/CER against the expected text.")
    else:
        notes.append("No expected text was provided, so content score falls back to target-vs-reference transcript similarity.")

    notes.append("Text normalization treats hyphens/dashes as separators, so forms like 'well-designed' and 'well designed' compare more naturally.")
    notes.append("Speaker score is derived from SpeechBrain speaker-verification output, then normalized to 0..1 using heuristic thresholds.")
    notes.append("Same-speaker prediction is a binary diagnostic only; the continuous verification score is the main speaker metric.")
    notes.append("Prosody score combines pitch contour distance, RMS energy contour distance, and speaking-rate difference.")
    notes.append("Acoustic score combines MFCC DTW, log-mel DTW, and a simple MCD-like distance.")
    notes.append("The score thresholds in this script are intentionally heuristic starting points and should be calibrated on your own good/bad examples.")

    if speaker_error:
        notes.append(speaker_error)

    return notes


def format_float(x: Optional[float], digits: int = 6) -> str:
    if x is None:
        return "None"
    return f"{x:.{digits}f}"


class TTSSimilarityEvaluator:
    """
    Stateful TTS similarity evaluator that caches expensive model instances.

    Whisper and SpeakerRecognition models are instantiated once on first use
    and reused across all subsequent `evaluate()` calls, dramatically reducing
    per-sample evaluation time in loops.
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
        self.scoring_config = scoring_config or ScoringConfig()

        self._whisper_model_cache = None

    @property
    def _whisper_model(self):
        if self._whisper_model_cache is None:
            from faster_whisper import WhisperModel
            self._whisper_model_cache = WhisperModel(
                self.whisper_model_name,
                device=self.whisper_device,
                compute_type=self.whisper_compute_type,
            )
        return self._whisper_model_cache

    def _transcribe_audio(
        self, path: str, language: Optional[str]
    ) -> TranscriptResult:
        """Transcribe audio using cached Whisper model."""
        segments, info = self._whisper_model.transcribe(
            path, language=language, vad_filter=False
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        detected_lang = getattr(info, "language", None)
        return TranscriptResult(text=text, language=detected_lang)

    def evaluate(
        self,
        target_path: str,
        reference_path: str,
        expected_text: Optional[str] = None,
        language: Optional[str] = None,
    ) -> EvalResult:
        """
        Evaluate TTS similarity between target and reference audio.

        Expensive models (Whisper, SpeechBrain) are cached and reused.

        Args:
            target_path: Path to generated/target audio file.
            reference_path: Path to reference/golden audio file.
            expected_text: Optional ground-truth text for content scoring.
            language: Optional language hint for transcription.

        Returns:
            EvalResult with metrics, category scores, and diagnostic notes.
        """
        target_audio, sr_t = load_audio_mono(target_path, self.sample_rate)
        reference_audio, sr_r = load_audio_mono(
            reference_path, self.sample_rate
        )
        if not (sr_t == sr_r == self.sample_rate):
            raise ValueError(
                f"Unexpected sample-rate mismatch after loading/resampling: "
                f"target={sr_t}, reference={sr_r}, requested={self.sample_rate}"
            )

        target_duration = compute_duration_s(target_audio, self.sample_rate)
        reference_duration = compute_duration_s(
            reference_audio, self.sample_rate
        )

        ref_tx = self._transcribe_audio(reference_path, language)
        tgt_tx = self._transcribe_audio(target_path, language)

        speaker_score_raw, speaker_prediction, speaker_error = (
            compute_speaker_similarity(target_path, reference_path)
        )

        text_metrics = compute_text_metrics(tgt_tx.text, ref_tx.text, expected_text)
        prosody_metrics = compute_prosody_metrics(
            target_audio, reference_audio, self.sample_rate, tgt_tx.text, ref_tx.text
        )
        acoustic_metrics = compute_acoustic_metrics(
            target_audio, reference_audio, self.sample_rate
        )

        merged: Dict[str, Any] = {
            "speaker_verification_score": speaker_score_raw,
            "speaker_same_prediction": speaker_prediction,
            "expected_text": expected_text,
            **text_metrics,
            **prosody_metrics,
            **acoustic_metrics,
        }

        scores = compute_category_scores(merged, config=self.scoring_config)
        notes = build_notes(merged, speaker_error, scoring_config=self.scoring_config)

        return EvalResult(
            target_path=target_path,
            reference_path=reference_path,
            analysis_sample_rate=self.sample_rate,
            target_duration_s=target_duration,
            reference_duration_s=reference_duration,
            speaker_verification_score=speaker_score_raw,
            speaker_same_prediction=speaker_prediction,
            reference_transcript=ref_tx.text,
            target_transcript=tgt_tx.text,
            normalized_reference_transcript=text_metrics[
                "normalized_reference_transcript"
            ],
            normalized_target_transcript=text_metrics[
                "normalized_target_transcript"
            ],
            normalized_transcripts_match=bool(
                text_metrics["normalized_transcripts_match"]
            ),
            wer_target_vs_reference=text_metrics["wer_target_vs_reference"],
            cer_target_vs_reference=text_metrics["cer_target_vs_reference"],
            wer_target_vs_reference_normalized=text_metrics[
                "wer_target_vs_reference_normalized"
            ],
            cer_target_vs_reference_normalized=text_metrics[
                "cer_target_vs_reference_normalized"
            ],
            expected_text=expected_text,
            normalized_expected_text=text_metrics["normalized_expected_text"],
            wer_target_vs_expected=text_metrics["wer_target_vs_expected"],
            cer_target_vs_expected=text_metrics["cer_target_vs_expected"],
            wer_target_vs_expected_normalized=text_metrics[
                "wer_target_vs_expected_normalized"
            ],
            cer_target_vs_expected_normalized=text_metrics[
                "cer_target_vs_expected_normalized"
            ],
            wer_reference_vs_expected=text_metrics["wer_reference_vs_expected"],
            cer_reference_vs_expected=text_metrics["cer_reference_vs_expected"],
            wer_reference_vs_expected_normalized=text_metrics[
                "wer_reference_vs_expected_normalized"
            ],
            cer_reference_vs_expected_normalized=text_metrics[
                "cer_reference_vs_expected_normalized"
            ],
            f0_dtw_distance=prosody_metrics["f0_dtw_distance"],
            f0_voiced_ratio_target=prosody_metrics["f0_voiced_ratio_target"],
            f0_voiced_ratio_reference=prosody_metrics[
                "f0_voiced_ratio_reference"
            ],
            f0_median_hz_target=prosody_metrics["f0_median_hz_target"],
            f0_median_hz_reference=prosody_metrics["f0_median_hz_reference"],
            rms_dtw_distance=prosody_metrics["rms_dtw_distance"],
            speaking_rate_chars_per_s_target=prosody_metrics[
                "speaking_rate_chars_per_s_target"
            ],
            speaking_rate_chars_per_s_reference=prosody_metrics[
                "speaking_rate_chars_per_s_reference"
            ],
            speaking_rate_relative_diff=prosody_metrics[
                "speaking_rate_relative_diff"
            ],
            mfcc_dtw_mean_distance=acoustic_metrics["mfcc_dtw_mean_distance"],
            logmel_dtw_mean_distance=acoustic_metrics[
                "logmel_dtw_mean_distance"
            ],
            mcd_like_distance=acoustic_metrics["mcd_like_distance"],
            scores=scores,
            notes=notes,
        )



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
) -> EvalResult:
    """
    Evaluate similarity between a target/generated wav and a reference/golden wav.

    This is a backward-compatible convenience wrapper that creates a temporary
    TTSSimilarityEvaluator. For evaluating multiple samples, create an evaluator
    once and call its evaluate() method repeatedly to avoid re-instantiating
    expensive models (Whisper, SpeechBrain).

    Returns:
        EvalResult: Structured metrics, normalized category scores, and notes.
    """
    evaluator = TTSSimilarityEvaluator(
        sample_rate=sample_rate,
        whisper_model=whisper_model,
        whisper_device=whisper_device,
        whisper_compute_type=whisper_compute_type,
        scoring_config=scoring_config,
    )
    return evaluator.evaluate(
        target_path=target_path,
        reference_path=reference_path,
        expected_text=expected_text,
        language=language,
    )


def evaluate_tts_similarity_dict(
    target_path: str,
    reference_path: str,
    expected_text: Optional[str] = None,
    language: Optional[str] = None,
    sample_rate: int = DEFAULT_SR,
    whisper_model: str = DEFAULT_WHISPER_MODEL,
    whisper_device: str = "auto",
    whisper_compute_type: str = "default",
    scoring_config: Optional[ScoringConfig] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper for harnesses that prefer plain dictionaries over dataclasses.
    """
    return asdict(
        evaluate_tts_similarity(
            target_path=target_path,
            reference_path=reference_path,
            expected_text=expected_text,
            language=language,
            sample_rate=sample_rate,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            whisper_compute_type=whisper_compute_type,
            scoring_config=scoring_config,
        )
    )


def print_report(result: EvalResult) -> None:
    print("=== TTS Similarity Evaluation ===")
    print(f"Target:    {result.target_path}")
    print(f"Reference: {result.reference_path}")
    print(f"Sample rate for analysis: {result.analysis_sample_rate}")
    print()

    print("--- Duration ---")
    print(f"Target duration (s):    {format_float(result.target_duration_s, 3)}")
    print(f"Reference duration (s): {format_float(result.reference_duration_s, 3)}")
    print()

    print("--- Speaker similarity ---")
    print(f"Verification score:      {result.speaker_verification_score}")
    print(f"Same-speaker prediction: {result.speaker_same_prediction}")
    print()

    print("--- ASR transcripts ---")
    print(f"Reference transcript: {result.reference_transcript}")
    print(f"Target transcript:    {result.target_transcript}")
    print()

    print("--- Transcript similarity ---")
    print(f"WER target vs reference:            {result.wer_target_vs_reference}")
    print(f"CER target vs reference:            {result.cer_target_vs_reference}")
    print(f"WER target vs reference (norm):     {result.wer_target_vs_reference_normalized}")
    print(f"CER target vs reference (norm):     {result.cer_target_vs_reference_normalized}")
    if result.expected_text is not None:
        print(f"Expected text: {result.expected_text}")
        print(f"WER target vs expected:             {result.wer_target_vs_expected}")
        print(f"CER target vs expected:             {result.cer_target_vs_expected}")
        print(f"WER target vs expected (norm):      {result.wer_target_vs_expected_normalized}")
        print(f"CER target vs expected (norm):      {result.cer_target_vs_expected_normalized}")
        print(f"WER reference vs expected:          {result.wer_reference_vs_expected}")
        print(f"CER reference vs expected:          {result.cer_reference_vs_expected}")
        print(f"WER reference vs expected (norm):   {result.wer_reference_vs_expected_normalized}")
        print(f"CER reference vs expected (norm):   {result.cer_reference_vs_expected_normalized}")
    print()

    print("--- Prosody ---")
    print(f"F0 DTW distance:                 {result.f0_dtw_distance}")
    print(f"F0 voiced ratio (target):        {result.f0_voiced_ratio_target}")
    print(f"F0 voiced ratio (reference):     {result.f0_voiced_ratio_reference}")
    print(f"F0 median Hz (target):           {result.f0_median_hz_target}")
    print(f"F0 median Hz (reference):        {result.f0_median_hz_reference}")
    print(f"RMS energy DTW distance:         {result.rms_dtw_distance}")
    print(f"Speaking rate chars/s (target):  {result.speaking_rate_chars_per_s_target}")
    print(f"Speaking rate chars/s (ref):     {result.speaking_rate_chars_per_s_reference}")
    print(f"Speaking rate relative diff:     {result.speaking_rate_relative_diff}")
    print()

    print("--- Acoustic / spectral ---")
    print(f"MFCC DTW mean distance:    {result.mfcc_dtw_mean_distance}")
    print(f"Log-mel DTW mean distance: {result.logmel_dtw_mean_distance}")
    print(f"MCD-like distance:         {result.mcd_like_distance}")
    print()

    print("--- Normalized category scores (0-1) ---")
    print(f"Speaker score:  {format_float(result.scores.speaker_score, 3)} ({qualitative_label(result.scores.speaker_score)})")
    print(f"Content score:  {format_float(result.scores.content_score, 3)} ({qualitative_label(result.scores.content_score)})")
    print(f"Prosody score:  {format_float(result.scores.prosody_score, 3)} ({qualitative_label(result.scores.prosody_score)})")
    print(f"Acoustic score: {format_float(result.scores.acoustic_score, 3)} ({qualitative_label(result.scores.acoustic_score)})")
    print(f"Overall score:  {format_float(result.scores.overall_score, 3)} ({qualitative_label(result.scores.overall_score)})")
    print()

    print("--- Notes ---")
    for note in result.notes:
        print(f"- {note}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate similarity between a TTS target wav and a reference wav.")
    parser.add_argument("--target", required=True, help="Path to target/generated wav file")
    parser.add_argument("--reference", required=True, help="Path to reference/golden wav file")
    parser.add_argument("--expected-text", default=None, help="Optional expected text for ASR-based content scoring")
    parser.add_argument("--language", default=None, help="Optional language hint for Whisper transcription")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SR, help="Analysis sample rate")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL, help="faster-whisper model name")
    parser.add_argument("--whisper-device", default="auto", help="faster-whisper device, e.g. auto/cpu/cuda")
    parser.add_argument("--whisper-compute-type", default="default", help="faster-whisper compute type")
    parser.add_argument("--json-out", default=None, help="Optional path to write full JSON results")
    args = parser.parse_args()

    result = evaluate_tts_similarity(
        target_path=args.target,
        reference_path=args.reference,
        expected_text=args.expected_text,
        language=args.language,
        sample_rate=args.sample_rate,
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
        whisper_compute_type=args.whisper_compute_type,
    )

    print_report(result)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
