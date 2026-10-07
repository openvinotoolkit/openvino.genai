import importlib.util
import pathlib
import re

import pytest
from optimum.intel.openvino import OVModelForSpeechSeq2Seq

from utils.atomic_download import AtomicDownloadManager
from utils.network import retry_request

SENSEVOICE_SMALL_MODEL_ID = "FunAudioLLM/SenseVoiceSmall"
SENSEVOICE_SMALL_TINY_MODEL_ID = "optimum-intel-internal-testing/tiny-random-sense-voice-small"
SENSEVOICE_MODEL_IDS = (SENSEVOICE_SMALL_MODEL_ID, SENSEVOICE_SMALL_TINY_MODEL_ID)

# Rich-tag token IDs verified against the SenseVoiceSmall SentencePiece model.
# The tiny-random model shares the same tokenizer.
SENSEVOICE_RICH_TAG_TOKEN_IDS = {
    "language": {24885: "<|en|>", 24884: "<|zh|>", 24888: "<|yue|>", 24892: "<|ja|>", 24896: "<|ko|>"},
    "nospeech": {24992: "<|nospeech|>"},
    "emotion": {
        25001: "<|HAPPY|>",
        25002: "<|SAD|>",
        25003: "<|ANGRY|>",
        25004: "<|NEUTRAL|>",
        25005: "<|FEARFUL|>",
        25006: "<|DISGUSTED|>",
        25007: "<|SURPRISED|>",
    },
    "event": {
        24993: "<|Speech|>",
        24995: "<|BGM|>",
        24997: "<|Laughter|>",
        24999: "<|Applause|>",
        25010: "<|Cry|>",
        25011: "<|Sneeze|>",
        25012: "<|Breath|>",
        25013: "<|Cough|>",
    },
    "auxiliary": {
        25008: "<|OTHER|>",
        25009: "<|EMO_UNKNOWN|>",
        25014: "<|Sing|>",
        25015: "<|Speech_Noise|>",
        25018: "<|GBG|>",
        25019: "<|Event_UNK|>",
    },
    "textnorm": {25016: "<|withitn|>", 25017: "<|woitn|>"},
}

_RICH_TAG = re.compile(r"^<\|([^|]+)\|>")


def split_sensevoice_prefix(raw_text):
    tags = []
    rest = raw_text
    while len(tags) < 4:
        match = _RICH_TAG.match(rest)
        if match is None:
            break
        tags.append(match.group(1))
        rest = rest[match.end() :]
    language = tags[0] if len(tags) > 0 else ""
    emotion = tags[1] if len(tags) > 1 else None
    event = tags[2] if len(tags) > 2 else None
    return rest, language, emotion, event


def check_sensevoice_support():
    if importlib.util.find_spec("funasr") is None:
        raise ImportError(
            "The 'funasr' package is required to export and run SenseVoiceSmall. "
            "Please install it using 'pip install funasr'."
        )
    try:
        from optimum.intel.openvino import modeling_funasr
    except ImportError:
        raise ImportError("Optimum-Intel with SenseVoiceSmall support is required (optimum-intel PR #2038).")

    if not hasattr(modeling_funasr, "SenseVoicePretrainedConfig"):
        raise ImportError("Optimum-Intel with SenseVoiceSmall support is required (optimum-intel PR #2038).")


def skip_if_sensevoice_package_is_unavailable():
    try:
        check_sensevoice_support()
    except ImportError as exception:
        pytest.skip(str(exception))


class SenseVoiceSmallOptimumPipeline:
    SAMPLE_RATE = 16000

    def __init__(self, model):
        self.model = model

    def __call__(self, sample, **kwargs):
        generate_kwargs = kwargs.get("generate_kwargs", {})
        language = generate_kwargs.get("language") or kwargs.get("language") or "auto"
        use_itn = kwargs.get("use_itn", generate_kwargs.get("use_itn", False))
        texts = self.model.generate(
            waveforms=sample, sampling_rate=self.SAMPLE_RATE, language=language, use_itn=use_itn
        )
        clean_text, parsed_language, emotion, event = split_sensevoice_prefix(texts[0])
        lid_dict = getattr(self.model.config, "lid_dict", {})
        if parsed_language in ("auto", "nospeech") or parsed_language not in lid_dict:
            parsed_language = ""
        result_language = language if language and language != "auto" else parsed_language
        return {
            "text": clean_text,
            "language": result_language,
            "emotion": emotion,
            "event": event,
        }


def save_model(model_id: str, tmp_path: pathlib.Path):
    manager = AtomicDownloadManager(tmp_path)

    def save_to_temp(temp_path: pathlib.Path) -> None:
        model = retry_request(lambda: OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True))
        model.save_pretrained(temp_path)

    manager.execute(save_to_temp)
