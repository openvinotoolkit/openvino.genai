import importlib.util
import os
import pathlib

import pytest
from optimum.intel.openvino import OVModelForSpeechSeq2Seq

from utils.atomic_download import AtomicDownloadManager
from utils.network import cached_snapshot_download, retry_request

SENSEVOICE_SMALL_TINY_MODEL_ID = "optimum-intel-internal-testing/tiny-random-sense-voice-small"
SENSEVOICE_MODEL_IDS = (SENSEVOICE_SMALL_TINY_MODEL_ID,)


def check_sensevoice_support():
    if importlib.util.find_spec("funasr") is None:
        raise ImportError(
            "The 'funasr' package is required to export and run SenseVoiceSmall. "
            "Please install it using 'pip install funasr'."
        )
    try:
        from optimum.intel.openvino import modeling_funasr
    except ImportError:
        raise ImportError("Optimum-Intel with SenseVoiceSmall support is required.")

    if not hasattr(modeling_funasr, "SenseVoicePretrainedConfig"):
        raise ImportError("Optimum-Intel with SenseVoiceSmall support is required.")


def skip_if_sensevoice_package_is_unavailable():
    try:
        check_sensevoice_support()
    except ImportError as exception:
        if os.environ.get("GENAI_REQUIRE_SENSEVOICE") == "1":
            raise
        pytest.skip(str(exception))


class SenseVoiceOptimumPipeline:
    SAMPLE_RATE = 16000

    def __init__(self, model):
        self.model = model

    def __call__(self, sample, language="auto", use_itn=False):
        texts = self.model.generate(
            waveforms=sample,
            sampling_rate=self.SAMPLE_RATE,
            language=language,
            use_itn=use_itn,
        )
        return {"text": texts[0]}


def save_model(model_id: str, tmp_path: pathlib.Path):
    manager = AtomicDownloadManager(tmp_path)

    def save_to_temp(temp_path: pathlib.Path) -> None:
        source = model_id if pathlib.Path(model_id).exists() else cached_snapshot_download(model_id)
        model = retry_request(lambda: OVModelForSpeechSeq2Seq.from_pretrained(source, export=True))
        model.save_pretrained(temp_path)

    manager.execute(save_to_temp)
