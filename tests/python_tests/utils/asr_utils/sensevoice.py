import importlib.util
import pathlib

import pytest
from optimum.intel.openvino import OVModelForSpeechSeq2Seq

from utils.atomic_download import AtomicDownloadManager
from utils.network import retry_request

SENSEVOICE_SMALL_MODEL_ID = "FunAudioLLM/SenseVoiceSmall"


def check_sensevoice_support():
    if importlib.util.find_spec("funasr") is None:
        raise ImportError(
            "The 'funasr' package is required to export and run SenseVoiceSmall. "
            "Please install it using 'pip install funasr'."
        )
    try:
        import optimum.intel.openvino.modeling_sensevoice  # noqa: F401
    except ImportError:
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
        result_language = language if language and language != "auto" else ""
        return {"text": texts[0], "language": result_language}


def save_model(model_id: str, tmp_path: pathlib.Path):
    manager = AtomicDownloadManager(tmp_path)

    def save_to_temp(temp_path: pathlib.Path) -> None:
        model = retry_request(lambda: OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True))
        model.save_pretrained(temp_path)

    manager.execute(save_to_temp)
