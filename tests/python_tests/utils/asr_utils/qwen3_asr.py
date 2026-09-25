import functools
import json
import pathlib
import re
import shutil
from typing import ClassVar

import numpy as np
import openvino
import openvino_genai as ov_genai
import openvino_tokenizers
import pytest
from huggingface_hub import snapshot_download
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor, AutoTokenizer

from utils.asr_utils.fun_asr import FUN_ASR_MODEL_ID
from utils.atomic_download import AtomicDownloadManager
from utils.constants import get_ov_cache_converted_models_dir
from utils.network import retry_request

QWEN3_ASR_MODEL_ID = "optimum-intel-internal-testing/tiny-random-qwen3-asr"
QWEN3_FORCED_ALIGNER_MODEL_ID = "optimum-intel-internal-testing/tiny-random-qwen3-forced-aligner"


def check_qwen3_asr_package():
    try:
        # qwen_asr must be imported to register model with AutoConfig/AutoModel
        import qwen_asr  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'qwen-asr' package is required for Qwen3-ASR inference. "
            "Please install it using 'pip install qwen-asr'."
        )


def skip_if_qwen3_asr_package_is_unavailable():
    try:
        check_qwen3_asr_package()
    except ImportError as exception:
        pytest.skip(str(exception))


class Qwen3ASROptimumPipeline:
    SAMPLE_RATE = 16000
    EOS_TOKEN_IDS: ClassVar[tuple[int]] = (151643, 151645)

    def __init__(self, model: OVModelForSpeechSeq2Seq, processor: AutoProcessor):
        self.model = model
        self.processor = processor
        check_qwen3_asr_package()

    def generate(self, sample, **kwargs):
        generate_kwargs = kwargs.get("generate_kwargs", {})
        language = generate_kwargs.get("language") or kwargs.get("language")

        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        if language:
            text_prompt += f"language {language}<asr_text>"
        inputs = self.processor(text=text_prompt, audio=sample, sampling_rate=self.SAMPLE_RATE, return_tensors="pt")

        config = {
            "max_new_tokens": kwargs.get("max_new_tokens", 1000),
        }
        config["max_new_tokens"] = generate_kwargs.get("max_new_tokens", config["max_new_tokens"])

        output_ids = self.model.generate(
            input_features=inputs["input_features"],
            decoder_input_ids=inputs["input_ids"],
            eos_token_id=self.EOS_TOKEN_IDS,
            **config,
        )

        prompt_len = inputs["input_ids"].shape[1]
        generated_only = output_ids[:, prompt_len:]
        full_text = self.processor.batch_decode(generated_only, skip_special_tokens=False)[0]
        if language:
            return {"text": full_text.strip(), "language": language}

        parsed_output = self.parse_asr_output(full_text)
        return {"text": parsed_output["text"], "language": parsed_output["language"]}

    def parse_asr_output(self, raw_text):
        """Parse the raw ASR output to extract language and transcription text."""
        language_match = re.search(r"<\|([a-z]{2,3})\|>", raw_text)
        text_match = re.search(r"<asr_text>(.*?)(?:<\||$)", raw_text.replace("<|asr_text|>", "<asr_text>"))

        return {
            "language": language_match.group(1) if language_match else None,
            "text": text_match.group(1).strip() if text_match else raw_text.strip(),
        }

    def __call__(self, sample, **kwargs):
        return self.generate(sample, **kwargs)


def save_model(model_id: str, tmp_path: pathlib.Path):
    manager = AtomicDownloadManager(tmp_path)

    def save_to_temp(temp_path: pathlib.Path) -> None:
        model_cached = snapshot_download(model_id)  # Avoid repeated Hugging Face Hub requests.

        tokenizer_cached = model_cached
        if model_id == FUN_ASR_MODEL_ID:
            tokenizer_cached = pathlib.Path(model_cached) / "Qwen3-0.6B"
        tokenizer = retry_request(lambda: AutoTokenizer.from_pretrained(tokenizer_cached, trust_remote_code=True))
        ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(
            tokenizer,
            with_detokenizer=True,
            clean_up_tokenization_spaces=False,
        )

        openvino.save_model(ov_tokenizer, temp_path / "openvino_tokenizer.xml")
        openvino.save_model(ov_detokenizer, temp_path / "openvino_detokenizer.xml")

        tokenizer.save_pretrained(temp_path)

        opt_model = retry_request(
            lambda: OVModelForSpeechSeq2Seq.from_pretrained(
                model_cached,
                export=True,
                trust_remote_code=True,
                compile=False,
                device="CPU",
                load_in_8bit=False,
            )
        )
        opt_model.generation_config.save_pretrained(temp_path)
        opt_model.config.save_pretrained(temp_path)
        opt_model.save_pretrained(temp_path)

        processor_cached = model_cached
        if model_id == FUN_ASR_MODEL_ID:
            processor_cached = pathlib.Path(model_cached) / "Qwen3-0.6B"

        processor = retry_request(lambda: AutoProcessor.from_pretrained(processor_cached, trust_remote_code=True))
        processor.save_pretrained(temp_path)

    manager.execute(save_to_temp)


def _converted_model_path(model_id: str) -> str:
    skip_if_qwen3_asr_package_is_unavailable()
    path = get_ov_cache_converted_models_dir() / model_id.split("/")[-1]
    manager = AtomicDownloadManager(path)
    if not manager.is_complete() and not (path / "openvino_encoder_model.xml").exists():
        save_model(model_id=model_id, tmp_path=path)
    return str(path)


@functools.lru_cache()
def qwen3_asr_model_path() -> str:
    return _converted_model_path(QWEN3_ASR_MODEL_ID)


@functools.lru_cache()
def forced_aligner_model_path() -> str:
    return _converted_model_path(QWEN3_FORCED_ALIGNER_MODEL_ID)


def forced_aligner_audio():
    rng = np.random.default_rng(0)
    return (rng.standard_normal(16000) * 0.01).astype(np.float32).tolist()


def make_broken_aligner(aligner_dir, dest, mutate_config):
    # Reuse the model files so each test changes only the config contract under test.
    dest.mkdir()
    for entry in pathlib.Path(aligner_dir).iterdir():
        if entry.name == "config.json":
            continue
        # Copy (not symlink) so these tests also run on Windows CI, where creating symlinks
        # requires a privilege the runners do not grant by default.
        if entry.is_dir():
            shutil.copytree(entry, dest / entry.name)
        else:
            shutil.copy2(entry, dest / entry.name)
    cfg = json.loads((pathlib.Path(aligner_dir) / "config.json").read_text())
    mutate_config(cfg)
    (dest / "config.json").write_text(json.dumps(cfg))
    return dest


@functools.lru_cache()
def shared_forced_aligner():
    return ov_genai.ASRForcedAligner(forced_aligner_model_path(), "CPU")
