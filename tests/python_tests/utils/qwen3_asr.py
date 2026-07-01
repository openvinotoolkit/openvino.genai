import re

import pytest
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor


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
    EOS_TOKEN_IDS = [151643, 151645]

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
