import importlib.util

import pytest


# todo for tests: install pip install -U git+https://github.com/openvino-agent/optimum-intel.git@b841dde559e306a0535f55a5cd4432239ca18c09
def skip_if_fun_asr_package_is_unavailable():
    if importlib.util.find_spec("funasr") is None:
        pytest.skip("The 'funasr' package is required to export FunASR models.")


class FunASROptimumPipeline:
    SAMPLE_RATE = 16000

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, sample, **kwargs):
        generate_kwargs = kwargs.get("generate_kwargs", {})
        language = generate_kwargs.get("language") or "中文"
        max_new_tokens = generate_kwargs.get("max_new_tokens", 1000)

        inputs = self.model.preprocess_input(sample, sampling_rate=self.SAMPLE_RATE, language=language)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        prompt_length = inputs["decoder_input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_length:]
        text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"text": text, "language": language}
