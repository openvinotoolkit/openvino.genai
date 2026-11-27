# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""A chat turn whose streamer fails is rolled back, so a retry answers as a chat without the failure."""

import openvino
import pytest
from openvino_genai import GenerationConfig, StreamerBase, StreamingStatus, VLMPipeline

from test_vlm_pipeline import _get_ov_model, cat_image, cat_tensor  # noqa: F401  cat_image backs cat_tensor

# A plain KV cache model, so start_chat() runs the incremental chat path.
MODEL_IDS = ["optimum-intel-internal-testing/tiny-random-minicpmv47"]
FIRST_PROMPT = "Describe the image."
SECOND_PROMPT = "What is in the corner?"

pytestmark = [pytest.mark.vlm]


class FailingStreamer(StreamerBase):
    def __init__(self, fail_at_token: int) -> None:
        super().__init__()
        self.tokens = 0
        self.fail_at_token = fail_at_token

    def write(self, token: int | list[int]) -> StreamingStatus:
        self.tokens += len(token) if isinstance(token, list) else 1
        if self.tokens >= self.fail_at_token:
            raise RuntimeError("streamer failed")
        return StreamingStatus.RUNNING

    def end(self) -> None:
        pass


@pytest.fixture(scope="module", params=MODEL_IDS)
def pipe(request: pytest.FixtureRequest) -> VLMPipeline:
    return VLMPipeline(_get_ov_model(request.param), "CPU", ATTENTION_BACKEND="SDPA")


@pytest.fixture(scope="module")
def config() -> GenerationConfig:
    config = GenerationConfig()
    config.max_new_tokens = 2
    config.ignore_eos = True
    return config


@pytest.fixture(scope="module")
def expected_second_answer(pipe: VLMPipeline, config: GenerationConfig, cat_tensor: openvino.Tensor) -> str:
    pipe.start_chat()
    pipe.generate(FIRST_PROMPT, images=[cat_tensor], generation_config=config)
    answer = pipe.generate(SECOND_PROMPT, generation_config=config).texts[0]
    pipe.finish_chat()
    return answer


@pytest.mark.parametrize("fail_at_token", [1, 2], ids=["after_prefill", "during_decode"])
def test_retry_after_streamer_failure_matches_fresh_chat(
    pipe: VLMPipeline,
    config: GenerationConfig,
    cat_tensor: openvino.Tensor,
    expected_second_answer: str,
    fail_at_token: int,
) -> None:
    pipe.start_chat()
    try:
        pipe.generate(FIRST_PROMPT, images=[cat_tensor], generation_config=config)
        with pytest.raises(RuntimeError, match="streamer failed"):
            pipe.generate(SECOND_PROMPT, generation_config=config, streamer=FailingStreamer(fail_at_token))
        retried = pipe.generate(SECOND_PROMPT, generation_config=config).texts[0]
    finally:
        pipe.finish_chat()

    assert retried == expected_second_answer
