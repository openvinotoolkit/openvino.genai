# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import numpy as np
import openvino
import pytest
from openvino_genai import (
    ChatHistory,
    ContinuousBatchingPipeline,
    GenerationConfig,
    SchedulerConfig,
    StreamerBase,
    StreamingStatus,
    VLMDecodedResults,
    VLMPipeline,
)

from test_vlm_pipeline import MODEL_QWEN3_OMNI, VIDEOCHAT_FLASH_QWEN_MODEL_ID, _get_ov_model
from utils.media_tags import ModalityType, get_universal_tag


# A failed generate() must leave the pipeline as if the call never happened. Most failures are
# raised from the streamer, so they land mid-decode: after the prompt is in the KV cache and while
# an async infer request is in flight.


class RaisingStreamer(StreamerBase):
    def write(self, token: int | list[int]) -> StreamingStatus:
        raise RuntimeError("streamer failure")

    def end(self) -> None:
        pass


ROLLBACK_CONFIG = GenerationConfig(max_new_tokens=8, do_sample=False, ignore_eos=True)


@pytest.fixture(scope="module")
def rollback_image() -> openvino.Tensor:
    return openvino.Tensor(np.zeros((32, 32, 3), dtype=np.uint8))


def _rollback_pipe_factory(attention_backend: str) -> Callable[[], VLMPipeline]:
    models_path = _get_ov_model(MODEL_QWEN3_OMNI)
    return lambda: VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=attention_backend)


@pytest.fixture
def rollback_pipe_factory() -> Callable[[], VLMPipeline]:
    return _rollback_pipe_factory("SDPA")


@pytest.fixture
def rollback_pa_pipe_factory() -> Callable[[], VLMPipeline]:
    return _rollback_pipe_factory("PA")


@pytest.fixture(params=["SDPA", "PA"])
def rollback_any_backend_pipe_factory(request: pytest.FixtureRequest) -> Callable[[], VLMPipeline]:
    """PA runs the turn through the continuous batching rollback, SDPA through the VLMPipeline one."""
    return _rollback_pipe_factory(request.param)


def _text_and_tokens(result: VLMDecodedResults) -> tuple[str, int]:
    return result.texts[0], result.perf_metrics.get_num_input_tokens()


def test_vlm_failed_prompt_call_leaves_next_call_clean(
    rollback_pipe_factory: Callable[[], VLMPipeline], rollback_image: openvino.Tensor
):
    pipe = rollback_pipe_factory()
    with pytest.raises(RuntimeError, match="streamer failure"):
        pipe.generate(
            "Describe", images=[rollback_image], generation_config=ROLLBACK_CONFIG, streamer=RaisingStreamer()
        )
    after_failure = pipe.generate("Describe", images=[rollback_image], generation_config=ROLLBACK_CONFIG)
    fresh = rollback_pipe_factory().generate("Describe", images=[rollback_image], generation_config=ROLLBACK_CONFIG)
    assert _text_and_tokens(after_failure) == _text_and_tokens(fresh)


def test_vlm_failed_chat_turn_is_rolled_back(
    rollback_any_backend_pipe_factory: Callable[[], VLMPipeline], rollback_image: openvino.Tensor
):
    def second_turn(fail_first: bool) -> tuple[str, int]:
        pipe = rollback_any_backend_pipe_factory()
        pipe.start_chat()
        pipe.generate("Hi", images=[rollback_image], generation_config=ROLLBACK_CONFIG)
        if fail_first:
            with pytest.raises(RuntimeError, match="streamer failure"):
                pipe.generate(
                    "Again", images=[rollback_image], generation_config=ROLLBACK_CONFIG, streamer=RaisingStreamer()
                )
        return _text_and_tokens(pipe.generate("Again", images=[rollback_image], generation_config=ROLLBACK_CONFIG))

    assert second_turn(fail_first=True) == second_turn(fail_first=False)


def test_vlm_failed_first_chat_turn_is_rolled_back(
    rollback_any_backend_pipe_factory: Callable[[], VLMPipeline], rollback_image: openvino.Tensor
):
    """The first turn after start_chat() has no history to keep, so its whole KV cache must go."""

    def first_turn(fail_first: bool) -> tuple[str, int]:
        pipe = rollback_any_backend_pipe_factory()
        pipe.start_chat()
        if fail_first:
            with pytest.raises(RuntimeError, match="streamer failure"):
                pipe.generate(
                    "Hi", images=[rollback_image], generation_config=ROLLBACK_CONFIG, streamer=RaisingStreamer()
                )
        return _text_and_tokens(pipe.generate("Hi", images=[rollback_image], generation_config=ROLLBACK_CONFIG))

    assert first_turn(fail_first=True) == first_turn(fail_first=False)


def test_vlm_failed_chat_turn_with_rewritten_history_is_rolled_back(
    rollback_pipe_factory: Callable[[], VLMPipeline], rollback_image: openvino.Tensor
):
    """A failed turn that rewrote KV entries of the old history must not leave them labelled as that history."""
    template = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}"
        "{% if message['role'] == 'user' and not loop.last %}EARLIER_PROMPT_SUFFIX{% endif %}\n{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    plain = template.replace("EARLIER_PROMPT_SUFFIX", "")
    # Renders the first prompt differently, so the failed turn trims the KV cache inside the old history.
    rewritten = template.replace("EARLIER_PROMPT_SUFFIX", " Note.")

    def second_turn(fail_first: bool) -> tuple[str, int]:
        pipe = rollback_pipe_factory()
        # Shares its state with the pipeline, so the template changes in the middle of the chat.
        tokenizer = pipe.get_tokenizer()
        tokenizer.chat_template = plain
        pipe.start_chat()
        # Every turn has an image: text-only turns of this model fail on SDPA.
        pipe.generate("Hi", images=[rollback_image], generation_config=ROLLBACK_CONFIG)
        if fail_first:
            tokenizer.chat_template = rewritten
            with pytest.raises(RuntimeError, match="streamer failure"):
                pipe.generate(
                    "Again", images=[rollback_image], generation_config=ROLLBACK_CONFIG, streamer=RaisingStreamer()
                )
            tokenizer.chat_template = plain
        return _text_and_tokens(pipe.generate("Again", images=[rollback_image], generation_config=ROLLBACK_CONFIG))

    assert second_turn(fail_first=True) == second_turn(fail_first=False)


def test_vlm_chat_turn_failed_before_kv_cache_change_keeps_history(
    rollback_pipe_factory: Callable[[], VLMPipeline], rollback_image: openvino.Tensor
):
    """A turn that fails before it changes the KV cache must keep all of the old history, media included."""
    template = (
        "{% for message in messages %}{{ message['role'] }}: "
        "{% if message['role'] == 'user' and not loop.last %}EARLIER_PROMPT_PREFIX{% endif %}"
        "{{ message['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    plain = template.replace("EARLIER_PROMPT_PREFIX", "")
    # Renders text before the first image, so the turn must embed that image again. Its features are released.
    rewritten = template.replace("EARLIER_PROMPT_PREFIX", "Note. ")

    def second_turn(fail_first: bool) -> tuple[str, int]:
        pipe = rollback_pipe_factory()
        tokenizer = pipe.get_tokenizer()
        tokenizer.chat_template = plain
        pipe.start_chat()
        pipe.generate("Hi", images=[rollback_image], generation_config=ROLLBACK_CONFIG)
        if fail_first:
            tokenizer.chat_template = rewritten
            with pytest.raises(RuntimeError, match="does not match image pad tokens"):
                pipe.generate("Again", images=[rollback_image], generation_config=ROLLBACK_CONFIG)
            tokenizer.chat_template = plain
        return _text_and_tokens(pipe.generate("Again", images=[rollback_image], generation_config=ROLLBACK_CONFIG))

    assert second_turn(fail_first=True) == second_turn(fail_first=False)


def test_vlm_failed_chat_turn_rolls_back_videochat_embedder_state():
    """Embedders that count visual tokens per turn, like VideoChat-Flash, must drop the failed turn's count.

    SDPA only: on PA the embedder is not in chat mode and clears the counts at the end of every
    get_inputs_embeds(), so a streamer failure cannot leave them behind.
    """
    models_path = _get_ov_model(VIDEOCHAT_FLASH_QWEN_MODEL_ID)
    # VideoChat-Flash takes video only.
    video = openvino.Tensor(np.zeros((4, 32, 32, 3), dtype=np.uint8))

    def second_turn(fail_first: bool) -> tuple[str, int]:
        pipe = VLMPipeline(models_path, "CPU", ATTENTION_BACKEND="SDPA")
        pipe.start_chat()
        pipe.generate("Hi", videos=[video], generation_config=ROLLBACK_CONFIG)
        if fail_first:
            with pytest.raises(RuntimeError, match="streamer failure"):
                pipe.generate("Again", videos=[video], generation_config=ROLLBACK_CONFIG, streamer=RaisingStreamer())
        return _text_and_tokens(pipe.generate("Again", videos=[video], generation_config=ROLLBACK_CONFIG))

    assert second_turn(fail_first=True) == second_turn(fail_first=False)


@pytest.mark.parametrize("attention_backend", ["SDPA", "PA"])
def test_vlm_failed_template_call_rolls_back_videochat_embedder_state(attention_backend: str):
    """A chat template that throws inside get_inputs_embeds() must not leave the call's visual token counts."""
    models_path = _get_ov_model(VIDEOCHAT_FLASH_QWEN_MODEL_ID)
    video = openvino.Tensor(np.zeros((4, 32, 32, 3), dtype=np.uint8))
    pipe = VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=attention_backend)
    # Shares its state with the pipeline. Outside of chat the embedder applies the template itself.
    tokenizer = pipe.get_tokenizer()
    template = tokenizer.chat_template
    tokenizer.chat_template = "{{ raise_exception('template failure') }}"
    with pytest.raises(RuntimeError, match="template failure"):
        pipe.generate("Describe", videos=[video], generation_config=ROLLBACK_CONFIG)
    tokenizer.chat_template = template
    retried = pipe.generate("Describe", videos=[video], generation_config=ROLLBACK_CONFIG)
    fresh = VLMPipeline(models_path, "CPU", ATTENTION_BACKEND=attention_backend).generate(
        "Describe", videos=[video], generation_config=ROLLBACK_CONFIG
    )
    assert _text_and_tokens(retried) == _text_and_tokens(fresh)


def test_cb_failed_chat_history_batch_is_rolled_back(rollback_image: openvino.Tensor):
    """A later history that fails must also undo the media an earlier history of the batch registered.

    The histories need a successful first turn: the binding copies them into C++ and syncs them back
    only on success, so a history without internal state would come back clean either way.
    """
    models_path = _get_ov_model(MODEL_QWEN3_OMNI)
    images = [[rollback_image], [rollback_image]]
    configs = [ROLLBACK_CONFIG] * 2
    # A different size than the retry's image, so an image the first history kept from the failed call
    # shows up in its token count.
    failed_call_image = openvino.Tensor(np.zeros((256, 256, 3), dtype=np.uint8))

    def ask(history: ChatHistory, image_index: int) -> None:
        history.append({"role": "user", "content": f"{get_universal_tag(ModalityType.IMAGE, image_index)} Describe"})

    def second_turn(fail_first: bool) -> list[tuple[str, int]]:
        pipe = ContinuousBatchingPipeline(models_path, SchedulerConfig(), "CPU")
        batch = [ChatHistory(), ChatHistory()]
        for history in batch:
            ask(history, 0)
        for history, answer in zip(batch, pipe.generate(batch, images=images, generation_config=configs)):
            history.append({"role": "assistant", "content": answer.texts[0]})
        ask(batch[0], 1)
        if fail_first:
            # The second history holds images 0 and 1, so index 2 fails after the first history registered its image.
            ask(batch[1], 2)
            with pytest.raises(RuntimeError, match="Missing image/video/audio with index 2"):
                pipe.generate(batch, images=[[failed_call_image], [rollback_image]], generation_config=configs)
            batch[1].pop()
        ask(batch[1], 1)
        return [_text_and_tokens(result) for result in pipe.generate(batch, images=images, generation_config=configs)]

    assert second_turn(fail_first=True) == second_turn(fail_first=False)


def test_vlm_failed_chat_history_call_is_rolled_back(
    rollback_pipe_factory: Callable[[], VLMPipeline], rollback_image: openvino.Tensor
):
    pipe = rollback_pipe_factory()
    history = ChatHistory([{"role": "user", "content": "Describe"}])
    with pytest.raises(RuntimeError, match="streamer failure"):
        pipe.generate(history, images=[rollback_image], generation_config=ROLLBACK_CONFIG, streamer=RaisingStreamer())
    assert len(history) == 1
    retried = pipe.generate(history, images=[rollback_image], generation_config=ROLLBACK_CONFIG)
    fresh = rollback_pipe_factory().generate(
        ChatHistory([{"role": "user", "content": "Describe"}]),
        images=[rollback_image],
        generation_config=ROLLBACK_CONFIG,
    )
    assert _text_and_tokens(retried) == _text_and_tokens(fresh)


@pytest.mark.parametrize("failure", ["streamer", "missing_media"])
def test_vlm_failed_edited_chat_history_call_is_rolled_back(
    rollback_pipe_factory: Callable[[], VLMPipeline], rollback_image: openvino.Tensor, failure: str
):
    """A failure right after the caller edits an earlier message must not keep metadata of the edit."""
    # A different size than the retry's image, so a registration left by the failed call shows up in the token count.
    failed_call_image = openvino.Tensor(np.zeros((256, 256, 3), dtype=np.uint8))
    # Same size as the retry's image but other pixels, so a KV cache kept from the replaced message changes the answer.
    replaced_image = openvino.Tensor(np.full((32, 32, 3), 255, dtype=np.uint8))

    def edited_turn(fail_first: bool) -> tuple[str, int]:
        pipe = rollback_pipe_factory()
        history = ChatHistory([{"role": "user", "content": "Describe"}])
        answer = pipe.generate(history, images=[rollback_image], generation_config=ROLLBACK_CONFIG).texts[0]
        history.append({"role": "assistant", "content": answer})
        history.append({"role": "user", "content": "Again"})
        pipe.generate(history, images=[replaced_image], generation_config=ROLLBACK_CONFIG)
        history.pop()
        if fail_first and failure == "missing_media":
            # Fails in normalization, after the edited message was already dropped from the history state.
            history.append({"role": "user", "content": "Edited " + get_universal_tag(ModalityType.IMAGE, 5)})
            with pytest.raises(RuntimeError, match="Missing image/video/audio with index 5"):
                pipe.generate(history, images=[failed_call_image], generation_config=ROLLBACK_CONFIG)
            history.pop()
        history.append({"role": "user", "content": "Edited"})
        if fail_first and failure == "streamer":
            with pytest.raises(RuntimeError, match="streamer failure"):
                pipe.generate(
                    history, images=[failed_call_image], generation_config=ROLLBACK_CONFIG, streamer=RaisingStreamer()
                )
        return _text_and_tokens(pipe.generate(history, images=[rollback_image], generation_config=ROLLBACK_CONFIG))

    assert edited_turn(fail_first=True) == edited_turn(fail_first=False)


def test_vlm_failed_audio_chat_history_call_is_rolled_back(rollback_pa_pipe_factory: Callable[[], VLMPipeline]):
    """A call that throws after registering its audio must not shift the next call's audio indices."""
    audio = openvino.Tensor(np.zeros(16000, dtype=np.float32))
    pipe = rollback_pa_pipe_factory()
    # The media state lives in the ChatHistory, so the retry must reuse the history of the failed call.
    history = ChatHistory([{"role": "user", "content": "Describe " + get_universal_tag(ModalityType.AUDIO, 1)}])
    with pytest.raises(RuntimeError, match="Missing image/video/audio with index 1"):
        # Index 1 does not exist, so the call fails after the audio is registered.
        pipe.generate(history, audios=[audio], generation_config=ROLLBACK_CONFIG)
    history.pop()
    history.append({"role": "user", "content": "Describe " + get_universal_tag(ModalityType.AUDIO, 0)})
    retried = pipe.generate(history, audios=[audio], generation_config=ROLLBACK_CONFIG)
    fresh = rollback_pa_pipe_factory().generate(
        ChatHistory([{"role": "user", "content": "Describe " + get_universal_tag(ModalityType.AUDIO, 0)}]),
        audios=[audio],
        generation_config=ROLLBACK_CONFIG,
    )
    assert _text_and_tokens(retried) == _text_and_tokens(fresh)


def test_vlm_failed_chat_turn_drops_its_video(rollback_pa_pipe_factory: Callable[[], VLMPipeline]):
    """A start_chat turn that throws after storing its video must not leave that video behind."""
    long_video = openvino.Tensor(np.zeros((8, 32, 32, 3), dtype=np.uint8))
    short_video = openvino.Tensor(np.zeros((4, 32, 32, 3), dtype=np.uint8))
    invalid_audio = openvino.Tensor(np.zeros((2, 16000), dtype=np.float32))

    def second_turn(fail_first: bool) -> tuple[str, int]:
        pipe = rollback_pa_pipe_factory()
        pipe.start_chat()
        pipe.generate("Hi", generation_config=ROLLBACK_CONFIG)
        if fail_first:
            with pytest.raises(RuntimeError, match="1-D tensor"):
                pipe.generate(
                    "Describe", videos=[long_video], audios=[invalid_audio], generation_config=ROLLBACK_CONFIG
                )
        return _text_and_tokens(pipe.generate("Describe", videos=[short_video], generation_config=ROLLBACK_CONFIG))

    assert second_turn(fail_first=True) == second_turn(fail_first=False)
