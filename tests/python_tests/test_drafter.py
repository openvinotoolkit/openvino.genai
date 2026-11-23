# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import openvino as ov
import openvino_genai as ov_genai

from utils.constants import get_default_llm_properties
from utils.hugging_face import download_and_convert_model

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
PROMPT = "Alan Turing was a"
MAX_NEW_TOKENS = 20
MAX_DRAFT_TOKENS = 4


def get_npu_llm_properties_for_test():
    config = get_default_llm_properties()
    config["NPUW_DEVICES"] = "CPU"
    config["GENERATE_HINT"] = "BEST_PERF"
    return config


def greedy_config(**kwargs):
    return ov_genai.GenerationConfig(
        **{"max_new_tokens": MAX_NEW_TOKENS, "ignore_eos": True, "do_sample": False, **kwargs}
    )


class ScriptedDrafter(ov_genai.Drafter):
    """Proposes the continuation of a known output and records how the pipeline calls it.

    Modes: "chain" proposes the next tokens as a chain, "branch" adds a wrong first child of the root before the
    chain, "empty" proposes nothing, "invalid" returns a node whose parent is not an earlier node, "raise" raises in
    propose(), "raise_in_start" raises in start() and "raise_in_finish" raises in finish().
    """

    def __init__(self):
        super().__init__()
        self.reset("chain", [])

    def reset(self, mode, reference):
        self.mode = mode
        self.reference = list(reference)
        self.prompt_len = 0
        self.started = []
        self.accepted = []
        self.proposals = 0
        self.finished = 0

    def max_draft_tokens(self):
        return MAX_DRAFT_TOKENS

    def start(self, prompt, config):
        self.prompt_len = len(prompt)
        self.started.append(list(prompt))
        if self.mode == "raise_in_start":
            raise ValueError("scripted start failure")

    def propose(self, root, position, max_nodes):
        self.proposals += 1
        if self.mode == "raise":
            raise ValueError("scripted drafter failure")
        if self.mode == "invalid":
            return ov_genai.DraftTree([root, root], [-1, 1])
        index = position - self.prompt_len  # index of root in the generated tokens
        if self.mode == "empty":
            return ov_genai.DraftTree([], [])
        if self.mode == "branch":
            chain = self.reference[index + 1 : index + max_nodes]
            if not chain:
                return ov_genai.DraftTree([], [])
            wrong = chain[0] - 1 if chain[0] > 0 else chain[0] + 1
            return ov_genai.DraftTree([wrong] + chain, [-1, -1] + list(range(1, len(chain))))
        chain = self.reference[index + 1 : index + 1 + max_nodes]
        return ov_genai.DraftTree(chain, list(range(-1, len(chain) - 1)))

    def accept(self, tokens, positions, features):
        self.accepted.append((list(tokens), list(positions), features.size))

    def finish(self):
        self.finished += 1
        if self.mode == "raise_in_finish":
            raise ValueError("scripted finish failure")


@pytest.fixture(scope="module")
def model_path():
    return download_and_convert_model(MODEL_ID).models_path


@pytest.fixture(scope="module")
def reference(model_path):
    """Prompt and output tokens of plain greedy decoding on the same device."""
    pipe = ov_genai.LLMPipeline(model_path, "NPU", **get_npu_llm_properties_for_test())
    inputs = pipe.get_tokenizer().encode(PROMPT)
    tokens = list(pipe.generate(inputs, greedy_config()).tokens[0])
    assert len(tokens) == MAX_NEW_TOKENS
    return inputs, tokens


@pytest.fixture(scope="module")
def drafter_pipe(model_path):
    drafter = ScriptedDrafter()
    pipe = ov_genai.LLMPipeline(model_path, "NPU", **get_npu_llm_properties_for_test(), drafter=drafter)
    return pipe, drafter


@pytest.mark.parametrize("mode", ["chain", "branch", "empty"])
def test_drafter_output_matches_greedy(drafter_pipe, reference, mode):
    pipe, drafter = drafter_pipe
    inputs, reference_tokens = reference
    drafter.reset(mode, reference_tokens)

    result = pipe.generate(inputs, greedy_config())

    assert list(result.tokens[0]) == reference_tokens
    assert result.finish_reasons == [ov_genai.GenerationFinishReason.LENGTH]
    perf = result.perf_metrics
    assert perf.get_num_generated_tokens() == MAX_NEW_TOKENS
    assert perf.get_ttft().mean > 0
    assert perf.get_throughput().mean > 0
    metrics = result.extended_perf_metrics
    assert metrics is not None
    if mode == "empty":
        assert metrics.get_num_draft_tokens() == 0
        assert metrics.get_num_accepted_tokens() == 0
    elif mode == "chain":
        assert metrics.get_num_accepted_tokens() > 0
        assert metrics.get_num_accepted_tokens() == metrics.get_num_draft_tokens()
    else:
        # Every step proposes one wrong branch next to the right one.
        assert 0 < metrics.get_num_accepted_tokens() < metrics.get_num_draft_tokens()
    assert drafter.finished == 1


def test_drafter_receives_prompt_and_accepted_tokens(drafter_pipe, reference):
    pipe, drafter = drafter_pipe
    inputs, reference_tokens = reference
    drafter.reset("chain", reference_tokens)

    pipe.generate(inputs, greedy_config())

    prompt = [int(token) for token in inputs.input_ids.data[0]]
    assert drafter.started == [prompt]
    # The prompt is reported first; the model has no last_hidden_state output, so the features are empty.
    assert drafter.accepted[0] == (prompt, list(range(len(prompt))), 0)
    # Then every step reports the tokens it added to the KV cache: the generated tokens in order.
    tokens = [token for step in drafter.accepted[1:] for token in step[0]]
    positions = [position for step in drafter.accepted[1:] for position in step[1]]
    assert tokens == reference_tokens[: len(tokens)]
    assert positions == list(range(len(prompt), len(prompt) + len(tokens)))
    assert len(drafter.accepted) == drafter.proposals + 1


@pytest.mark.parametrize("max_new_tokens", [1, 2, 3, MAX_DRAFT_TOKENS + 2])
def test_drafter_receives_only_output_tokens(drafter_pipe, reference, max_new_tokens):
    pipe, drafter = drafter_pipe
    inputs, reference_tokens = reference
    drafter.reset("chain", reference_tokens)

    result = pipe.generate(inputs, greedy_config(max_new_tokens=max_new_tokens))

    output = list(result.tokens[0])
    assert output == reference_tokens[:max_new_tokens]
    # Near the limit the steps verify fewer nodes: the drafter gets every output token but the last (the next root).
    accepted = [token for step in drafter.accepted[1:] for token in step[0]]
    assert accepted == output[:-1]


def test_stop_token_ends_generation(drafter_pipe, reference):
    pipe, drafter = drafter_pipe
    inputs, reference_tokens = reference
    drafter.reset("chain", reference_tokens)
    # Stop at the first output token, from the third on, that doesn't appear earlier (EOS also stops).
    stop_token = next(token for i, token in enumerate(reference_tokens) if i >= 2 and token not in reference_tokens[:i])
    eos_token = pipe.get_generation_config().eos_token_id
    end = next(i for i, token in enumerate(reference_tokens) if token in (stop_token, eos_token))

    result = pipe.generate(
        inputs, ov_genai.GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, do_sample=False, stop_token_ids={stop_token})
    )

    assert list(result.tokens[0]) == reference_tokens[: end + 1]
    assert result.finish_reasons == [ov_genai.GenerationFinishReason.STOP]


@pytest.mark.parametrize(
    "status,reason",
    [
        pytest.param(ov_genai.StreamingStatus.STOP, ov_genai.GenerationFinishReason.STOP, id="stop"),
        pytest.param(
            ov_genai.StreamingStatus.TOOL_CALL_STOP, ov_genai.GenerationFinishReason.TOOL_CALL, id="tool_call_stop"
        ),
    ],
)
def test_streamer_stop_sets_finish_reason(drafter_pipe, reference, status, reason):
    pipe, drafter = drafter_pipe
    inputs, reference_tokens = reference
    drafter.reset("chain", reference_tokens)

    class StopAtSecondWrite(ov_genai.StreamerBase):
        def __init__(self):
            super().__init__()
            self.writes = 0

        def write(self, token):
            self.writes += 1
            return status if self.writes == 2 else ov_genai.StreamingStatus.RUNNING

        def end(self):
            pass

    result = pipe.generate(inputs, greedy_config(), streamer=StopAtSecondWrite())

    assert len(result.tokens[0]) < MAX_NEW_TOKENS
    assert result.finish_reasons == [reason]


def test_masked_prompt_rejected(drafter_pipe, reference):
    pipe, drafter = drafter_pipe
    inputs, reference_tokens = reference
    drafter.reset("chain", reference_tokens)
    mask = inputs.attention_mask.data.copy()
    mask[0, 0] = 0

    with pytest.raises(RuntimeError, match="attention masks"):
        pipe.generate(ov_genai.TokenizedInputs(inputs.input_ids, ov.Tensor(mask)), greedy_config())
    assert drafter.started == []


def test_zero_assistant_tokens_skips_drafter(drafter_pipe, reference):
    pipe, drafter = drafter_pipe
    inputs, reference_tokens = reference
    drafter.reset("chain", reference_tokens)

    result = pipe.generate(inputs, greedy_config(num_assistant_tokens=0))

    assert list(result.tokens[0]) == reference_tokens
    assert drafter.started == [] and drafter.proposals == 0 and drafter.finished == 0


def test_too_many_assistant_tokens_rejected(drafter_pipe, reference):
    pipe, drafter = drafter_pipe
    inputs, reference_tokens = reference
    drafter.reset("chain", reference_tokens)

    with pytest.raises(RuntimeError, match="max_draft_tokens"):
        pipe.generate(inputs, greedy_config(num_assistant_tokens=MAX_DRAFT_TOKENS + 1))


def test_sampling_rejected(drafter_pipe, reference):
    pipe, drafter = drafter_pipe
    inputs, reference_tokens = reference
    drafter.reset("chain", reference_tokens)

    with pytest.raises(RuntimeError, match="greedy decoding only"):
        pipe.generate(inputs, ov_genai.GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, do_sample=True))


@pytest.mark.parametrize(
    "option",
    [
        {"num_beams": 2},
        {"repetition_penalty": 1.1},
        {"presence_penalty": 0.5},
        {"frequency_penalty": 0.5},
        {"min_new_tokens": 5},
        {"stop_strings": {"Turing"}},
        {"logprobs": 2},
        {"echo": True},
    ],
)
def test_unsupported_options_rejected(drafter_pipe, reference, option):
    pipe, drafter = drafter_pipe
    inputs, reference_tokens = reference
    drafter.reset("chain", reference_tokens)

    with pytest.raises(RuntimeError, match="doesn't support"):
        pipe.generate(inputs, greedy_config(**option))
    assert drafter.started == []


def test_max_length_limits_generation(drafter_pipe, reference):
    pipe, drafter = drafter_pipe
    inputs, reference_tokens = reference
    drafter.reset("chain", reference_tokens)
    max_length = len(inputs.input_ids.data[0]) + MAX_NEW_TOKENS // 2

    result = pipe.generate(inputs, ov_genai.GenerationConfig(max_length=max_length, ignore_eos=True, do_sample=False))

    assert list(result.tokens[0]) == reference_tokens[: MAX_NEW_TOKENS // 2]


def test_kv_cache_limit_ends_generation(model_path):
    kv_config = {"MAX_PROMPT_LEN": 64, "MIN_RESPONSE_LEN": 64}
    drafter = ScriptedDrafter()
    pipe = ov_genai.LLMPipeline(model_path, "NPU", **get_npu_llm_properties_for_test(), **kv_config, drafter=drafter)
    inputs = pipe.get_tokenizer().encode(PROMPT)
    # NPUW stores past tokens outside the generate window (MAX_DRAFT_TOKENS + 1 rows, aligned to 8).
    kv_cache_size = kv_config["MAX_PROMPT_LEN"] + kv_config["MIN_RESPONSE_LEN"] - 8
    expected_len = kv_cache_size - inputs.input_ids.get_shape()[1] + 1

    drafter.reset("empty", [])
    one_token_steps = pipe.generate(inputs, greedy_config(max_new_tokens=1000))
    drafter.reset("chain", one_token_steps.tokens[0])
    tree_steps = pipe.generate(inputs, greedy_config(max_new_tokens=1000))

    for result in (one_token_steps, tree_steps):
        assert len(result.tokens[0]) == expected_len
        assert result.finish_reasons == [ov_genai.GenerationFinishReason.NONE]
    assert list(tree_steps.tokens[0]) == list(one_token_steps.tokens[0])


def test_invalid_draft_tree_rejected(drafter_pipe, reference):
    pipe, drafter = drafter_pipe
    inputs, reference_tokens = reference
    drafter.reset("invalid", reference_tokens)

    with pytest.raises(RuntimeError, match="earlier node"):
        pipe.generate(inputs, greedy_config())
    assert drafter.finished == 1


@pytest.mark.parametrize(
    "mode,message",
    [
        ("raise", "scripted drafter failure"),
        ("raise_in_start", "scripted start failure"),
        ("raise_in_finish", "scripted finish failure"),
    ],
)
def test_drafter_exception_propagates(drafter_pipe, reference, mode, message):
    pipe, drafter = drafter_pipe
    inputs, reference_tokens = reference
    drafter.reset(mode, reference_tokens)

    with pytest.raises(ValueError, match=message):
        pipe.generate(inputs, greedy_config())
    assert drafter.finished == 1


def test_drafter_requires_npu(model_path):
    with pytest.raises(RuntimeError, match="NPU"):
        ov_genai.LLMPipeline(model_path, "CPU", **get_default_llm_properties(), drafter=ScriptedDrafter())


def test_drafter_and_draft_model_are_exclusive(model_path):
    draft_model = ov_genai.draft_model(model_path, "NPU", **get_npu_llm_properties_for_test())
    with pytest.raises(RuntimeError, match="draft_model and drafter"):
        ov_genai.LLMPipeline(
            model_path,
            "NPU",
            **get_npu_llm_properties_for_test(),
            drafter=ScriptedDrafter(),
            draft_model=draft_model,
        )
