# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

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
    return ov_genai.GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, ignore_eos=True, do_sample=False, **kwargs)


class ScriptedDrafter(ov_genai.Drafter):
    """Proposes the continuation of a known output and records how the pipeline calls it.

    Modes: "chain" proposes the next tokens as a chain, "branch" adds a wrong first child of the root before the
    chain, "empty" proposes nothing, "invalid" returns a node whose parent is not an earlier node, "raise" raises in
    propose() and "raise_in_finish" raises in finish().
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


def test_invalid_draft_tree_rejected(drafter_pipe, reference):
    pipe, drafter = drafter_pipe
    inputs, reference_tokens = reference
    drafter.reset("invalid", reference_tokens)

    with pytest.raises(RuntimeError, match="earlier node"):
        pipe.generate(inputs, greedy_config())
    assert drafter.finished == 1


@pytest.mark.parametrize(
    "mode,message", [("raise", "scripted drafter failure"), ("raise_in_finish", "scripted finish failure")]
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
