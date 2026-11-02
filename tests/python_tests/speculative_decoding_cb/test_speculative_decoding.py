# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end scenarios shared by the speculative decoding methods in continuous batching.

Each scenario runs for the target and draft pairs it applies to. Greedy speculative decoding must reproduce the
target alone, whatever the draft proposes.
"""

import functools
import gc
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openvino as ov
import openvino.properties.hint as hints
import pytest
from huggingface_hub import snapshot_download
from openvino_genai import (
    ContinuousBatchingPipeline,
    GenerationConfig,
    GenerationFinishReason,
    GenerationStatus,
    LLMPipeline,
    SchedulerConfig,
    Tokenizer,
    VLMPipeline,
    draft_model,
)

from test_continuous_batching import _build_input_ids_with_exact_token_count
from test_vlm_pipeline import (
    PROMPTS,
    _get_ov_model,
    _setup_generation_config,
    cat_image,  # noqa: F401
    cat_tensor,  # noqa: F401
)
from utils.atomic_download import AtomicDownloadManager
from utils.constants import get_ov_cache_converted_models_dir
from utils.hugging_face import download_and_convert_model, export_with_optimum_cli, sanitize_model_id
from utils.network import retry_request
from utils.ov_genai_pipelines import dict_to_scheduler_config

QWEN3 = "optimum-intel-internal-testing/tiny-random-qwen3"
QWEN35 = "optimum-intel-internal-testing/tiny-random-qwen3.5"
# MiniCPM-V 4.7 needs transformers>=5.18, so its pairs skip in older environments.
MINICPMV47 = "optimum-intel-internal-testing/tiny-random-minicpmv47"
VLM_TARGETS = {QWEN35, MINICPMV47}
# The Markov head drafts with its dense step instead of its Top-K candidate scorer.
MARKOV_STEP = {"markov_head_top_k": 0}

# (target, draft, draft properties)
QWEN3_DFLASH = pytest.param((QWEN3, "optimum-intel-internal-testing/tiny-random-qwen3-dflash", {}), id="qwen3-dflash")
QWEN3_DEEPSPEC = pytest.param(
    (QWEN3, "optimum-intel-internal-testing/tiny-random-qwen3-deepspec-dflash", {}), id="qwen3-deepspec"
)
QWEN3_DSPARK = pytest.param((QWEN3, "optimum-intel-internal-testing/tiny-random-qwen3-dspark", {}), id="qwen3-dspark")
QWEN3_DSPARK_MARKOV_STEP = pytest.param(
    (QWEN3, "optimum-intel-internal-testing/tiny-random-qwen3-dspark", MARKOV_STEP), id="qwen3-dspark-markov_step"
)
QWEN35_DFLASH = pytest.param(
    (QWEN35, "optimum-intel-internal-testing/tiny-random-qwen3.5-dflash", {}), id="qwen3.5-dflash"
)
MINICPMV47_DSPARK = pytest.param(
    (MINICPMV47, "optimum-intel-internal-testing/tiny-random-minicpmv47-dspark", {}), id="minicpmv47-dspark"
)
MINICPMV47_DSPARK_MARKOV_STEP = pytest.param(
    (MINICPMV47, "optimum-intel-internal-testing/tiny-random-minicpmv47-dspark", MARKOV_STEP),
    id="minicpmv47-dspark-markov_step",
)

TEXT_TARGET_PAIRS = [QWEN3_DFLASH, QWEN3_DEEPSPEC, QWEN3_DSPARK, QWEN3_DSPARK_MARKOV_STEP]
VLM_TARGET_PAIRS = [QWEN35_DFLASH, MINICPMV47_DSPARK, MINICPMV47_DSPARK_MARKOV_STEP]
PAIRS = TEXT_TARGET_PAIRS + VLM_TARGET_PAIRS

# A compressed KV cache lets speculative and autoregressive decoding diverge: a validation window attends to
# its own rows at full precision, while autoregressive decoding reads them back from the cache.
PARITY_PROPERTIES = {hints.inference_precision: ov.Type.f32, hints.kv_cache_precision: ov.Type.f32}


@dataclass(frozen=True, eq=False)
class ModelPair:
    target_path: Path
    draft_path: Path
    draft_properties: dict
    is_vlm: bool


@pytest.fixture(scope="module")
def model_pair(request: pytest.FixtureRequest) -> ModelPair:
    target, draft, draft_properties = request.param
    is_vlm = target in VLM_TARGETS
    target_path = Path(_get_ov_model(target)) if is_vlm else download_and_convert_model(target).models_path
    draft_path = get_ov_cache_converted_models_dir() / sanitize_model_id(draft)
    AtomicDownloadManager(draft_path).execute(
        lambda temp_path: export_with_optimum_cli(
            retry_request(lambda: snapshot_download(draft)),
            "text-generation-with-past",
            temp_path,
            trust_remote_code=True,
        )
    )
    return ModelPair(target_path, draft_path, draft_properties, is_vlm)


def _scheduler_config(**options) -> SchedulerConfig:
    return dict_to_scheduler_config(options)


def _draft(pair: ModelPair, draft_scheduler_config: SchedulerConfig | None = None) -> dict:
    properties = PARITY_PROPERTIES | pair.draft_properties
    if draft_scheduler_config is not None:
        properties["scheduler_config"] = draft_scheduler_config
    return {"draft_model": draft_model(pair.draft_path, "CPU", **properties)}


def _cb_pipeline(
    pair: ModelPair,
    scheduler_config: SchedulerConfig | None = None,
    draft_scheduler_config: SchedulerConfig | None = None,
) -> ContinuousBatchingPipeline:
    if scheduler_config is None:
        scheduler_config = _scheduler_config()
    properties = PARITY_PROPERTIES | _draft(pair, draft_scheduler_config)
    return ContinuousBatchingPipeline(pair.target_path, scheduler_config, "CPU", properties)


def _target_pipeline(pair: ModelPair, scheduler_config: SchedulerConfig | None = None) -> ContinuousBatchingPipeline:
    if scheduler_config is None:
        scheduler_config = _scheduler_config()
    return ContinuousBatchingPipeline(pair.target_path, scheduler_config, "CPU", PARITY_PROPERTIES)


def _llm_pipeline(pair: ModelPair, scheduler_config: SchedulerConfig, speculative: bool = True) -> LLMPipeline:
    properties = PARITY_PROPERTIES | (_draft(pair) if speculative else {"ATTENTION_BACKEND": "PA"})
    return LLMPipeline(pair.target_path, "CPU", properties, scheduler_config=scheduler_config)


def _vlm_pipeline(
    pair: ModelPair,
    scheduler_config: SchedulerConfig | None = None,
    draft_scheduler_config: SchedulerConfig | None = None,
    speculative: bool = True,
) -> VLMPipeline:
    properties = dict(PARITY_PROPERTIES)
    if speculative:
        properties |= _draft(pair, draft_scheduler_config)
    if scheduler_config is not None:
        properties["scheduler_config"] = scheduler_config
    return VLMPipeline(pair.target_path, "CPU", **properties)


PROMPTS_4 = [
    "Why is the Sun yellow?",
    "What's OpenVINO?",
    "Tell me something about Canada.",
    "Why is the grass green?",
]


def _greedy_config(num_assistant_tokens: int = 3, max_new_tokens: int = 20) -> GenerationConfig:
    return GenerationConfig(
        do_sample=False, max_new_tokens=max_new_tokens, ignore_eos=True, num_assistant_tokens=num_assistant_tokens
    )


def _target_ids(pair: ModelPair, prompts: list, config: GenerationConfig) -> list:
    pipe = _target_pipeline(pair)
    return [result.m_generation_ids for result in pipe.generate(prompts, [config] * len(prompts))]


# Block prompts cross the 32-token blocks of the CPU KV cache.
BLOCK_PROMPT_LENGTHS = (5, 37, 70)
BLOCK_NUM_ASSISTANT_TOKENS = 4
BLOCK_MAX_NEW_TOKENS = 40
BLOCK_PROMPT_TEXT = (
    "Speculative decoding drafts several tokens and the target model validates them in a single step. "
    "A scheduler splits long prompts into chunks, preempts requests when the cache is full, and reuses cached "
    "prefixes of repeated prompts."
)


def _text_with_token_count(tokenizer: Tokenizer, num_tokens: int) -> str:
    words = BLOCK_PROMPT_TEXT.split()
    text = ""
    for index in range(10 * num_tokens):
        candidate = f"{text} {words[index % len(words)]}".strip()
        if len(tokenizer.encode(candidate, add_special_tokens=False).input_ids.data[0]) > num_tokens:
            break
        text = candidate
    return text


def _block_prompts(pair: ModelPair) -> list:
    if not pair.is_vlm:
        rng = np.random.default_rng(7)
        return [ov.Tensor(rng.integers(10, 1000, size=(1, length), dtype=np.int64)) for length in BLOCK_PROMPT_LENGTHS]
    # A VLM pipeline takes text, so its prompts have the closest token counts the tokenizer allows.
    tokenizer = Tokenizer(pair.target_path)
    return [_text_with_token_count(tokenizer, length) for length in BLOCK_PROMPT_LENGTHS]


def _block_config(num_assistant_tokens: int = BLOCK_NUM_ASSISTANT_TOKENS) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=BLOCK_MAX_NEW_TOKENS,
        ignore_eos=True,
        num_assistant_tokens=num_assistant_tokens,
        apply_chat_template=False,
    )


def _block_scheduler_config(max_num_batched_tokens: int = 256, **options) -> SchedulerConfig:
    return _scheduler_config(
        **{
            "max_num_batched_tokens": max_num_batched_tokens,
            "dynamic_split_fuse": True,
            "enable_prefix_caching": False,
            **options,
        },
    )


def _block_pipeline(
    pair: ModelPair,
    scheduler_config: SchedulerConfig | None = None,
    draft_scheduler_config: SchedulerConfig | None = None,
) -> ContinuousBatchingPipeline:
    if scheduler_config is None:
        scheduler_config = _block_scheduler_config()
    return _cb_pipeline(pair, scheduler_config, draft_scheduler_config)


def _generate(pipe: ContinuousBatchingPipeline, prompts: list, config: GenerationConfig) -> tuple[list, list]:
    results = pipe.generate(prompts, [config] * len(prompts))
    return [result.m_generation_ids[0] for result in results], results


def _assert_all_tokens_generated(outputs: list, results: list) -> None:
    if isinstance(outputs[0], str):
        # Text prompts return text. The perf metrics of one generate() call count the tokens of all its requests,
        # and no request generates more than max_new_tokens.
        num_generated_tokens = [result.perf_metrics.get_num_generated_tokens() for result in results]
        assert num_generated_tokens == [BLOCK_MAX_NEW_TOKENS * len(results)] * len(results)
    else:
        assert all(len(tokens) == BLOCK_MAX_NEW_TOKENS for tokens in outputs)


@functools.cache
def _block_target_outputs(pair: ModelPair) -> list:
    outputs, results = _generate(
        _target_pipeline(pair, _block_scheduler_config()), _block_prompts(pair), _block_config(0)
    )
    _assert_all_tokens_generated(outputs, results)
    return outputs


# Sequence::update_generated_log_prob corrects a token's log-probability but not the sequence score, so the score keeps
# the draft's log-probabilities of the tokens the target checked. Remove this skip once
# openvinotoolkit/openvino.genai#4769 is merged.
SCORE_SKIP = pytest.mark.skip(reason="The score keeps the draft's log-probabilities, CVS-195613")


def _run_add_requests(pipe: ContinuousBatchingPipeline, prompts: list, config: GenerationConfig, arrival_steps: dict):
    handles = {}
    step = 0
    while len(handles) < len(prompts) or pipe.has_non_finished_requests():
        for request_id, arrival in arrival_steps.items():
            if arrival == step:
                handles[request_id] = pipe.add_request(request_id, prompts[request_id], config)
        if pipe.has_non_finished_requests():
            pipe.step()
        step += 1
    return [handles[request_id].read_all()[0].generated_ids for request_id in range(len(prompts))]


@pytest.mark.parametrize("ignore_eos", [False, True])
@pytest.mark.parametrize("model_pair", PAIRS, indirect=True)
def test_generate_matches_target(model_pair, ignore_eos):
    config = GenerationConfig(do_sample=False, max_new_tokens=20, ignore_eos=ignore_eos, num_assistant_tokens=7)
    prompt = "Why is the Sun yellow?"
    main_only_pipe = _target_pipeline(model_pair)
    reference = [result.m_generation_ids for result in main_only_pipe.generate([prompt], [config])]
    del main_only_pipe
    gc.collect()

    result = _cb_pipeline(model_pair).generate([prompt], [config])[0]

    assert [result.m_generation_ids] == reference
    metrics = result.extended_perf_metrics
    assert metrics.get_num_draft_tokens() > 0
    assert metrics.get_num_accepted_tokens() <= metrics.get_num_draft_tokens()
    assert (
        0
        < metrics.draft_model_metrics.get_num_generated_tokens()
        <= config.max_new_tokens * config.num_assistant_tokens
    )


# The DeepSpec draft has candidates accepted on this prompt, and the other drafts have none.
@pytest.mark.parametrize(
    "model_pair",
    [
        QWEN3_DFLASH,
        pytest.param(*QWEN3_DEEPSPEC.values, id=QWEN3_DEEPSPEC.id, marks=SCORE_SKIP),
        QWEN3_DSPARK,
        QWEN3_DSPARK_MARKOV_STEP,
    ],
    indirect=True,
)
def test_score_matches_target(model_pair):
    config = GenerationConfig(do_sample=False, max_new_tokens=20, num_assistant_tokens=3, logprobs=1)
    prompt = "What's OpenVINO?"
    reference = _target_pipeline(model_pair).generate([prompt], [config])[0]

    result = _cb_pipeline(model_pair).generate([prompt], [config])[0]

    assert result.m_generation_ids == reference.m_generation_ids
    assert result.m_scores == pytest.approx(reference.m_scores, abs=1e-4)


@pytest.mark.parametrize("model_pair", PAIRS, indirect=True)
def test_requests_drafted_together_match_target(model_pair):
    target_outputs = _block_target_outputs(model_pair)
    pipe = _block_pipeline(model_pair)
    outputs, results = _generate(pipe, _block_prompts(model_pair), _block_config())
    assert outputs == target_outputs
    assert results[0].extended_perf_metrics.get_num_draft_tokens() > 0
    # one draft forward and one candidate selection drafts the blocks of several requests
    candidates_per_draft_step = results[0].extended_perf_metrics.draft_model_metrics.raw_metrics.m_batch_sizes
    assert max(candidates_per_draft_step) > BLOCK_NUM_ASSISTANT_TOKENS

    single_outputs = [_generate(pipe, [prompt], _block_config())[0][0] for prompt in _block_prompts(model_pair)]
    assert single_outputs == target_outputs


@pytest.mark.parametrize("model_pair", PAIRS, indirect=True)
def test_requests_without_dynamic_split_fuse_match_target(model_pair):
    # Without dynamic split fuse a step runs either prompt or generation windows, so the first block follows the
    # prompt rows one draft step later.
    pipe = _block_pipeline(model_pair, _block_scheduler_config(dynamic_split_fuse=False))
    outputs, results = _generate(pipe, _block_prompts(model_pair), _block_config())
    assert outputs == _block_target_outputs(model_pair)
    assert results[0].extended_perf_metrics.get_num_draft_tokens() > 0


@pytest.mark.parametrize("model_pair", PAIRS, indirect=True)
def test_staggered_requests_match_target(model_pair):
    prompts = _block_prompts(model_pair)
    # a new request arrives while the earlier ones are prefilling, validating or drafting
    arrival_steps = {0: 0, 1: 3, 2: 7}
    reference = _run_add_requests(
        _target_pipeline(model_pair, _block_scheduler_config()), prompts, _block_config(0), arrival_steps
    )
    assert all(len(tokens) == BLOCK_MAX_NEW_TOKENS for tokens in reference)

    outputs = _run_add_requests(_block_pipeline(model_pair), prompts, _block_config(), arrival_steps)

    assert outputs == reference


@pytest.mark.parametrize("dynamic_split_fuse", [True, False])
@pytest.mark.parametrize("model_pair", PAIRS, indirect=True)
def test_staggered_requests_with_mixed_assistant_tokens(model_pair, dynamic_split_fuse):
    # num_assistant_tokens=0 runs the request on the target alone
    configs = [_greedy_config(num_assistant_tokens) for num_assistant_tokens in (3, 0, 5, 1)]
    scheduler_config = _scheduler_config(dynamic_split_fuse=dynamic_split_fuse, max_num_batched_tokens=sys.maxsize)

    def run_staggered(pipe):
        handles = [pipe.add_request(request_id, PROMPTS_4[request_id], configs[request_id]) for request_id in (0, 1)]
        for _ in range(2):
            pipe.step()
        handles += [pipe.add_request(request_id, PROMPTS_4[request_id], configs[request_id]) for request_id in (2, 3)]
        while pipe.has_non_finished_requests():
            pipe.step()
        outputs = [handle.read_all() for handle in handles]
        assert all(len(output) == 1 and output[0].finish_reason == GenerationFinishReason.LENGTH for output in outputs)
        return [list(output[0].generated_ids) for output in outputs]

    reference = run_staggered(_target_pipeline(model_pair, scheduler_config))

    assert run_staggered(_cb_pipeline(model_pair, scheduler_config)) == reference


@pytest.mark.parametrize("model_pair", PAIRS, indirect=True)
def test_batched_prompts_with_chunked_prefill_match_target(model_pair):
    config = _greedy_config()
    reference = _target_ids(model_pair, PROMPTS_4, config)
    # prefill chunks shorter than the prompts and than a draft block
    scheduler_config = _scheduler_config(dynamic_split_fuse=True, max_num_batched_tokens=4)

    results = _cb_pipeline(model_pair, scheduler_config).generate(PROMPTS_4, [config] * len(PROMPTS_4))

    assert [result.m_generation_ids for result in results] == reference


@pytest.mark.parametrize("model_pair", [QWEN3_DFLASH], indirect=True)
def test_rejects_a_token_budget_below_the_validation_window(model_pair):
    # The target validates the seed and its 3 candidates in one forward, which a budget of 3 tokens never fits.
    config = _greedy_config(num_assistant_tokens=3)
    scheduler_config = _scheduler_config(dynamic_split_fuse=True, max_num_batched_tokens=3)

    with pytest.raises(RuntimeError, match="too small to ever schedule the speculative validation window"):
        _cb_pipeline(model_pair, scheduler_config).generate(PROMPTS_4[:1], [config])


@pytest.mark.parametrize("model_pair", [QWEN3_DEEPSPEC], indirect=True)
def test_counts_candidates_validated_in_a_later_step(model_pair):
    # A budget of 9 fits two validation windows of the 4 requests, and the target validates the others in a later
    # step. The drafts and the outputs do not depend on the budget, so neither do the accepted tokens. The target
    # accepts some candidates of the tiny DeepSpec draft, and none of the DFlash one.
    config = _greedy_config(num_assistant_tokens=3, max_new_tokens=30)

    def generate(max_num_batched_tokens):
        scheduler_config = _scheduler_config(dynamic_split_fuse=True, max_num_batched_tokens=max_num_batched_tokens)
        results = _cb_pipeline(model_pair, scheduler_config).generate(PROMPTS_4, [config] * len(PROMPTS_4))
        metrics = results[0].extended_perf_metrics
        return (
            [result.m_generation_ids for result in results],
            metrics.get_num_draft_tokens(),
            metrics.get_num_accepted_tokens(),
        )

    assert generate(9) == generate(256)


@pytest.mark.parametrize("model_pair", [QWEN3_DEEPSPEC], indirect=True)
def test_counts_the_window_that_ends_a_request(model_pair):
    # Without dynamic split fuse the first draft window holds only prompt rows, so with 4 new tokens each request drafts
    # one candidate after its second token: only one fits before the last token. A candidate the target accepts ends
    # the request, and its window counts all the same.
    config = _greedy_config(num_assistant_tokens=3, max_new_tokens=4)
    scheduler_config = _scheduler_config(dynamic_split_fuse=False)

    results = _cb_pipeline(model_pair, scheduler_config).generate(PROMPTS_4, [config] * len(PROMPTS_4))

    metrics = results[0].extended_perf_metrics
    assert metrics.get_num_draft_tokens() == len(PROMPTS_4)
    assert metrics.get_num_accepted_tokens() > 0


@pytest.mark.parametrize("model_pair", PAIRS, indirect=True)
def test_chunked_prefill_matches_target(model_pair):
    # 16 tokens split the prompts over several steps and fit the 1 + BLOCK_NUM_ASSISTANT_TOKENS rows of the
    # validation windows of all three requests.
    pipe = _block_pipeline(model_pair, _block_scheduler_config(max_num_batched_tokens=16))
    outputs, results = _generate(pipe, _block_prompts(model_pair), _block_config())
    assert outputs == _block_target_outputs(model_pair)
    assert results[0].extended_perf_metrics.get_num_draft_tokens() > 0


@pytest.mark.parametrize("model_pair", PAIRS, indirect=True)
def test_zero_assistant_tokens_generate_matches_target(model_pair):
    scheduler_config = _scheduler_config(dynamic_split_fuse=False, max_num_batched_tokens=sys.maxsize)
    sd_cb_pipe = _cb_pipeline(model_pair, scheduler_config)
    main_only_cb_pipe = _target_pipeline(model_pair, scheduler_config)

    prompt = "Explain why sunsets often appear orange."
    sd_config = GenerationConfig(do_sample=False, max_new_tokens=20, num_assistant_tokens=0)
    main_only_config = GenerationConfig(do_sample=False, max_new_tokens=20)

    sd_result = sd_cb_pipe.generate([prompt], [sd_config])
    main_only_result = main_only_cb_pipe.generate([prompt], [main_only_config])

    assert len(sd_result) == 1
    assert len(main_only_result) == 1
    assert len(sd_result[0].m_generation_ids) == 1
    assert len(main_only_result[0].m_generation_ids) == 1
    assert len(sd_result[0].m_generation_ids[0]) > 0
    assert sd_result[0].m_generation_ids[0] == main_only_result[0].m_generation_ids[0]


@pytest.mark.parametrize("model_pair", TEXT_TARGET_PAIRS, indirect=True)
def test_zero_assistant_tokens_llm_pipeline_matches_target(model_pair):
    scheduler_config = _scheduler_config(dynamic_split_fuse=False, max_num_batched_tokens=sys.maxsize)
    sd_pipe = _llm_pipeline(model_pair, scheduler_config)
    main_only_pipe = _llm_pipeline(model_pair, scheduler_config, speculative=False)

    prompt = "Explain why sunsets often appear orange."
    sd_config = GenerationConfig(do_sample=False, max_new_tokens=20, num_assistant_tokens=0)
    main_only_config = GenerationConfig(do_sample=False, max_new_tokens=20)

    sd_result = sd_pipe.generate([prompt], sd_config)
    main_only_result = main_only_pipe.generate([prompt], main_only_config)

    assert len(sd_result.texts) == 1
    assert len(main_only_result.texts) == 1
    assert len(sd_result.texts[0]) > 0
    assert sd_result.texts[0] == main_only_result.texts[0]


@pytest.mark.parametrize("model_pair", PAIRS, indirect=True)
def test_zero_assistant_tokens_add_request_mixed_batch_matches_target(model_pair):
    scheduler_config = _scheduler_config(dynamic_split_fuse=False, max_num_batched_tokens=sys.maxsize)
    sd_cb_pipe = _cb_pipeline(model_pair, scheduler_config)
    main_only_cb_pipe = _target_pipeline(model_pair, scheduler_config)

    main_only_prompt = "Write one sentence about OpenVINO."
    speculative_prompt = "Write one sentence about the Moon."
    zero_assistant_config = GenerationConfig(do_sample=False, max_new_tokens=20, num_assistant_tokens=0)
    speculative_config = GenerationConfig(do_sample=False, max_new_tokens=20, num_assistant_tokens=4)

    handle_main_only = sd_cb_pipe.add_request(0, main_only_prompt, generation_config=zero_assistant_config)
    handle_speculative = sd_cb_pipe.add_request(1, speculative_prompt, generation_config=speculative_config)
    while sd_cb_pipe.has_non_finished_requests():
        sd_cb_pipe.step()

    sd_main_only_outputs = handle_main_only.read_all()
    sd_speculative_outputs = handle_speculative.read_all()

    baseline_handle = main_only_cb_pipe.add_request(
        2, main_only_prompt, generation_config=GenerationConfig(do_sample=False, max_new_tokens=20)
    )
    while main_only_cb_pipe.has_non_finished_requests():
        main_only_cb_pipe.step()
    baseline_outputs = baseline_handle.read_all()

    assert len(sd_main_only_outputs) == 1
    assert len(sd_speculative_outputs) == 1
    assert len(baseline_outputs) == 1
    assert len(sd_main_only_outputs[0].generated_ids) > 0
    assert len(sd_speculative_outputs[0].generated_ids) > 0
    assert sd_main_only_outputs[0].generated_ids == baseline_outputs[0].generated_ids


@pytest.mark.parametrize("model_pair", PAIRS, indirect=True)
def test_requests_without_drafting_match_target(model_pair):
    # num_assistant_tokens=0 runs a request on the target alone in generate(), as it does in add_request(), and an
    # echo-only request (max_new_tokens=0) returns the prompt without generating anything to draft
    configs = [
        _greedy_config(num_assistant_tokens=0),
        GenerationConfig(max_new_tokens=0, echo=True, num_assistant_tokens=3),
    ]
    reference = [result.m_generation_ids for result in _target_pipeline(model_pair).generate(PROMPTS_4[:2], configs)]

    results = _cb_pipeline(model_pair).generate(PROMPTS_4[:2], configs)

    assert [result.m_generation_ids for result in results] == reference
    assert results[0].extended_perf_metrics.get_num_draft_tokens() == 0


@pytest.mark.parametrize("target_prompt_tokens", [127, 128, 129])
@pytest.mark.parametrize("api", ["generate", "add_request"])
@pytest.mark.parametrize("model_pair", TEXT_TARGET_PAIRS, indirect=True)
def test_prefix_caching_matches_target(model_pair, api, target_prompt_tokens):
    scheduler_config = _scheduler_config(
        enable_prefix_caching=True, dynamic_split_fuse=False, max_num_batched_tokens=sys.maxsize
    )
    reference_scheduler_config = _scheduler_config(
        enable_prefix_caching=False, dynamic_split_fuse=False, max_num_batched_tokens=sys.maxsize
    )
    generation_config = GenerationConfig(max_new_tokens=20, num_assistant_tokens=4, apply_chat_template=False)
    reference_config = GenerationConfig(max_new_tokens=20, apply_chat_template=False)

    if api == "generate":
        ov_pipe = _llm_pipeline(model_pair, scheduler_config)
        input_ids = _build_input_ids_with_exact_token_count(ov_pipe.get_tokenizer(), target_prompt_tokens)
        reference_pipe = _llm_pipeline(model_pair, reference_scheduler_config, speculative=False)
        reference = reference_pipe.generate(input_ids, reference_config).tokens

        def run(request_id):
            result = ov_pipe.generate(input_ids, generation_config)
            assert result.extended_perf_metrics.get_num_draft_tokens() > 0
            return result.tokens, result.perf_metrics.get_num_prefix_cache_hit_tokens()

    else:
        cb_pipe = _cb_pipeline(model_pair, scheduler_config)
        input_ids = _build_input_ids_with_exact_token_count(cb_pipe.get_tokenizer(), target_prompt_tokens)
        reference_pipe = _target_pipeline(model_pair, reference_scheduler_config)
        reference_handle = reference_pipe.add_request(0, input_ids, generation_config=reference_config)
        while reference_pipe.has_non_finished_requests():
            reference_pipe.step()
        reference = [output.generated_ids for output in reference_handle.read_all()]

        def run(request_id):
            handle = cb_pipe.add_request(request_id, input_ids, generation_config=generation_config)
            while cb_pipe.has_non_finished_requests():
                cb_pipe.step()
            outputs = handle.read_all()
            hits = handle.get_perf_metrics().get_num_prefix_cache_hit_tokens()
            return [output.generated_ids for output in outputs], hits

    first_tokens, first_hits = run(0)
    try:
        second_tokens, second_hits = run(1)
    except RuntimeError as exc:
        pytest.fail(
            f"Second {api} with prefix cache reuse must not raise RuntimeError. "
            f"prompt_tokens={target_prompt_tokens}, error={exc}"
        )

    assert len(first_tokens) == 1
    assert len(second_tokens) == 1
    assert len(first_tokens[0]) > 0
    assert len(second_tokens[0]) > 0
    assert first_tokens == reference
    assert second_tokens == reference
    assert first_hits == 0
    assert second_hits > 0


SHARED_PREFIX = " ".join(
    ["Speculative decoding drafts several tokens and the target model validates them in a single step."] * 12
)


@pytest.mark.parametrize("model_pair", TEXT_TARGET_PAIRS, indirect=True)
def test_prefix_caching_with_shared_prefix_matches_target(model_pair):
    config = _greedy_config()
    # a cold prompt, the same prompt again, and a prompt that extends it
    prompts = [SHARED_PREFIX, SHARED_PREFIX, SHARED_PREFIX + " Why is the Sun yellow?"]
    reference = _target_ids(model_pair, prompts, config)
    pipe = _cb_pipeline(model_pair, _scheduler_config(enable_prefix_caching=True))

    results = [pipe.generate([prompt], [config])[0] for prompt in prompts]

    assert [result.m_generation_ids for result in results] == reference
    hits = [result.perf_metrics.get_num_prefix_cache_hit_tokens() for result in results]
    assert hits[0] == 0
    assert hits[1] > 0 and hits[2] > 0
    assert all(result.extended_perf_metrics.get_num_draft_tokens() > 0 for result in results)


@pytest.mark.parametrize("model_pair", PAIRS, indirect=True)
def test_prefix_caching_with_repeated_prompts_matches_target(model_pair):
    prompts = _block_prompts(model_pair)
    pipe = _block_pipeline(model_pair, _block_scheduler_config(enable_prefix_caching=True))
    outputs = [_generate(pipe, [prompt], _block_config())[0][0] for prompt in prompts + prompts]
    target_outputs = _block_target_outputs(model_pair)
    assert outputs == target_outputs + target_outputs


@pytest.mark.parametrize("main_prefix", [False, True])
@pytest.mark.parametrize("model_pair", PAIRS, indirect=True)
def test_rejects_prefix_caching_mismatch(model_pair, main_prefix):
    draft_scheduler_config = dict_to_scheduler_config({"num_kv_blocks": 64, "enable_prefix_caching": not main_prefix})
    with pytest.raises(RuntimeError, match="must use the same enable_prefix_caching"):
        ContinuousBatchingPipeline(
            model_pair.target_path,
            _scheduler_config(enable_prefix_caching=main_prefix),
            "CPU",
            _draft(model_pair, draft_scheduler_config),
        )


# VLM generate() results do not report the generation status, so these scenarios run on text targets.
@pytest.mark.parametrize("dynamic_split_fuse", [True, False])
@pytest.mark.parametrize("model_pair", TEXT_TARGET_PAIRS, indirect=True)
def test_small_draft_cache_leaves_requests_to_the_target(model_pair, dynamic_split_fuse):
    config = _greedy_config(max_new_tokens=64)
    reference = _target_ids(model_pair, PROMPTS_4, config)
    pipe = _cb_pipeline(
        model_pair,
        _scheduler_config(dynamic_split_fuse=dynamic_split_fuse, max_num_batched_tokens=sys.maxsize),
        dict_to_scheduler_config({"num_kv_blocks": 2}),
    )

    results = pipe.generate(PROMPTS_4, [config] * len(PROMPTS_4))

    # the target finishes every request whose draft stopped
    assert [result.m_status for result in results] == [GenerationStatus.FINISHED] * len(PROMPTS_4)
    assert [result.m_generation_ids for result in results] == reference


@pytest.mark.parametrize("num_kv_blocks", [3, 4, 6])
@pytest.mark.parametrize("model_pair", PAIRS, indirect=True)
def test_draft_kv_cache_limit_leaves_requests_to_the_target(model_pair, num_kv_blocks):
    # The draft KV cache holds num_kv_blocks blocks of 32 tokens, fewer than the three requests need: a request whose
    # window does not run stops drafting, and the target finishes it alone.
    pipe = _block_pipeline(model_pair, draft_scheduler_config=_block_scheduler_config(num_kv_blocks=num_kv_blocks))
    outputs, results = _generate(pipe, _block_prompts(model_pair), _block_config())
    assert outputs == _block_target_outputs(model_pair)
    assert results[0].extended_perf_metrics.get_num_draft_tokens() > 0


@pytest.mark.parametrize("model_pair", PAIRS, indirect=True)
def test_requests_with_fixed_target_cache_match_target(model_pair):
    # 16 blocks hold all three requests: the target never preempts
    pipe = _block_pipeline(model_pair, _block_scheduler_config(num_kv_blocks=16))
    outputs, results = _generate(pipe, _block_prompts(model_pair), _block_config())
    assert outputs == _block_target_outputs(model_pair)
    assert results[0].extended_perf_metrics.get_num_draft_tokens() > 0


@pytest.mark.parametrize("model_pair", TEXT_TARGET_PAIRS, indirect=True)
def test_preemption_completes_all_requests(model_pair):
    # Both models preempt requests; the draft waits for a recomputing target and leaves a request whose
    # window was preempted to the target.
    config = _greedy_config(max_new_tokens=64)
    reference = _target_ids(model_pair, PROMPTS_4, config)
    scheduler_config = _scheduler_config(num_kv_blocks=6, dynamic_split_fuse=True, max_num_batched_tokens=256)

    results = _cb_pipeline(model_pair, scheduler_config).generate(PROMPTS_4, [config] * len(PROMPTS_4))

    assert [result.m_status for result in results] == [GenerationStatus.FINISHED] * len(PROMPTS_4)
    assert [result.m_generation_ids for result in results] == reference


@pytest.mark.parametrize("model_pair", PAIRS, indirect=True)
def test_target_and_draft_preemption_complete(model_pair):
    # 6 blocks for the target and, by default, for the draft: the target recomputes preempted requests, and the
    # target finishes alone the requests whose draft windows do not run.
    scheduler_config = _block_scheduler_config(num_kv_blocks=6)
    prompts = _block_prompts(model_pair)
    try:
        outputs, results = _generate(_block_pipeline(model_pair, scheduler_config), prompts, _block_config())
        if isinstance(outputs[0], str):
            # The perf metrics of generate() undercount the tokens of requests preempted during validation, so the
            # tokens of text prompts are counted through add_request.
            arrival_steps = dict.fromkeys(range(len(prompts)), 0)
            generated_ids = _run_add_requests(
                _block_pipeline(model_pair, scheduler_config), prompts, _block_config(), arrival_steps
            )
            assert all(len(ids) == BLOCK_MAX_NEW_TOKENS for ids in generated_ids)
        else:
            _assert_all_tokens_generated(outputs, results)
    except RuntimeError as error:
        if "recompute/eviction of a validating sequence is not supported" not in str(error):
            raise
        pytest.xfail("Linear-attention targets cannot recompute requests preempted during validation, CVS-198275")
    assert results[0].extended_perf_metrics.get_num_draft_tokens() > 0
    assert outputs == _block_target_outputs(model_pair)


@pytest.mark.parametrize("model_pair", [QWEN3_DFLASH], indirect=True)
def test_sampled_requests_are_reproducible(model_pair):
    # The target accepts or replaces every sampled candidate by rejection sampling against the distribution the draft
    # drew it from; the draws of both follow the request seed.
    config = GenerationConfig(
        do_sample=True,
        temperature=0.8,
        top_k=4,
        rng_seed=3,
        max_new_tokens=20,
        ignore_eos=True,
        num_assistant_tokens=3,
    )

    def generate():
        results = _cb_pipeline(model_pair).generate(PROMPTS_4, [config] * len(PROMPTS_4))
        assert results[0].extended_perf_metrics.get_num_draft_tokens() > 0
        return [result.m_generation_ids for result in results]

    assert generate() == generate()


@pytest.mark.parametrize("model_pair", PAIRS, indirect=True)
def test_greedy_requests_with_penalties_match_target(model_pair):
    # Penalties can change the argmax, so the draft draws its candidates with its sampler, over the token history it
    # shares with the target.
    def penalized(config):
        config.repetition_penalty = 1.3
        config.presence_penalty = 0.2
        return config

    target, _ = _generate(
        _target_pipeline(model_pair, _block_scheduler_config()),
        _block_prompts(model_pair),
        penalized(_block_config(0)),
    )
    outputs, results = _generate(_block_pipeline(model_pair), _block_prompts(model_pair), penalized(_block_config()))
    assert outputs == target
    assert results[0].extended_perf_metrics.get_num_draft_tokens() > 0


# The draft gets no properties, so the Markov step pair would repeat the plain DSpark run.
@pytest.mark.parametrize("model_pair", [QWEN3_DFLASH, QWEN3_DEEPSPEC, QWEN3_DSPARK], indirect=True)
def test_llm_pipeline_draft_without_properties_matches_target(model_pair):
    # A draft without properties compiles with the target's, to which LLMPipeline adds the GenAI-only
    # sampler_num_threads.
    pipe = LLMPipeline(
        model_pair.target_path, "CPU", draft_model=draft_model(model_pair.draft_path, "CPU"), **PARITY_PROPERTIES
    )
    for prompt, target in zip(_block_prompts(model_pair), _block_target_outputs(model_pair)):
        result = pipe.generate(prompt, _block_config())
        assert list(result.tokens[0]) == target
        assert result.extended_perf_metrics.get_num_draft_tokens() > 0


@pytest.mark.parametrize("prompt", [PROMPTS[0], PROMPTS[2]])
@pytest.mark.parametrize("model_pair", VLM_TARGET_PAIRS, indirect=True)
def test_image_request_matches_target(model_pair, prompt, cat_tensor):
    ov_pipe = _vlm_pipeline(model_pair, speculative=False)
    generation_config = _setup_generation_config(ov_pipe, max_new_tokens=20, do_sample=False)
    result_without_draft = ov_pipe.generate(prompt, images=[cat_tensor], generation_config=generation_config)

    ov_pipe_with_draft = _vlm_pipeline(model_pair)
    generation_config_with_draft = _setup_generation_config(ov_pipe_with_draft, max_new_tokens=20, do_sample=False)
    generation_config_with_draft.num_assistant_tokens = 3
    result_with_draft = ov_pipe_with_draft.generate(
        prompt, images=[cat_tensor], generation_config=generation_config_with_draft
    )

    assert result_without_draft.texts[0].strip() == result_with_draft.texts[0].strip()


@pytest.mark.parametrize("model_pair", VLM_TARGET_PAIRS, indirect=True)
def test_batched_image_and_text_requests_match_target(model_pair, cat_tensor):
    scheduler_config = _scheduler_config(enable_prefix_caching=False)
    config = GenerationConfig(max_new_tokens=20, do_sample=False, ignore_eos=True, num_assistant_tokens=3)
    prompts = [PROMPTS[2], "Why is the Sun yellow?", PROMPTS[2], "What is OpenVINO?"]
    images = [[cat_tensor], [], [cat_tensor], []]

    def generate(pipe):
        results = pipe.generate(prompts, images=images, generation_config=[config] * len(prompts))
        return [result.texts[0] for result in results], results

    reference, _ = generate(_target_pipeline(model_pair, scheduler_config))
    texts, results = generate(_cb_pipeline(model_pair, scheduler_config))

    assert texts == reference
    assert results[0].extended_perf_metrics.get_num_draft_tokens() > 0


@pytest.mark.parametrize("images", ["same_image", "different_images"])
@pytest.mark.parametrize("model_pair", VLM_TARGET_PAIRS, indirect=True)
def test_prefix_caching_with_images_matches_target(model_pair, images, cat_tensor):
    # Two same-sized images with different content share one text prompt. With prefix caching the draft cache
    # identity is bound to the main model's image-aware prefix hash, so the repeated second-image request must
    # restore the second image's draft blocks rather than the first image's. The repeated request must also reuse
    # a nonzero aligned prefix, which fails if the draft pipeline never publishes its completed prompt blocks.
    image_a = cat_tensor
    image_b = ov.Tensor(255 - cat_tensor.data)
    prompt = PROMPTS[2]
    if images == "same_image":
        sequence = [image_a, image_a]
        reference = _vlm_pipeline(model_pair, _scheduler_config(enable_prefix_caching=False), speculative=False)
        pipe = _vlm_pipeline(model_pair, _scheduler_config(enable_prefix_caching=True))
    else:
        sequence = [image_a, image_b, image_b]
        reference = _vlm_pipeline(model_pair, speculative=False)
        pipe = _vlm_pipeline(
            model_pair,
            _scheduler_config(enable_prefix_caching=True, max_num_seqs=1, num_kv_blocks=128),
            dict_to_scheduler_config({"enable_prefix_caching": True, "num_kv_blocks": 128}),
        )
    reference_config = _setup_generation_config(reference, max_new_tokens=20, do_sample=False)
    expected = [
        reference.generate(prompt, images=[image], generation_config=reference_config).texts[0] for image in sequence
    ]
    config = _setup_generation_config(pipe, max_new_tokens=20, do_sample=False)
    config.num_assistant_tokens = 3

    prefix_hits = []
    for image, expected_text in zip(sequence, expected):
        result = pipe.generate(prompt, images=[image], generation_config=config)
        assert result.texts[0] == expected_text, "Output with prefix caching must match the no-draft reference."
        assert result.extended_perf_metrics.draft_model_metrics.get_num_generated_tokens() > 0, (
            "Draft model must stay active across the cross-image prefix-cache sequence."
        )
        prefix_hits.append(result.perf_metrics.get_num_prefix_cache_hit_tokens())

    assert prefix_hits[0] == 0, f"First request must not reuse any prefix on a cold cache, got {prefix_hits[0]}."
    assert prefix_hits[-1] > 0, (
        f"Repeated request must restore a nonzero aligned prefix, got {prefix_hits[-1]}; "
        "the draft pipeline likely did not publish its completed prompt blocks."
    )
