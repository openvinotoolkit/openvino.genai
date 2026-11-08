# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
from openvino_genai import (
    ChatHistory,
    ContinuousBatchingPipeline,
    GenerationConfig,
    GenerationStatus,
    SchedulerConfig,
    VLMPipeline,
    draft_model,
)

from test_speculative_decoding import (
    MINICPMV47_DSPARK,
    MINICPMV47_DSPARK_MARKOV_STEP,
    PARITY_PROPERTIES,
    PROMPTS_4,
    QWEN3_DEEPSPEC,
    QWEN3_DFLASH,
    QWEN3_DSPARK,
    QWEN35_DFLASH,
    SCORE_SKIP,
    VLM_TARGET_PAIRS,
    _greedy_config,
    _target_ids,
    model_pair,  # noqa: F401
)
from test_vlm_pipeline import (
    PROMPTS,
    _setup_generation_config,
    cat_image,  # noqa: F401
    cat_tensor,  # noqa: F401
)


@pytest.mark.parametrize(
    "model_pair",
    [
        QWEN35_DFLASH,
        pytest.param(*MINICPMV47_DSPARK.values, id=MINICPMV47_DSPARK.id, marks=SCORE_SKIP),
        pytest.param(*MINICPMV47_DSPARK_MARKOV_STEP.values, id=MINICPMV47_DSPARK_MARKOV_STEP.id, marks=SCORE_SKIP),
    ],
    indirect=True,
)
def test_vlm_echo_and_logprobs_match_target(model_pair, cat_tensor):
    scheduler_config = SchedulerConfig()
    scheduler_config.enable_prefix_caching = False

    def generate(pipe, num_assistant_tokens=None):
        generation_config = _setup_generation_config(pipe, max_new_tokens=20, do_sample=False)
        generation_config.echo = True
        generation_config.logprobs = 2
        if num_assistant_tokens is not None:
            generation_config.num_assistant_tokens = num_assistant_tokens
        return pipe.generate(PROMPTS[2], images=[cat_tensor], generation_config=generation_config)

    reference = generate(
        VLMPipeline(model_pair.target_path, "CPU", scheduler_config=scheduler_config, **PARITY_PROPERTIES)
    )
    ov_pipe_with_draft = VLMPipeline(
        model_pair.target_path,
        "CPU",
        draft_model=draft_model(model_pair.draft_path, "CPU", **PARITY_PROPERTIES, **model_pair.draft_properties),
        scheduler_config=scheduler_config,
        **PARITY_PROPERTIES,
    )
    result = generate(ov_pipe_with_draft, num_assistant_tokens=3)

    assert result.texts == reference.texts
    assert result.perf_metrics.get_num_input_tokens() == reference.perf_metrics.get_num_input_tokens()
    assert result.extended_perf_metrics.get_num_draft_tokens() > 0
    assert result.scores == pytest.approx(reference.scores, abs=1e-4)


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="VLM add_request and generate(ChatHistory) keep prompt token IDs only when the request config already "
    "assists generation, so DFlash prefix caching rejects requests with num_assistant_tokens unset or 0, CVS-199580",
)
@pytest.mark.parametrize("model_pair", VLM_TARGET_PAIRS, indirect=True)
def test_vlm_prefix_caching_without_explicit_assistant_tokens(model_pair, cat_tensor):
    # num_assistant_tokens is unset, so the draft drafts with its default; with 0 the request fails the same way.
    # Both need prompt token IDs for prefix caching on embedding inputs.
    config = GenerationConfig(max_new_tokens=20, do_sample=False, ignore_eos=True)
    history = ChatHistory()
    history.append({"role": "user", "content": PROMPTS[2]})

    def run(pipe):
        handle = pipe.add_request(0, PROMPTS[2], images=[cat_tensor], generation_config=config)
        while pipe.has_non_finished_requests():
            pipe.step()
        assert handle.get_status() == GenerationStatus.FINISHED
        generated_ids = list(handle.read_all()[0].generated_ids)
        chat_text = pipe.generate([history], images=[[cat_tensor]], generation_config=[config])[0].texts[0]
        return generated_ids, chat_text

    no_prefix_config = SchedulerConfig()
    no_prefix_config.enable_prefix_caching = False
    reference = run(ContinuousBatchingPipeline(model_pair.target_path, no_prefix_config, "CPU", PARITY_PROPERTIES))
    prefix_config = SchedulerConfig()
    prefix_config.enable_prefix_caching = True
    draft = draft_model(model_pair.draft_path, "CPU", **PARITY_PROPERTIES, **model_pair.draft_properties)
    pipe = ContinuousBatchingPipeline(
        model_pair.target_path, prefix_config, "CPU", {**PARITY_PROPERTIES, "draft_model": draft}
    )

    assert run(pipe) == reference


@pytest.mark.parametrize("model_pair", [QWEN3_DFLASH], indirect=True)
def test_draft_runs_on_paged_attention_only(model_pair):
    config = _greedy_config()
    prompts = PROMPTS_4[:1]
    reference = _target_ids(model_pair, prompts, config)

    def pipeline(attention_backend: str):
        draft = draft_model(model_pair.draft_path, "CPU", ATTENTION_BACKEND=attention_backend, **PARITY_PROPERTIES)
        return ContinuousBatchingPipeline(
            model_pair.target_path, SchedulerConfig(), "CPU", {**PARITY_PROPERTIES, "draft_model": draft}
        )

    assert [pipeline("PA").generate(prompts, [config])[0].m_generation_ids] == reference
    with pytest.raises(RuntimeError, match="PagedAttention only"):
        pipeline("SDPA")


def _negated_weights_copy(model_path: Path, copy_path: Path) -> Path:
    # The graph of the model with other weights: every floating-point constant of more than one value changes sign.
    shutil.copytree(model_path, copy_path)
    weights = np.fromfile(model_path / "openvino_model.bin", dtype=np.uint8)
    dtypes = {"f16": np.float16, "f32": np.float32}
    for data in ElementTree.parse(model_path / "openvino_model.xml").iter("data"):
        dtype = dtypes.get(data.get("element_type"))
        if dtype is not None and "offset" in data.attrib and int(data.get("size")) > np.dtype(dtype).itemsize:
            begin = int(data.get("offset"))
            weights[begin : begin + int(data.get("size"))].view(dtype)[:] *= -1
    weights.tofile(copy_path / "openvino_model.bin")
    return copy_path


@pytest.mark.parametrize("model_pair", [QWEN3_DFLASH], indirect=True)
def test_drafts_without_properties_do_not_share_cached_blobs(model_pair, tmp_path):
    # A draft without properties takes the target's, whose model path keys the cached target in place of its weights.
    other_draft_path = _negated_weights_copy(model_pair.draft_path, tmp_path / "draft")
    cache_dir = tmp_path / "cache"
    properties = {**PARITY_PROPERTIES, "CACHE_DIR": str(cache_dir)}

    for path in (model_pair.draft_path, other_draft_path):
        ContinuousBatchingPipeline(
            model_pair.target_path, SchedulerConfig(), "CPU", {**properties, "draft_model": draft_model(path, "CPU")}
        )

    # each pipeline compiles a draft and a target, which takes the FC weights of its draft
    assert len(list(cache_dir.glob("*.blob"))) == 4


@pytest.mark.parametrize("model_pair", [QWEN3_DSPARK, MINICPMV47_DSPARK], indirect=True)
def test_dspark_draft_needs_its_markov_head_model(model_pair, tmp_path):
    draft_dir = tmp_path / "no_markov_head"
    shutil.copytree(model_pair.draft_path, draft_dir)
    for suffix in (".xml", ".bin"):
        (draft_dir / f"openvino_markov_head_model{suffix}").unlink()
    with pytest.raises(RuntimeError, match="openvino_markov_head_model.xml is not next to it"):
        draft_model(draft_dir, "CPU")


@pytest.mark.parametrize("model_pair", [QWEN3_DFLASH, QWEN3_DEEPSPEC, QWEN35_DFLASH], indirect=True)
def test_markov_head_top_k_needs_a_draft_with_a_markov_head(model_pair):
    with pytest.raises(RuntimeError, match="applies only to drafts with a Markov head"):
        draft_model(model_pair.draft_path, "CPU", markov_head_top_k=4)
