# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil

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
    QWEN3_DEEPSPEC,
    QWEN3_DFLASH,
    QWEN3_DSPARK,
    QWEN35_DFLASH,
    SCORE_SKIP,
    VLM_TARGET_PAIRS,
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
