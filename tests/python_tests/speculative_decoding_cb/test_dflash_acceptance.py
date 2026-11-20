# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DFlash, DSpark and DFlash-2 drafts whose every candidate the target accepts.

The tiny random drafts of the shared speculative decoding scenarios reject almost every candidate. Here every token
embedding carries a large channel 0 that only token 0 reads out, so the target and all drafts predict token 0: greedy
speculative decoding must reproduce the target and accept every candidate, also when requests are drafted together,
split, preempted or cached.
"""

import functools
from copy import deepcopy
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
import pytest
from huggingface_hub import snapshot_download

from utils.network import retry_request

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
model_patcher = pytest.importorskip("optimum.exporters.openvino.model_patcher")
if not all(
    hasattr(model_patcher, name)
    for name in ("Qwen3DFlashForCausalLM", "Qwen3DSparkForCausalLM", "Qwen3DFlash2ForCausalLM")
):
    pytest.skip(
        "This optimum-intel version cannot export DSpark and DFlash-2 drafts (huggingface/optimum-intel#2098).",
        allow_module_level=True,
    )

TINY_QWEN3 = "optimum-intel-internal-testing/tiny-random-qwen3"
NUM_ASSISTANT_TOKENS = 4
MAX_NEW_TOKENS = 40
# prompts cross the 32-token blocks of the CPU draft KV cache
PROMPT_LENGTHS = (5, 37, 70)
PRECISION = {"INFERENCE_PRECISION_HINT": "f32", "KV_CACHE_PRECISION": "f32"}
VARIANTS = ("dflash", "dspark", "dflash2")
# draft variant and draft properties: DSpark's Markov head drafts with its candidate scorer or its dense step
DRAFTS = {
    "dflash": ("dflash", {}),
    "dspark": ("dspark", {}),
    "dspark-markov-step": ("dspark", {"markov_head_top_k": 0}),
    "dflash2": ("dflash2", {}),
}


def _keep_first_channel(norm):
    # ties would leave the argmax to TopK and sampler tie-breaking, which differ
    with torch.no_grad():
        norm.weight.zero_()
        norm.weight[0] = 1.0


@functools.cache
def _tiny_qwen3_dir():
    return retry_request(lambda: snapshot_download(TINY_QWEN3))


def _save_checkpoint(model, output_dir: Path, architectures):
    from transformers import AutoTokenizer

    model.save_pretrained(output_dir)
    model.config.architectures = architectures
    model.config.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(_tiny_qwen3_dir()).save_pretrained(output_dir)


def _build_target(output_dir: Path):
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    config = AutoConfig.from_pretrained(_tiny_qwen3_dir())
    config.tie_word_embeddings = False
    torch.manual_seed(1)
    model = Qwen3ForCausalLM(config).eval()
    with torch.no_grad():
        model.model.embed_tokens.weight[:, 0] = 50.0
        model.lm_head.weight[:, 0] = 0.0
        model.lm_head.weight[0, 0] = 1.0
    _keep_first_channel(model.model.norm)
    _save_checkpoint(model, output_dir, ["Qwen3ForCausalLM"])
    return config


def _build_draft(output_dir: Path, variant: str, target_config):
    config = deepcopy(target_config)
    num_target_layers = target_config.num_hidden_layers
    target_layer_ids = [0, num_target_layers - 1]
    mask_token_id = config.vocab_size - 1
    config.num_hidden_layers = 2
    config.num_target_layers = num_target_layers
    if getattr(config, "layer_types", None):
        config.layer_types = list(config.layer_types[: config.num_hidden_layers])
    config.is_causal = False
    if variant == "dflash":
        architectures = ["DFlashDraftModel"]
        config.dflash_config = {"mask_token_id": mask_token_id, "target_layer_ids": target_layer_ids}
        model_class = model_patcher.Qwen3DFlashForCausalLM
    elif variant == "dflash2":
        architectures = ["DFlash2DraftModel"]
        config.dflash_config = {
            "conv_group_size": 8 if config.hidden_size % 8 == 0 else 1,
            "conv_kernel_size": 2,
            "mask_token_id": mask_token_id,
            "selector_rank": 8,
            "selector_top_k": 4,
            "target_layer_ids": target_layer_ids,
        }
        model_class = model_patcher.Qwen3DFlash2ForCausalLM
    else:
        architectures = ["Qwen3DSparkModel"]
        config.markov_rank = 8
        config.mask_token_id = mask_token_id
        config.target_layer_ids = target_layer_ids
        model_class = model_patcher.Qwen3DSparkForCausalLM
    config.architectures = architectures
    torch.manual_seed(0)
    model = model_class(config).eval()
    # the draft reads out through the target embedding and LM head
    _keep_first_channel(model.norm)
    _save_checkpoint(model, output_dir, architectures)


def _export(checkpoint_dir: Path, output_dir: Path):
    from optimum.exporters.openvino import main_export

    main_export(str(checkpoint_dir), output_dir, task="text-generation-with-past", convert_tokenizer=True)
    return output_dir


@pytest.fixture(scope="module")
def dflash_models(tmp_path_factory):
    root = tmp_path_factory.mktemp("dflash_acceptance")
    target_config = _build_target(root / "target_hf")
    models = {"target": _export(root / "target_hf", root / "target")}
    for variant in VARIANTS:
        _build_draft(root / f"{variant}_hf", variant, target_config)
        models[variant] = _export(root / f"{variant}_hf", root / variant)
    return models


def _prompts(lengths=PROMPT_LENGTHS):
    rng = np.random.default_rng(7)
    return [ov.Tensor(rng.integers(10, 1000, size=(1, length), dtype=np.int64)) for length in lengths]


def _generation_config():
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = MAX_NEW_TOKENS
    config.ignore_eos = True
    config.num_assistant_tokens = NUM_ASSISTANT_TOKENS
    return config


def _target_config():
    config = _generation_config()
    config.num_assistant_tokens = 0
    return config


def _scheduler_config(max_num_batched_tokens=256, **options):
    config = ov_genai.SchedulerConfig()
    config.max_num_batched_tokens = max_num_batched_tokens
    config.dynamic_split_fuse = True
    config.enable_prefix_caching = False
    for name, value in options.items():
        setattr(config, name, value)
    return config


def _pipeline(models, draft=None, scheduler_config=None):
    properties = dict(PRECISION)
    if draft is not None:
        variant, head_properties = DRAFTS[draft]
        properties["draft_model"] = ov_genai.draft_model(models[variant], "CPU", **PRECISION, **head_properties)
    return ov_genai.ContinuousBatchingPipeline(
        models["target"], scheduler_config or _scheduler_config(), "CPU", properties
    )


def _generate(pipe, prompts, config):
    results = pipe.generate(prompts, [config] * len(prompts))
    return [result.m_generation_ids[0] for result in results], results


def _assert_every_candidate_accepted(results):
    metrics = results[0].extended_perf_metrics
    assert metrics.get_num_draft_tokens() > 0
    assert metrics.get_num_accepted_tokens() == metrics.get_num_draft_tokens()


@pytest.fixture(scope="module")
def target_outputs(dflash_models):
    outputs, _ = _generate(_pipeline(dflash_models), _prompts(), _target_config())
    assert all(len(tokens) == MAX_NEW_TOKENS for tokens in outputs)
    return outputs


@pytest.mark.parametrize("dynamic_split_fuse", [True, False])
@pytest.mark.parametrize("draft", DRAFTS)
def test_requests_drafted_together_accept_every_candidate(dflash_models, target_outputs, draft, dynamic_split_fuse):
    # Without dynamic split fuse a step runs either prompt or generation windows, so the first block follows the
    # prompt rows one draft step later.
    pipe = _pipeline(dflash_models, draft, _scheduler_config(dynamic_split_fuse=dynamic_split_fuse))
    outputs, results = _generate(pipe, _prompts(), _generation_config())
    assert outputs == target_outputs
    _assert_every_candidate_accepted(results)


@pytest.mark.parametrize("draft", DRAFTS)
def test_staggered_requests_accept_every_candidate(dflash_models, target_outputs, draft):
    pipe = _pipeline(dflash_models, draft)
    prompts = _prompts()
    handles = {}
    # a new request arrives while the earlier ones are prefilling, validating or drafting
    arrival_steps = {0: 0, 1: 3, 2: 7}
    step = 0
    while len(handles) < len(prompts) or pipe.has_non_finished_requests():
        for request_id, arrival in arrival_steps.items():
            if arrival == step:
                handles[request_id] = pipe.add_request(request_id, prompts[request_id], _generation_config())
        if pipe.has_non_finished_requests():
            pipe.step()
        step += 1
    outputs = [handles[request_id].read_all()[0].generated_ids for request_id in range(len(prompts))]
    assert outputs == target_outputs


@pytest.mark.parametrize("max_num_batched_tokens", [5, 7, 16])
@pytest.mark.parametrize("draft", DRAFTS)
def test_chunked_prefill_accepts_every_candidate(dflash_models, target_outputs, draft, max_num_batched_tokens):
    # Validation windows have 1 + NUM_ASSISTANT_TOKENS rows: 5 and 7 fit one window and defer the others to a later
    # step, 16 fits the windows of all three requests.
    pipe = _pipeline(dflash_models, draft, _scheduler_config(max_num_batched_tokens))
    outputs, results = _generate(pipe, _prompts(), _generation_config())
    assert outputs == target_outputs
    _assert_every_candidate_accepted(results)


@pytest.mark.parametrize("draft", DRAFTS)
def test_target_and_draft_preemption_complete(dflash_models, target_outputs, draft):
    # 6 blocks for the target and, by default, for the draft: the target recomputes preempted requests, and the
    # target finishes alone the requests whose draft windows do not run.
    pipe = _pipeline(dflash_models, draft, _scheduler_config(num_kv_blocks=6))
    outputs, results = _generate(pipe, _prompts(), _generation_config())
    assert outputs == target_outputs
    _assert_every_candidate_accepted(results)


@pytest.mark.parametrize("draft", DRAFTS)
def test_prefix_caching_accepts_every_candidate(dflash_models, target_outputs, draft):
    prompts = _prompts()
    pipe = _pipeline(dflash_models, draft, _scheduler_config(enable_prefix_caching=True))
    runs = [_generate(pipe, [prompt], _generation_config()) for prompt in prompts + prompts]
    assert [outputs[0] for outputs, _ in runs] == target_outputs + target_outputs
    for _, results in runs:
        _assert_every_candidate_accepted(results)


@pytest.mark.parametrize("draft", DRAFTS)
def test_greedy_requests_with_penalties_accept_every_candidate(dflash_models, draft):
    # Penalties can change the argmax, so the draft draws its candidates with its sampler, over the token history it
    # shares with the target.
    def penalized(config):
        config.repetition_penalty = 1.3
        config.presence_penalty = 0.2
        return config

    target, _ = _generate(_pipeline(dflash_models), _prompts(), penalized(_target_config()))
    outputs, results = _generate(_pipeline(dflash_models, draft), _prompts(), penalized(_generation_config()))
    assert outputs == target
    _assert_every_candidate_accepted(results)


@pytest.mark.parametrize("draft", DRAFTS)
def test_sampled_requests_accept_candidates(dflash_models, draft):
    # The target accepts or replaces every sampled candidate by speculative sampling against the distribution the
    # draft drew it from; the draws of both follow the request seed.
    config = _generation_config()
    config.do_sample = True
    config.temperature = 0.8
    # A Top-K candidate lattice proposes only its candidates: over the whole vocabulary, the tiny target would give
    # them too little probability to accept any.
    config.top_k = 4
    config.rng_seed = 3
    outputs, results = _generate(_pipeline(dflash_models, draft), _prompts(), config)
    assert all(len(tokens) == MAX_NEW_TOKENS for tokens in outputs)
    assert results[0].extended_perf_metrics.get_num_accepted_tokens() > 0
    assert _generate(_pipeline(dflash_models, draft), _prompts(), config)[0] == outputs
