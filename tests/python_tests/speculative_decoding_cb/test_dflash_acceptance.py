# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DFlash, DSpark and DFlash-2 drafts whose every candidate the target accepts.

The tiny random drafts of the shared speculative decoding scenarios reject almost every candidate. Here every token
embedding carries a large channel 0 that only token 0 reads out, so the target and all drafts predict token 0: greedy
speculative decoding must reproduce the target and accept every candidate, also when requests are drafted together,
split or preempted. The pruned DSpark draft predicts a random subset of the target vocabulary, which holds token 0,
with its own lm_head.
"""

import functools
from copy import deepcopy
from pathlib import Path

import pytest
from huggingface_hub import snapshot_download

from test_speculative_decoding import (
    BLOCK_MAX_NEW_TOKENS,
    ModelPair,
    _block_config,
    _block_pipeline,
    _block_prompts,
    _block_scheduler_config,
    _generate,
    _target_pipeline,
)
from utils.network import retry_request

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
model_patcher = pytest.importorskip("optimum.exporters.openvino.model_patcher")
if not all(
    hasattr(model_patcher, name)
    for name in ("Qwen3DFlashForCausalLM", "Qwen3DSparkForCausalLM", "Qwen3DFlash2ForCausalLM")
):
    pytest.skip(
        "This optimum-intel version predates huggingface/optimum-intel#2098, which exports DSpark and DFlash-2 drafts.",
        allow_module_level=True,
    )
dflash_utils = pytest.importorskip("optimum.exporters.openvino.dflash_utils")

TINY_QWEN3 = "optimum-intel-internal-testing/tiny-random-qwen3"
VARIANTS = ("dflash", "dspark", "dspark_pruned", "dflash2")
PRUNED_DRAFT_VOCAB_SIZE = 512
# huggingface/optimum-intel#2099 exports DSpark drafts with a pruned vocabulary
PRUNED_VOCABULARY_EXPORT = hasattr(dflash_utils, "has_pruned_draft_vocabulary")
# draft variant and draft properties: DSpark's Markov head drafts with its candidate scorer or its dense step
DRAFTS = {
    "dflash": ("dflash", {}),
    "dspark": ("dspark", {}),
    "dspark-markov-step": ("dspark", {"markov_head_top_k": 0}),
    "dspark-pruned": ("dspark_pruned", {}),
    "dspark-pruned-markov-step": ("dspark_pruned", {"markov_head_top_k": 0}),
    "dflash2": ("dflash2", {}),
}
# no pair of the shared speculative decoding scenarios has a draft with a pruned vocabulary
PRUNED_DRAFTS = ("dspark-pruned", "dspark-pruned-markov-step")


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
    return model


def _build_draft(output_dir: Path, variant: str, target):
    target_config = target.config
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
        if variant == "dspark_pruned":
            config.draft_vocab_size = PRUNED_DRAFT_VOCAB_SIZE
            config.tie_word_embeddings = False
    config.architectures = architectures
    torch.manual_seed(0)
    model = model_class(config).eval()
    if variant == "dspark_pruned":
        # Draft token i is target token i + d2t[i], read out by the target LM head row of that token.
        others = torch.randperm(config.vocab_size - 1)[: PRUNED_DRAFT_VOCAB_SIZE - 1] + 1
        target_ids = torch.cat([others[:7], torch.zeros(1, dtype=torch.long), others[7:]])
        with torch.no_grad():
            model.d2t.copy_(target_ids - torch.arange(PRUNED_DRAFT_VOCAB_SIZE))
            model.lm_head.weight.copy_(target.lm_head.weight[target_ids])
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
    target = _build_target(root / "target_hf")
    models = {"target": _export(root / "target_hf", root / "target")}
    for variant in VARIANTS:
        if variant == "dspark_pruned" and not PRUNED_VOCABULARY_EXPORT:
            continue
        _build_draft(root / f"{variant}_hf", variant, target)
        models[variant] = _export(root / f"{variant}_hf", root / variant)
    return models


def _pair(models, draft=None) -> ModelPair:
    variant, draft_properties = DRAFTS[draft] if draft else (None, {})
    if variant is not None and variant not in models:
        pytest.skip(
            "This optimum-intel version cannot export DSpark drafts with a pruned vocabulary "
            "(huggingface/optimum-intel#2099)."
        )
    return ModelPair(models["target"], models.get(variant), draft_properties, is_vlm=False)


def _assert_every_candidate_accepted(results):
    metrics = results[0].extended_perf_metrics
    assert metrics.get_num_draft_tokens() > 0
    assert metrics.get_num_accepted_tokens() == metrics.get_num_draft_tokens()


@pytest.fixture(scope="module")
def target_outputs(dflash_models):
    pair = _pair(dflash_models)
    outputs, _ = _generate(_target_pipeline(pair, _block_scheduler_config()), _block_prompts(pair), _block_config(0))
    assert all(len(tokens) == BLOCK_MAX_NEW_TOKENS for tokens in outputs)
    return outputs


@pytest.mark.parametrize("dynamic_split_fuse", [True, False])
@pytest.mark.parametrize("draft", DRAFTS)
def test_requests_drafted_together_accept_every_candidate(dflash_models, target_outputs, draft, dynamic_split_fuse):
    # Without dynamic split fuse a step runs either prompt or generation windows, so the first block follows the
    # prompt rows one draft step later.
    pair = _pair(dflash_models, draft)
    pipe = _block_pipeline(pair, _block_scheduler_config(dynamic_split_fuse=dynamic_split_fuse))
    outputs, results = _generate(pipe, _block_prompts(pair), _block_config())
    assert outputs == target_outputs
    _assert_every_candidate_accepted(results)


@pytest.mark.parametrize("max_num_batched_tokens", [5, 7, 16])
@pytest.mark.parametrize("draft", DRAFTS)
def test_chunked_prefill_accepts_every_candidate(dflash_models, target_outputs, draft, max_num_batched_tokens):
    # Validation windows have 1 + BLOCK_NUM_ASSISTANT_TOKENS = 5 rows: 5 and 7 fit one window and defer the others to a
    # later step, 16 fits the windows of all three requests.
    pair = _pair(dflash_models, draft)
    pipe = _block_pipeline(pair, _block_scheduler_config(max_num_batched_tokens))
    outputs, results = _generate(pipe, _block_prompts(pair), _block_config())
    assert outputs == target_outputs
    _assert_every_candidate_accepted(results)


@pytest.mark.parametrize("draft", DRAFTS)
def test_target_and_draft_preemption_complete(dflash_models, target_outputs, draft):
    # 6 blocks for the target and, by default, for the draft: the target recomputes preempted requests, and the
    # target finishes alone the requests whose draft windows do not run.
    pair = _pair(dflash_models, draft)
    pipe = _block_pipeline(pair, _block_scheduler_config(num_kv_blocks=6))
    outputs, results = _generate(pipe, _block_prompts(pair), _block_config())
    assert outputs == target_outputs
    _assert_every_candidate_accepted(results)


@pytest.mark.parametrize("draft", DRAFTS)
def test_greedy_requests_with_penalties_accept_every_candidate(dflash_models, draft):
    # Penalties can change the argmax, so the draft draws its candidates with its sampler, over the token history it
    # shares with the target.
    def penalized(config):
        config.repetition_penalty = 1.3
        config.presence_penalty = 0.2
        return config

    pair = _pair(dflash_models, draft)
    prompts = _block_prompts(pair)
    target, _ = _generate(_target_pipeline(pair, _block_scheduler_config()), prompts, penalized(_block_config(0)))
    outputs, results = _generate(_block_pipeline(pair), prompts, penalized(_block_config()))
    assert outputs == target
    _assert_every_candidate_accepted(results)


@pytest.mark.parametrize("draft", PRUNED_DRAFTS)
def test_staggered_requests_accept_every_candidate(dflash_models, target_outputs, draft):
    pair = _pair(dflash_models, draft)
    pipe = _block_pipeline(pair)
    prompts = _block_prompts(pair)
    handles = {}
    # a new request arrives while the earlier ones are prefilling, validating or drafting
    arrival_steps = {0: 0, 1: 3, 2: 7}
    step = 0
    while len(handles) < len(prompts) or pipe.has_non_finished_requests():
        for request_id, arrival in arrival_steps.items():
            if arrival == step:
                handles[request_id] = pipe.add_request(request_id, prompts[request_id], _block_config())
        if pipe.has_non_finished_requests():
            pipe.step()
        step += 1
    outputs = [handles[request_id].read_all()[0].generated_ids for request_id in range(len(prompts))]
    assert outputs == target_outputs


@pytest.mark.parametrize("draft", PRUNED_DRAFTS)
def test_prefix_caching_accepts_every_candidate(dflash_models, target_outputs, draft):
    pair = _pair(dflash_models, draft)
    prompts = _block_prompts(pair)
    pipe = _block_pipeline(pair, _block_scheduler_config(enable_prefix_caching=True))
    runs = [_generate(pipe, [prompt], _block_config()) for prompt in prompts + prompts]
    assert [outputs[0] for outputs, _ in runs] == target_outputs + target_outputs
    for _, results in runs:
        _assert_every_candidate_accepted(results)


@pytest.mark.parametrize("draft", PRUNED_DRAFTS)
def test_sampled_requests_accept_candidates(dflash_models, draft):
    # The target accepts or replaces every sampled candidate by speculative sampling against the distribution the
    # draft drew it from; the draws of both follow the request seed.
    config = _block_config()
    config.do_sample = True
    config.temperature = 0.8
    # A Top-K candidate lattice proposes only its candidates: over the whole vocabulary, the tiny target would give
    # them too little probability to accept any.
    config.top_k = 4
    config.rng_seed = 3
    pair = _pair(dflash_models, draft)
    prompts = _block_prompts(pair)
    outputs, results = _generate(_block_pipeline(pair), prompts, config)
    assert all(len(tokens) == BLOCK_MAX_NEW_TOKENS for tokens in outputs)
    assert results[0].extended_perf_metrics.get_num_accepted_tokens() > 0
    assert _generate(_block_pipeline(pair), prompts, config)[0] == outputs
