# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Exercise GGUF MTP conversion and pipeline constraints without downloading weights."""

import numpy as np
import pytest
from gguf import GGUFWriter

import openvino as ov
import openvino_genai as genai


def write_qwen35_gguf(path, mtp=True):
    writer = GGUFWriter(path, "qwen35")
    for key, value in {
        "block_count": 3,
        "embedding_length": 16,
        "context_length": 256,
        "attention.head_count": 4,
        "attention.head_count_kv": 2,
        "rope.dimension_count": 4,
        "nextn_predict_layers": int(mtp),
        "full_attention_interval": 1,
        "ssm.conv_kernel": 4,
        "ssm.state_size": 4,
        "ssm.group_count": 1,
        "ssm.time_step_rank": 2,
        "ssm.inner_size": 8,
    }.items():
        writer.add_uint32("qwen35." + key, value)
    writer.add_float32("qwen35.attention.layer_norm_rms_epsilon", 1e-5)
    if mtp:
        writer.add_int32("general.sampling.top_k", 20)
        writer.add_float32("general.sampling.top_p", 0.95)
        writer.add_float32("general.sampling.temp", 0.6)
    writer.add_array("qwen35.rope.dimension_sections", [1, 1, 0, 0])
    tokens = ["<unk>", "<s>", "</s>", "<pad>", "\u2581", "a", "b", "c"]
    tokens += [f"<0x{i:02X}>" for i in range(256)]
    tokens += [f"unused{i}" for i in range(512 - len(tokens))]
    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * len(tokens))
    writer.add_token_types([2, 3, 3, 3] + [1] * 4 + [6] * 256 + [4] * (512 - 264))
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)
    writer.add_add_space_prefix(False)
    writer.add_chat_template(
        "{% for message in messages %}{{ message['role'] }} {{ message['content'] }} {% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )

    rng = np.random.default_rng(42)

    def weight(name, shape, norm=False):
        data = rng.normal(1.0 if norm else 0.0, 0.1, shape).astype(np.float32)
        writer.add_tensor(name, data)

    weight("token_embd.weight", (512, 16))
    weight("output.weight", (512, 16))
    weight("output_norm.weight", (16,), True)
    for layer in range(3):
        prefix = f"blk.{layer}."
        for name in ("attn_norm", "ffn_norm"):
            weight(prefix + name + ".weight", (16,), True)
        for name in ("attn_q_norm", "attn_k_norm"):
            weight(prefix + name + ".weight", (4,), True)
        for name, shape in {
            "attn_q": (32, 16),
            "attn_k": (8, 16),
            "attn_v": (8, 16),
            "attn_output": (16, 16),
            "ffn_gate": (32, 16),
            "ffn_up": (32, 16),
            "ffn_down": (16, 32),
        }.items():
            weight(prefix + name + ".weight", shape)
        if mtp and layer == 2:
            weight(prefix + "nextn.eh_proj.weight", (16, 32))
            for name in ("enorm", "hnorm", "shared_head_norm"):
                weight(prefix + "nextn." + name + ".weight", (16,), True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


@pytest.fixture(scope="module")
def mtp_gguf(tmp_path_factory):
    return write_qwen35_gguf(tmp_path_factory.mktemp("gguf_mtp") / "model.gguf")


@pytest.fixture(scope="module")
def plain_gguf(tmp_path_factory):
    return write_qwen35_gguf(tmp_path_factory.mktemp("gguf_plain") / "model.gguf", mtp=False)


DEVICE_PROPERTIES = dict(INFERENCE_PRECISION_HINT="f32", DYNAMIC_QUANTIZATION_GROUP_SIZE=0, INFERENCE_NUM_THREADS=2)


def make_scheduler():
    scheduler = genai.SchedulerConfig()
    scheduler.max_num_seqs = 1
    scheduler.max_num_batched_tokens = 32
    scheduler.num_kv_blocks = 32
    scheduler.enable_prefix_caching = False
    return scheduler


def mtp_properties(**extra):
    return {"speculative_type": "mtp", **DEVICE_PROPERTIES, **extra}


def test_gguf_mtp_rejects_legacy_reader(mtp_gguf):
    with pytest.raises(RuntimeError, match='requires GGUF_READER="FRONTEND"'):
        genai.ContinuousBatchingPipeline(mtp_gguf, make_scheduler(), "CPU", mtp_properties(GGUF_READER="LEGACY"))


def test_gguf_mtp_checkpoint_loads_as_regular_target(mtp_gguf):
    assert genai.LLMPipeline(mtp_gguf, "CPU", GGUF_READER="FRONTEND", **DEVICE_PROPERTIES) is not None


@pytest.mark.parametrize(
    "properties, message",
    [
        (dict(speculative_type="eagle3"), "Unsupported speculative_type"),
        (dict(speculative_type="mtp", prompt_lookup=True), "mutually exclusive"),
        (dict(speculative_type="mtp", ATTENTION_BACKEND="SDPA"), "paged attention backend"),
    ],
)
def test_gguf_speculative_type_rejects_invalid_requests(mtp_gguf, properties, message):
    with pytest.raises(RuntimeError, match=message):
        genai.LLMPipeline(mtp_gguf, "CPU", **properties)


def test_gguf_mtp_draft_runs_only_with_its_target(mtp_gguf):
    with pytest.raises(RuntimeError, match="runs only with its own GGUF target"):
        genai.draft_model(mtp_gguf, "CPU")


def test_gguf_mtp_requires_nextn_layers(plain_gguf):
    with pytest.raises(RuntimeError, match="must contain supported MTP layers"):
        genai.ContinuousBatchingPipeline(plain_gguf, make_scheduler(), "CPU", mtp_properties())


def test_gguf_mtp_draft_path_must_name_the_target(mtp_gguf, plain_gguf):
    with pytest.raises(RuntimeError, match="same file"):
        genai.ContinuousBatchingPipeline(
            plain_gguf, make_scheduler(), "CPU", {"draft_model_path": str(mtp_gguf), "GGUF_READER": "FRONTEND"}
        )


def test_gguf_mtp_draft_path_must_exist(mtp_gguf, tmp_path):
    with pytest.raises(RuntimeError, match="same file"):
        genai.LLMPipeline(mtp_gguf, "CPU", draft_model_path=str(tmp_path / "missing.gguf"), GGUF_READER="FRONTEND")


@pytest.fixture(scope="module")
def paired_pipelines(mtp_gguf):
    reader = dict(GGUF_READER="FRONTEND", **DEVICE_PROPERTIES)
    target = genai.ContinuousBatchingPipeline(mtp_gguf, make_scheduler(), "CPU", reader)
    mtp = genai.ContinuousBatchingPipeline(mtp_gguf, make_scheduler(), "CPU", mtp_properties())
    return target, mtp


@pytest.mark.parametrize(("prompt", "drafts"), [("a b c", True), ("a", False)])
def test_gguf_mtp_greedy_parity(paired_pipelines, prompt, drafts):
    target, mtp = paired_pipelines
    config = genai.GenerationConfig(max_new_tokens=8, ignore_eos=True, apply_chat_template=False)
    token_count = target.get_tokenizer().encode(prompt, add_special_tokens=False).input_ids.get_size()
    assert (token_count > 1) == drafts

    def generate_ids(pipe):
        handle = pipe.add_request(0, prompt, config)
        while pipe.has_non_finished_requests():
            pipe.step()
        return handle.read_all()[0].generated_ids

    reference = generate_ids(target)
    assert len(reference) == config.max_new_tokens
    assert generate_ids(mtp) == reference
    result = mtp.generate([prompt], [config])[0]
    assert (result.extended_perf_metrics.get_num_draft_tokens() > 0) == drafts


@pytest.mark.parametrize("property_name", ["speculative_type", "draft_model_path"])
def test_gguf_mtp_llm_pipeline_greedy_parity(mtp_gguf, property_name):
    config = genai.GenerationConfig(max_new_tokens=8, ignore_eos=True, apply_chat_template=False)
    reference = genai.LLMPipeline(mtp_gguf, "CPU", GGUF_READER="FRONTEND", ATTENTION_BACKEND="PA", **DEVICE_PROPERTIES)
    value = "mtp" if property_name == "speculative_type" else str(mtp_gguf)
    mtp = genai.LLMPipeline(mtp_gguf, "CPU", GGUF_READER="FRONTEND", **{property_name: value}, **DEVICE_PROPERTIES)
    assert mtp.generate("a b c", config) == reference.generate("a b c", config)


def test_gguf_mtp_chat_history_with_prefix_caching(mtp_gguf):
    scheduler = make_scheduler()
    scheduler.enable_prefix_caching = True
    target = genai.ContinuousBatchingPipeline(
        mtp_gguf, scheduler, "CPU", dict(GGUF_READER="FRONTEND", **DEVICE_PROPERTIES)
    )
    mtp = genai.ContinuousBatchingPipeline(mtp_gguf, scheduler, "CPU", mtp_properties())
    config = genai.GenerationConfig(max_new_tokens=8, ignore_eos=True)
    history = genai.ChatHistory()
    history.append({"role": "user", "content": "a b c"})
    reference = target.generate([history], generation_config=[config])[0]
    result = mtp.generate([history], generation_config=[config])[0]
    assert result.m_generation_ids == reference.m_generation_ids
    assert result.extended_perf_metrics.get_num_draft_tokens() > 0


def test_gguf_mtp_rejects_images(paired_pipelines):
    _, mtp = paired_pipelines
    config = genai.GenerationConfig(max_new_tokens=2, apply_chat_template=False)
    image = ov.Tensor(np.zeros((1, 8, 8, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="does not accept images"):
        mtp.add_request(0, "a", [image], config)


def test_gguf_mtp_accepts_disabled_prompt_lookup(mtp_gguf):
    pipe = genai.LLMPipeline(mtp_gguf, "CPU", prompt_lookup=False, **mtp_properties())
    config = genai.GenerationConfig(max_new_tokens=2, ignore_eos=True, apply_chat_template=False)
    assert pipe.generate("a b c", config)


def test_gguf_recommended_sampling_is_the_default_config(mtp_gguf, plain_gguf):
    reader = dict(GGUF_READER="FRONTEND", **DEVICE_PROPERTIES)
    configs = [
        genai.LLMPipeline(mtp_gguf, "CPU", **reader).get_generation_config(),
        genai.ContinuousBatchingPipeline(mtp_gguf, make_scheduler(), "CPU", reader).get_config(),
        genai.ContinuousBatchingPipeline(mtp_gguf, make_scheduler(), "CPU", mtp_properties()).get_config(),
    ]
    for config in configs:
        assert config.do_sample
        assert (config.top_k, config.top_p, config.temperature) == (20, pytest.approx(0.95), pytest.approx(0.6))
    assert configs[2].num_assistant_tokens is None
    assert not genai.LLMPipeline(plain_gguf, "CPU", **reader).get_generation_config().do_sample
