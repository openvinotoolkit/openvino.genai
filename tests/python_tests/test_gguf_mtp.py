# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Exercise GGUF MTP conversion and pipeline constraints without downloading weights."""

import numpy as np
import openvino as ov
import pytest
from gguf import GGUFWriter
from openvino import opset13 as ops

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


def test_gguf_mtp_rejects_legacy_reader(mtp_gguf):
    with pytest.raises(RuntimeError, match='requires? GGUF_READER="FRONTEND"'):
        genai.draft_model(mtp_gguf, "CPU", GGUF_READER="LEGACY")


def test_gguf_mtp_draft_loads(mtp_gguf):
    assert genai.draft_model(mtp_gguf, "CPU", GGUF_READER="FRONTEND", **DEVICE_PROPERTIES) is not None


def test_gguf_mtp_checkpoint_loads_as_regular_target(mtp_gguf):
    assert genai.LLMPipeline(mtp_gguf, "CPU", GGUF_READER="FRONTEND", **DEVICE_PROPERTIES) is not None


@pytest.mark.parametrize(
    "properties, message",
    [
        (dict(speculative_type="eagle3"), "Unsupported speculative_type"),
        (dict(speculative_type="mtp", prompt_lookup=True), "mutually exclusive"),
    ],
)
def test_gguf_speculative_type_rejects_invalid_requests(mtp_gguf, properties, message):
    with pytest.raises(RuntimeError, match=message):
        genai.LLMPipeline(mtp_gguf, "CPU", **properties)


def test_gguf_speculative_type_rejects_explicit_draft(mtp_gguf):
    draft = genai.draft_model(mtp_gguf, "CPU")
    with pytest.raises(RuntimeError, match="cannot be combined"):
        genai.LLMPipeline(mtp_gguf, "CPU", speculative_type="mtp", draft_model=draft)


def test_gguf_mtp_requires_nextn_layers(plain_gguf):
    with pytest.raises(RuntimeError, match="must contain supported MTP layers"):
        genai.draft_model(plain_gguf, "CPU")


@pytest.mark.parametrize("property_name", ["speculative_type", "draft_model_path"])
def test_gguf_mtp_rejects_llm_pipeline(mtp_gguf, property_name):
    with pytest.raises(RuntimeError, match="only by ContinuousBatchingPipeline"):
        value = "mtp" if property_name == "speculative_type" else str(mtp_gguf)
        genai.LLMPipeline(mtp_gguf, "CPU", **{property_name: value})


def test_gguf_mtp_rejects_explicit_draft_in_llm_pipeline(mtp_gguf):
    draft = genai.draft_model(mtp_gguf, "CPU", GGUF_READER="FRONTEND")
    with pytest.raises(RuntimeError, match="only by ContinuousBatchingPipeline"):
        genai.LLMPipeline(mtp_gguf, "CPU", draft_model=draft, GGUF_READER="FRONTEND")


def test_gguf_mtp_rejects_raw_gguf_continuous_batching_target(mtp_gguf):
    with pytest.raises(RuntimeError, match="converted VLM model directory"):
        genai.ContinuousBatchingPipeline(
            mtp_gguf,
            make_scheduler(),
            "CPU",
            {"draft_model_path": str(mtp_gguf), "GGUF_READER": "FRONTEND", **DEVICE_PROPERTIES},
        )


@pytest.fixture(scope="module")
def paired_pipelines(mtp_gguf, tmp_path_factory):
    reader = dict(GGUF_READER="FRONTEND", **DEVICE_PROPERTIES)
    source = genai.LLMPipeline(mtp_gguf, "CPU", enable_save_ov_model=True, **reader)
    tokenizer = source.get_tokenizer()
    target_dir = tmp_path_factory.mktemp("gguf_mtp_target")
    model = ov.Core().read_model(mtp_gguf.parent / "openvino_model.xml")
    ids = model.input("input_ids")
    # Split the synthetic model's token embedding lookup to form a matching VLM target.
    lookup = next(
        node
        for node in model.get_ops()
        if node.get_type_name() == "Gather" and node.input_value(0).partial_shape == ov.PartialShape([512, 16])
    )
    embeds = ops.parameter([-1, -1, 16], ov.Type.f32, name="inputs_embeds")
    embeds.output(0).set_names({"inputs_embeds"})
    lookup.output(0).replace(embeds.output(0))
    for consumer in ids.get_target_inputs():
        if consumer.get_node().get_type_name() == "ShapeOf":
            shape = ops.gather(ops.shape_of(embeds), [0, 1], 0)
            consumer.get_node().output(0).replace(shape.output(0))
    # Qwen VLM supplies four position lanes; text uses the same positions in each lane.
    positions = model.input("position_ids")
    mrope = ops.parameter([4, -1, -1], positions.element_type, name="position_ids")
    mrope.output(0).set_names({"position_ids"})
    positions.replace(ops.reshape(ops.gather(mrope, 0, 0), [1, -1], False).output(0))
    model.remove_parameter(ids.get_node())
    model.remove_parameter(positions.get_node())
    model.add_parameters([embeds, mrope])
    model.validate_nodes_and_infer_types()
    ov.serialize(model, target_dir / "openvino_model.xml")
    lookup_model = ov.Model([ops.gather(lookup.input_value(0), ids, 0)], [ids.get_node()])
    ov.serialize(lookup_model, target_dir / "openvino_text_embeddings_model.xml")
    pixels = ops.parameter([-1, 16], ov.Type.f32)
    ov.serialize(ov.Model([pixels], [pixels]), target_dir / "openvino_vision_embeddings_model.xml")
    positions = ops.parameter([4, -1, 16], ov.Type.f32)
    ov.serialize(ov.Model([positions], [positions]), target_dir / "openvino_vision_embeddings_pos_model.xml")
    (target_dir / "config.json").write_text('{"model_type":"qwen3_5","hidden_size":16}', encoding="utf-8")

    scheduler = make_scheduler()
    scheduler.enable_prefix_caching = True
    target = genai.ContinuousBatchingPipeline(target_dir, tokenizer, scheduler, "CPU", **DEVICE_PROPERTIES)
    mtp = genai.ContinuousBatchingPipeline(
        target_dir, tokenizer, scheduler, "CPU", draft_model_path=str(mtp_gguf), **reader
    )
    return target, mtp


@pytest.mark.parametrize(("prompt", "drafts"), [("a b c", True), ("a", False)])
def test_gguf_mtp_greedy_parity(paired_pipelines, prompt, drafts):
    target, mtp = paired_pipelines
    config = genai.GenerationConfig(max_new_tokens=8, ignore_eos=True, apply_chat_template=False)
    token_count = target.get_tokenizer().encode(prompt, add_special_tokens=False).input_ids.get_size()
    assert (token_count > 1) == drafts

    def generate_ids(pipe):
        handle = pipe.add_request(0, prompt, images=[], generation_config=config)
        while pipe.has_non_finished_requests():
            pipe.step()
        return handle.read_all()[0].generated_ids

    reference = generate_ids(target)
    assert len(reference) == config.max_new_tokens
    assert generate_ids(mtp) == reference
    result = mtp.generate([prompt], [config])[0]
    assert (result.extended_perf_metrics.get_num_draft_tokens() > 0) == drafts


def test_gguf_mtp_chat_history_with_prefix_caching(paired_pipelines):
    target, mtp = paired_pipelines
    config = genai.GenerationConfig(max_new_tokens=8, ignore_eos=True)
    history = genai.ChatHistory()
    history.append({"role": "user", "content": "a b c"})
    reference = target.generate([history], images=[[]], generation_config=[config])[0]
    result = mtp.generate([history], images=[[]], generation_config=[config])[0]
    assert result.texts == reference.texts
    assert result.extended_perf_metrics.get_num_draft_tokens() > 0
