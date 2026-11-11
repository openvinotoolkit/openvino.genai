# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Exercise embedded Qwen3.5 GGUF MTP without downloading model weights."""

import numpy as np
import pytest
from gguf import GGUFWriter

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


class TokenCollector(genai.StreamerBase):
    def __init__(self, status=genai.StreamingStatus.RUNNING):
        super().__init__()
        self.tokens = []
        self.status = status

    def write(self, tokens):
        self.tokens.extend([tokens] if isinstance(tokens, int) else tokens)
        return self.status

    def end(self):
        pass


DEVICE_PROPERTIES = dict(INFERENCE_PRECISION_HINT="f32", DYNAMIC_QUANTIZATION_GROUP_SIZE=0, INFERENCE_NUM_THREADS=2)


def make_scheduler():
    scheduler = genai.SchedulerConfig()
    scheduler.max_num_seqs = 1
    scheduler.max_num_batched_tokens = 32
    scheduler.num_kv_blocks = 32
    scheduler.enable_prefix_caching = False
    return scheduler


@pytest.fixture(scope="module")
def pipelines(mtp_gguf):
    scheduler = make_scheduler()
    properties = dict(GGUF_READER="FRONTEND", **DEVICE_PROPERTIES)
    plain = genai.LLMPipeline(mtp_gguf, "CPU", scheduler_config=scheduler, **properties)
    draft = genai.draft_model(mtp_gguf, "CPU", **properties)
    mtp = genai.LLMPipeline(mtp_gguf, "CPU", scheduler_config=scheduler, draft_model=draft, **properties)
    return plain, mtp


@pytest.mark.parametrize("candidates", [1, 2])
def test_gguf_mtp_matches_greedy_target_and_resets(pipelines, candidates):
    plain, mtp = pipelines
    for prompt in ("a b c", "c b a " * 12, "a b c"):
        expected, actual = TokenCollector(), TokenCollector()
        config = dict(max_new_tokens=12, ignore_eos=True, apply_chat_template=False)
        plain.generate(prompt, streamer=expected, **config)
        mtp.generate(prompt, streamer=actual, num_assistant_tokens=candidates, **config)
        assert len(actual.tokens) == 12
        assert actual.tokens == expected.tokens


def test_gguf_mtp_token_input_matches_target(pipelines):
    plain, mtp = pipelines
    input_ids = plain.get_tokenizer().encode("c b a " * 12, add_special_tokens=False).input_ids
    config = dict(max_new_tokens=12, ignore_eos=True, apply_chat_template=False)
    expected = plain.generate(input_ids, **config)
    actual = mtp.generate(input_ids, num_assistant_tokens=2, **config)
    assert len(actual.tokens[0]) == 12
    assert actual.tokens == expected.tokens


@pytest.mark.parametrize("status", [genai.StreamingStatus.STOP, genai.StreamingStatus.CANCEL])
def test_gguf_mtp_streaming_terminal_status(pipelines, status):
    _, mtp = pipelines
    stream = TokenCollector(status)
    mtp.generate(
        "a b c", max_new_tokens=12, ignore_eos=True, apply_chat_template=False, num_assistant_tokens=2, streamer=stream
    )
    assert stream.tokens
    resumed = TokenCollector()
    mtp.generate(
        "a b c", max_new_tokens=3, ignore_eos=True, apply_chat_template=False, num_assistant_tokens=2, streamer=resumed
    )
    assert len(resumed.tokens) == 3


def test_gguf_mtp_rejects_legacy_reader(mtp_gguf):
    with pytest.raises(RuntimeError, match='requires? GGUF_READER="FRONTEND"'):
        genai.draft_model(mtp_gguf, "CPU", GGUF_READER="LEGACY")


def test_gguf_mtp_chat_matches_target(pipelines):
    plain, mtp = pipelines
    for pipeline in (plain, mtp):
        pipeline.start_chat()
    try:
        for prompt in ("a b c", "c b a"):
            expected, actual = TokenCollector(), TokenCollector()
            config = dict(max_new_tokens=8, ignore_eos=True)
            plain.generate(prompt, streamer=expected, **config)
            mtp.generate(prompt, streamer=actual, num_assistant_tokens=2, **config)
            assert actual.tokens == expected.tokens
    finally:
        for pipeline in (plain, mtp):
            pipeline.finish_chat()


@pytest.mark.parametrize(
    "status",
    [
        genai.StreamingStatus.RUNNING,
        genai.StreamingStatus.STOP,
        genai.StreamingStatus.CANCEL,
    ],
)
def test_gguf_mtp_continuous_batching_preserves_status(mtp_gguf, status):
    properties = dict(
        GGUF_READER="FRONTEND",
        INFERENCE_PRECISION_HINT="f32",
        DYNAMIC_QUANTIZATION_GROUP_SIZE=0,
        INFERENCE_NUM_THREADS=2,
    )
    draft = genai.draft_model(mtp_gguf, "CPU", **properties)
    scheduler = genai.SchedulerConfig()
    scheduler.num_kv_blocks = 32
    scheduler.enable_prefix_caching = False
    pipeline = genai.ContinuousBatchingPipeline(mtp_gguf, scheduler, "CPU", {"draft_model": draft, **properties})
    config = genai.GenerationConfig(
        max_new_tokens=8, ignore_eos=True, num_assistant_tokens=2, apply_chat_template=False
    )
    results = pipeline.generate(["a b c"], [config], TokenCollector(status))
    expected = {
        genai.StreamingStatus.RUNNING: genai.GenerationStatus.FINISHED,
        genai.StreamingStatus.STOP: genai.GenerationStatus.STOP,
        genai.StreamingStatus.CANCEL: genai.GenerationStatus.CANCEL,
    }
    assert results[0].m_request_id == 0
    assert results[0].m_status == expected[status]


def test_gguf_mtp_chat_history_matches_target(pipelines):
    plain, mtp = pipelines
    histories = [genai.ChatHistory(), genai.ChatHistory()]
    for prompt in ("a b c", "c b a"):
        for history in histories:
            history.append({"role": "user", "content": prompt})
        expected, actual = TokenCollector(), TokenCollector()
        config = dict(max_new_tokens=8, ignore_eos=True)
        reference = plain.generate(histories[0], streamer=expected, **config)
        result = mtp.generate(histories[1], streamer=actual, num_assistant_tokens=2, **config)
        assert actual.tokens == expected.tokens
        histories[0].append({"role": "assistant", "content": reference.texts[0]})
        histories[1].append({"role": "assistant", "content": result.texts[0]})


def generate_tokens(pipeline, prompt, **config):
    collector = TokenCollector()
    result = pipeline.generate(
        [prompt], streamer=collector, max_new_tokens=12, ignore_eos=True, apply_chat_template=False, **config
    )
    return collector.tokens, result


@pytest.mark.parametrize("with_scheduler", [True, False])
def test_gguf_speculative_type_mtp_uses_embedded_draft(pipelines, mtp_gguf, with_scheduler):
    plain, _ = pipelines
    scheduler = dict(scheduler_config=make_scheduler()) if with_scheduler else {}
    mtp = genai.LLMPipeline(mtp_gguf, "CPU", speculative_type="mtp", **scheduler, **DEVICE_PROPERTIES)
    for prompt in ("a b c", "c b a " * 12):
        expected, _ = generate_tokens(plain, prompt)
        actual, result = generate_tokens(mtp, prompt)
        assert actual == expected
        assert type(result.extended_perf_metrics).__name__ == "SDPerModelsPerfMetrics"


def test_gguf_draft_model_path_detects_mtp(pipelines, mtp_gguf):
    plain, _ = pipelines
    mtp = genai.LLMPipeline(
        mtp_gguf, "CPU", scheduler_config=make_scheduler(), draft_model_path=mtp_gguf, **DEVICE_PROPERTIES
    )
    expected, _ = generate_tokens(plain, "a b c")
    actual, _ = generate_tokens(mtp, "a b c", num_assistant_tokens=2)
    assert actual == expected


def test_gguf_mtp_accepts_separate_draft_for_plain_target(plain_gguf, mtp_gguf):
    properties = dict(GGUF_READER="FRONTEND", **DEVICE_PROPERTIES)
    plain = genai.LLMPipeline(plain_gguf, "CPU", scheduler_config=make_scheduler(), **properties)
    mtp = genai.LLMPipeline(
        plain_gguf, "CPU", scheduler_config=make_scheduler(), draft_model_path=mtp_gguf, **properties
    )
    expected, _ = generate_tokens(plain, "a b c")
    actual, _ = generate_tokens(mtp, "a b c", num_assistant_tokens=2)
    assert actual == expected


def test_gguf_speculative_type_mtp_continuous_batching(pipelines, mtp_gguf):
    plain, _ = pipelines
    pipeline = genai.ContinuousBatchingPipeline(
        mtp_gguf, make_scheduler(), "CPU", {"speculative_type": "mtp", **DEVICE_PROPERTIES}
    )
    config = genai.GenerationConfig(max_new_tokens=12, ignore_eos=True, apply_chat_template=False)
    _, expected = generate_tokens(plain, "a b c")
    assert pipeline.generate(["a b c"], [config])[0].m_generation_ids[0] == expected.texts[0]


@pytest.mark.parametrize(
    "properties, message",
    [
        (dict(speculative_type="eagle3"), "Unsupported speculative_type"),
        (dict(speculative_type="mtp", GGUF_READER="LEGACY"), "GGUF_READER"),
        (dict(draft_model_path="{model}", GGUF_READER="LEGACY"), "GGUF_READER"),
        (dict(speculative_type="mtp", prompt_lookup=True), "mutually exclusive"),
    ],
)
def test_gguf_speculative_type_rejects_invalid_requests(mtp_gguf, properties, message):
    properties = {key: str(mtp_gguf) if value == "{model}" else value for key, value in properties.items()}
    with pytest.raises(RuntimeError, match=message):
        genai.LLMPipeline(mtp_gguf, "CPU", **properties)


def test_gguf_speculative_type_rejects_explicit_draft(mtp_gguf):
    draft = genai.draft_model(mtp_gguf, "CPU")
    with pytest.raises(RuntimeError, match="cannot be combined"):
        genai.LLMPipeline(mtp_gguf, "CPU", speculative_type="mtp", draft_model=draft)


def test_gguf_mtp_requires_nextn_layers(plain_gguf, mtp_gguf):
    with pytest.raises(RuntimeError, match="requires MTP layers"):
        genai.LLMPipeline(plain_gguf, "CPU", speculative_type="mtp")
    with pytest.raises(RuntimeError, match="requires MTP layers"):
        genai.LLMPipeline(mtp_gguf, "CPU", speculative_type="mtp", draft_model_path=plain_gguf)
    with pytest.raises(RuntimeError, match="must contain MTP layers"):
        genai.LLMPipeline(mtp_gguf, "CPU", draft_model_path=plain_gguf)
