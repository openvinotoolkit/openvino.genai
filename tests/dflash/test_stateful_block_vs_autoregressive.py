# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
import os
from pathlib import Path
import numpy as np
import pytest
PROMPTS = [
    "Write a compact C++ function that checks if a number is prime.",
    "Explain speculative decoding in one paragraph.",
    "Continue this sequence with code-like tokens: def foo(x):",
]
DEFAULT_MAX_NEW_TOKENS = 8
LOGIT_ATOL = 1e-3
KV_ATOL = 1e-3


@dataclass
class TensorDiff:
    max_abs_diff: float
    mean_abs_diff: float
    first_diff_index: int | None
    actual_at_first_diff: float | None
    expected_at_first_diff: float | None


@dataclass
class AutoregressiveTrace:
    generated_tokens: list[int]
    logits_by_prefix_len: dict[int, np.ndarray]
    state_by_prefix_len: dict[int, dict[str, np.ndarray]]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_model_cases():
    root = _repo_root()
    override = os.environ.get("OV_GENAI_BLOCK_VS_AR_MODEL")
    if override:
        path = Path(override)
        return [(path.name.replace("-", "_"), path)]
    return [
        ("qwen3_0_6b_fp16", root / "models" / "qwen3-0.6b-fp16-ov"),
        ("qwen3_0_6b_4bit", root / "models" / "qwen3-0.6b-4bit-ov"),
    ]


def _require_enabled():
    if os.environ.get("OV_GENAI_ENABLE_BLOCK_VS_AR") != "1":
        pytest.skip(
            "Set OV_GENAI_ENABLE_BLOCK_VS_AR=1 to run bare-model block-vs-autoregressive diagnostics")


def _compile_model(model_path: Path):
    ov = pytest.importorskip("openvino")
    import openvino.properties.hint as hints
    core = ov.Core()
    device = os.environ.get("OV_GENAI_BLOCK_VS_AR_DEVICE", "CPU")
    model = core.read_model(model_path / "openvino_model.xml")
    properties = {}
    if "GPU" not in device.upper():
        properties = {
            hints.inference_precision: ov.Type.f32,
            hints.kv_cache_precision: ov.Type.f32,
        }
    compiled_model = core.compile_model(model, device, properties)
    if "GPU" in device.upper():
        kv_cache_precision = compiled_model.get_property(hints.kv_cache_precision)
        assert kv_cache_precision in (ov.Type.f16, ov.Type.f32), (
            f"Unexpected GPU KV cache precision: {kv_cache_precision}. "
            "Expected f16 or f32 when precision hints are not forced."
        )
    return compiled_model


def _state_name(state) -> str:
    return state.name if hasattr(state, "name") else state.get_name()


def _snapshot_state(request) -> dict[str, np.ndarray]:
    return {
        _state_name(state): np.array(state.state.data, copy=True)
        for state in request.query_state()
    }


def _input_names(compiled_model) -> set[str]:
    return {compiled_model.input(idx).get_any_name() for idx in range(len(compiled_model.inputs))}


def _np_dtype(element_type):
    ov = pytest.importorskip("openvino")
    if element_type == ov.Type.i32:
        return np.int32
    if element_type == ov.Type.i64:
        return np.int64
    raise AssertionError(f"Unsupported integral input type: {element_type}")


def _set_tensor(request, compiled_model, name: str, values):
    ov = pytest.importorskip("openvino")
    dtype = _np_dtype(compiled_model.input(name).get_element_type())
    request.set_tensor(name, ov.Tensor(np.asarray(values, dtype=dtype)))


def _infer_tokens(request, compiled_model, token_ids: list[int], position_start: int, total_sequence_length: int) -> np.ndarray:
    input_names = _input_names(compiled_model)
    _set_tensor(request, compiled_model, "input_ids", [token_ids])
    _set_tensor(request, compiled_model, "attention_mask", [
                np.ones(total_sequence_length, dtype=np.int64)])
    _set_tensor(request, compiled_model, "position_ids", [
                np.arange(position_start, position_start + len(token_ids))])
    if "beam_idx" in input_names:
        _set_tensor(request, compiled_model, "beam_idx", [0])
    request.infer()
    return np.array(request.get_output_tensor(0).data, copy=True)


def _encode_prompt(model_path: Path, prompt: str) -> list[int]:
    ov_genai = pytest.importorskip("openvino_genai")
    tokenizer = ov_genai.Tokenizer(model_path)
    return tokenizer.encode(prompt).input_ids.data[0].astype(np.int64).tolist()


def _build_autoregressive_trace(compiled_model, prompt_tokens: list[int], max_new_tokens: int) -> AutoregressiveTrace:
    request = compiled_model.create_infer_request()
    prompt_len = len(prompt_tokens)
    logits = _infer_tokens(request, compiled_model,
                           prompt_tokens, 0, prompt_len)
    generated_tokens: list[int] = []
    logits_by_prefix_len: dict[int, np.ndarray] = {}
    state_by_prefix_len: dict[int, dict[str, np.ndarray]] = {}
    next_token = int(np.argmax(logits[0, -1]))
    for token_idx in range(max_new_tokens):
        generated_tokens.append(next_token)
        prefix_len = token_idx + 1
        position = prompt_len + token_idx
        logits = _infer_tokens(request, compiled_model, [
                               next_token], position, position + 1)
        logits_by_prefix_len[prefix_len] = logits[0, -1].copy()
        state_by_prefix_len[prefix_len] = _snapshot_state(request)
        next_token = int(np.argmax(logits[0, -1]))
    return AutoregressiveTrace(generated_tokens, logits_by_prefix_len, state_by_prefix_len)


def _diff_tensors(actual: np.ndarray, expected: np.ndarray, atol: float) -> TensorDiff | None:
    assert actual.shape == expected.shape, f"Shape mismatch: actual={actual.shape}, expected={expected.shape}"
    diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    first = np.flatnonzero(diff > atol)
    if first.size == 0:
        return None
    first_idx = int(first[0])
    return TensorDiff(
        max_abs_diff=float(diff.max()),
        mean_abs_diff=float(diff.mean()),
        first_diff_index=first_idx,
        actual_at_first_diff=float(actual.reshape(-1)[first_idx]),
        expected_at_first_diff=float(expected.reshape(-1)[first_idx]),
    )


def _compare_logits(actual: np.ndarray, expected: np.ndarray, prompt: str, prefix_len: int, model_path: Path):
    actual_argmax = int(np.argmax(actual))
    expected_argmax = int(np.argmax(expected))
    diff = _diff_tensors(actual, expected, LOGIT_ATOL)
    assert actual_argmax == expected_argmax and diff is None, (
        f"Logits diverged for model={model_path}, prompt={prompt!r}, prefix_len={prefix_len}\n"
        f"actual_argmax={actual_argmax}, expected_argmax={expected_argmax}, diff={diff}"
    )


def _compare_states(actual: dict[str, np.ndarray], expected: dict[str, np.ndarray], prompt: str, prefix_len: int, model_path: Path):
    assert actual.keys() == expected.keys()
    for name, expected_tensor in expected.items():
        actual_tensor = actual[name]
        diff = _diff_tensors(actual_tensor, expected_tensor, KV_ATOL)
        assert diff is None, (
            f"KV state diverged for model={model_path}, prompt={prompt!r}, prefix_len={prefix_len}, state={name}\n"
            f"diff={diff}"
        )


@pytest.mark.parametrize("model_name,model_path", _default_model_cases())
def test_stateful_block_matches_autoregressive_for_known_tokens(model_name, model_path):
    _require_enabled()
    if not model_path.exists():
        pytest.skip(f"Model path is not available: {model_path}")
    compiled_model = _compile_model(model_path)
    max_new_tokens = int(os.environ.get(
        "OV_GENAI_BLOCK_VS_AR_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS))
    for prompt in PROMPTS:
        prompt_tokens = _encode_prompt(model_path, prompt)
        trace = _build_autoregressive_trace(
            compiled_model, prompt_tokens, max_new_tokens)
        prompt_len = len(prompt_tokens)
        for prefix_len in range(1, max_new_tokens + 1):
            block_request = compiled_model.create_infer_request()
            _infer_tokens(block_request, compiled_model,
                          prompt_tokens, 0, prompt_len)
            block_logits = _infer_tokens(
                block_request,
                compiled_model,
                trace.generated_tokens[:prefix_len],
                prompt_len,
                prompt_len + prefix_len,
            )
            for block_position in range(prefix_len):
                compared_prefix_len = block_position + 1
                _compare_logits(
                    block_logits[0, block_position],
                    trace.logits_by_prefix_len[compared_prefix_len],
                    prompt,
                    compared_prefix_len,
                    model_path,
                )
            _compare_states(
                _snapshot_state(block_request),
                trace.state_by_prefix_len[prefix_len],
                prompt,
                prefix_len,
                model_path,
            )
