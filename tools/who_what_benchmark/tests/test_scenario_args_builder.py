# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RED-phase tests for ``whowhatbench.scenario.args_builder``.

The module under test does not exist yet — these tests are expected to fail
(import errors / attribute errors) until the production code is implemented.
The assertions encode the contract from the plan.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
from pathlib import Path
from typing import Any

from whowhatbench.scenario import args_builder as args_builder_module
from whowhatbench.scenario.args_builder import build_args_namespace
from whowhatbench.scenario.schema import Scenario

# Full set of attributes produced by ``whowhatbench.wwb.parse_args`` — every
# one of these MUST appear on the namespace returned by ``build_args_namespace``
# so the runner can hand it to the existing wwb plumbing without surprises.
EXPECTED_ATTRS: set[str] = {
    "base_model",
    "target_model",
    "tokenizer",
    "omit_chat_template",
    "gt_data",
    "target_data",
    "model_type",
    "data_encoder",
    "dataset",
    "dataset_field",
    "split",
    "output",
    "num_samples",
    "verbose",
    "device",
    "ov_config",
    "language",
    "hf",
    "genai",
    "cb_config",
    "llamacpp",
    "from_onnx",
    "image_size",
    "num_inference_steps",
    "seed",
    "taylorseer_config",
    "adapters",
    "alphas",
    "long_prompt",
    "empty_adapters",
    "embeds_pooling_type",
    "embeds_normalize",
    "embeds_padding_side",
    "embeds_batch_size",
    "rag_config",
    "gguf_file",
    "draft_model",
    "draft_device",
    "draft_cb_config",
    "num_assistant_tokens",
    "assistant_confidence_threshold",
    "video_frames_num",
    "speaker_embeddings",
    "tts_eval_whisper_model",
    "vocoder_path",
    "pruning_ratio",
    "relevance_weight",
    "max_new_tokens",
}


def make_scenario(**overrides: Any) -> Scenario:
    """Build a minimal valid Scenario, with optional top-level overrides."""
    data: dict[str, Any] = {
        "schema_version": 1,
        "name": "test-scenario",
        "models": {
            "base": {"path": "org/base", "backend": "hf"},
            "target_genai": {"path": "/ov/target", "backend": "genai", "device": "CPU"},
            "target_gpu": {"path": "/ov/target", "backend": "genai", "device": "GPU.0"},
        },
        "datasets": {
            "ds_builtin": {"type": "builtin"},
            "ds_hf": {
                "type": "huggingface",
                "path": "squad",
                "split": "validation[:32]",
                "field": "question",
            },
            "ds_inline": {"type": "inline", "prompts": ["hello", "world"]},
        },
        "tasks": [
            {
                "id": "t1",
                "type": "text",
                "base": "base",
                "targets": ["target_genai"],
                "dataset": "ds_builtin",
            }
        ],
    }
    data.update(overrides)
    return Scenario.model_validate(data)


def _first_task(scenario: Scenario) -> Any:
    return scenario.tasks[0]


def _build(
    scenario: Scenario,
    target_id: str = "target_genai",
    output_dir: Path | None = None,
    gt_data_path: str | None = None,
) -> argparse.Namespace:
    out = output_dir if output_dir is not None else Path("/tmp/wwb_out")
    return build_args_namespace(
        scenario=scenario,
        task=_first_task(scenario),
        target_id=target_id,
        output_dir=out,
        gt_data_path=gt_data_path,
    )


def test_all_parse_args_attributes_covered(tmp_path: Path) -> None:
    """Drift detector — every attribute parse_args produces must be present."""
    scenario = make_scenario()
    ns = _build(scenario, output_dir=tmp_path)
    actual = set(vars(ns).keys())
    missing = EXPECTED_ATTRS - actual
    assert not missing, f"build_args_namespace is missing attributes: {sorted(missing)}"


def test_hf_backend_sets_correct_flags(tmp_path: Path) -> None:
    scenario = make_scenario()
    # Make the target an hf-backed model.
    scenario.models["target_genai"].backend = "hf"
    ns = _build(scenario, output_dir=tmp_path)
    assert ns.hf is True
    assert ns.genai is False
    assert ns.llamacpp is False
    assert ns.from_onnx is False


def test_genai_backend_sets_correct_flags(tmp_path: Path) -> None:
    scenario = make_scenario()
    scenario.models["target_genai"].backend = "genai"
    ns = _build(scenario, output_dir=tmp_path)
    assert ns.genai is True
    assert ns.hf is False
    assert ns.llamacpp is False
    assert ns.from_onnx is False


def test_llamacpp_backend_sets_correct_flags(tmp_path: Path) -> None:
    scenario = make_scenario()
    scenario.models["target_genai"].backend = "llamacpp"
    ns = _build(scenario, output_dir=tmp_path)
    assert ns.llamacpp is True
    assert ns.hf is False
    assert ns.genai is False
    assert ns.from_onnx is False


def test_onnx_backend_sets_correct_flags(tmp_path: Path) -> None:
    scenario = make_scenario()
    scenario.models["target_genai"].backend = "onnx"
    ns = _build(scenario, output_dir=tmp_path)
    assert ns.from_onnx is True
    assert ns.hf is False
    assert ns.genai is False
    assert ns.llamacpp is False


def test_ov_config_dict_serialized_to_json_string(tmp_path: Path) -> None:
    scenario = make_scenario()
    scenario.models["target_genai"].ov_config = {"CACHE_DIR": "/tmp"}
    ns = _build(scenario, output_dir=tmp_path)
    assert isinstance(ns.ov_config, str)
    assert json.loads(ns.ov_config) == {"CACHE_DIR": "/tmp"}


def test_ov_config_none_stays_none(tmp_path: Path) -> None:
    scenario = make_scenario()
    # Confirm precondition — the helper does not set ov_config by default.
    assert getattr(scenario.models["target_genai"], "ov_config", None) is None
    ns = _build(scenario, output_dir=tmp_path)
    assert ns.ov_config is None


def test_target_model_path(tmp_path: Path) -> None:
    scenario = make_scenario()
    ns = _build(scenario, output_dir=tmp_path)
    assert ns.target_model == scenario.models["target_genai"].path


def test_base_model_path_set(tmp_path: Path) -> None:
    scenario = make_scenario()
    ns = _build(scenario, output_dir=tmp_path)
    assert ns.base_model == scenario.models["base"].path


def test_device_from_target_model(tmp_path: Path) -> None:
    scenario = make_scenario(
        tasks=[
            {
                "id": "t1",
                "type": "text",
                "base": "base",
                "targets": ["target_gpu"],
                "dataset": "ds_builtin",
            }
        ]
    )
    ns = _build(scenario, target_id="target_gpu", output_dir=tmp_path)
    assert ns.device == "GPU.0"


def test_gt_data_path_propagated(tmp_path: Path) -> None:
    scenario = make_scenario()
    gt_path = "/some/cache/abcdef.csv"
    ns = _build(scenario, output_dir=tmp_path, gt_data_path=gt_path)
    assert ns.gt_data == gt_path


def test_model_type_propagated(tmp_path: Path) -> None:
    scenario = make_scenario(
        tasks=[
            {
                "id": "t1",
                "type": "text-chat",
                "base": "base",
                "targets": ["target_genai"],
                "dataset": "ds_builtin",
            }
        ]
    )
    ns = _build(scenario, output_dir=tmp_path)
    assert ns.model_type == "text-chat"


def test_max_new_tokens_propagated(tmp_path: Path) -> None:
    scenario = make_scenario(
        tasks=[
            {
                "id": "t1",
                "type": "text",
                "base": "base",
                "targets": ["target_genai"],
                "dataset": "ds_builtin",
                "generation": {"max_new_tokens": 256},
            }
        ]
    )
    ns = _build(scenario, output_dir=tmp_path)
    assert ns.max_new_tokens == 256


def test_builtin_dataset_clears_dataset_arg(tmp_path: Path) -> None:
    scenario = make_scenario()
    ns = _build(scenario, output_dir=tmp_path)
    assert ns.dataset is None


def test_hf_dataset_sets_dataset_arg(tmp_path: Path) -> None:
    scenario = make_scenario(
        tasks=[
            {
                "id": "t1",
                "type": "text",
                "base": "base",
                "targets": ["target_genai"],
                "dataset": "ds_hf",
            }
        ]
    )
    ns = _build(scenario, output_dir=tmp_path)
    assert ns.dataset == "squad"
    assert ns.split == "validation[:32]"
    assert ns.dataset_field == "question"


def test_hf_dataset_with_name(tmp_path: Path) -> None:
    scenario = make_scenario(
        datasets={
            "ds_builtin": {"type": "builtin"},
            "ds_named": {
                "type": "huggingface",
                "path": "wikitext",
                "name": "wikitext-2-v1",
            },
        },
        tasks=[
            {
                "id": "t1",
                "type": "text",
                "base": "base",
                "targets": ["target_genai"],
                "dataset": "ds_named",
            }
        ],
    )
    ns = _build(scenario, output_dir=tmp_path)
    assert ns.dataset == "wikitext,wikitext-2-v1"


def test_embedding_params_propagated(tmp_path: Path) -> None:
    scenario = make_scenario(
        tasks=[
            {
                "id": "t1",
                "type": "text-embedding",
                "base": "base",
                "targets": ["target_genai"],
                "dataset": "ds_builtin",
                "embeddings": {
                    "pooling_type": "mean",
                    "normalize": True,
                    "padding_side": "right",
                    "batch_size": 8,
                },
            }
        ]
    )
    ns = _build(scenario, output_dir=tmp_path)
    assert ns.embeds_pooling_type == "mean"
    assert ns.embeds_normalize is True
    assert ns.embeds_padding_side == "right"
    assert ns.embeds_batch_size == 8


def test_speech_params_propagated(tmp_path: Path) -> None:
    scenario = make_scenario(
        tasks=[
            {
                "id": "t1",
                "type": "speech-generation",
                "base": "base",
                "targets": ["target_genai"],
                "dataset": "ds_builtin",
                "speech": {
                    "speaker_embeddings": "/path.bin",
                    "whisper_model": "openai/whisper-small",
                    "vocoder_path": "/v",
                },
            }
        ]
    )
    ns = _build(scenario, output_dir=tmp_path)
    assert ns.speaker_embeddings == "/path.bin"
    assert ns.tts_eval_whisper_model == "openai/whisper-small"
    assert ns.vocoder_path == "/v"


def test_draft_model_propagated(tmp_path: Path) -> None:
    scenario = make_scenario(
        models={
            "base": {"path": "org/base", "backend": "hf"},
            "target_genai": {
                "path": "/ov/target",
                "backend": "genai",
                "device": "CPU",
                "draft": {
                    "path": "/d/model",
                    "device": "CPU",
                    "cb_config": {"cache_size": 1},
                },
            },
            "target_gpu": {"path": "/ov/target", "backend": "genai", "device": "GPU.0"},
        }
    )
    ns = _build(scenario, output_dir=tmp_path)
    assert ns.draft_model == "/d/model"
    assert ns.draft_device == "CPU"
    assert isinstance(ns.draft_cb_config, str)
    assert json.loads(ns.draft_cb_config) == {"cache_size": 1}


def test_verbose_always_false(tmp_path: Path) -> None:
    scenario = make_scenario()
    ns = _build(scenario, output_dir=tmp_path)
    assert ns.verbose is False


# Alias so the new tests below match the naming convention used in the prompt.
_make_scenario = make_scenario


def test_target_data_always_none() -> None:
    """args.target_data must always be None — runner handles predictions separately."""
    scenario = _make_scenario()
    task = scenario.tasks[0]
    ns = build_args_namespace(scenario, task, "target_genai", Path("/out"), None)
    assert ns.target_data is None


def test_cb_config_dict_serialized_to_json_string() -> None:
    """cb_config dict must be serialized to a JSON string, same as ov_config."""
    scenario = Scenario.model_validate(
        {
            "schema_version": 1,
            "name": "cb-test",
            "models": {
                "base": {"path": "org/base", "backend": "hf"},
                "target_cb": {
                    "path": "/ov/cb",
                    "backend": "genai",
                    "cb_config": {"cache_size": 2, "num_batched_tokens": 256},
                },
            },
            "datasets": {"ds": {"type": "builtin"}},
            "tasks": [
                {
                    "id": "t1",
                    "type": "text",
                    "base": "base",
                    "targets": ["target_cb"],
                    "dataset": "ds",
                }
            ],
        }
    )
    task = scenario.tasks[0]
    ns = build_args_namespace(scenario, task, "target_cb", Path("/out"), None)
    assert ns.cb_config is not None
    parsed = json.loads(ns.cb_config)
    assert parsed["cache_size"] == 2
    assert parsed["num_batched_tokens"] == 256


def test_cb_config_none_stays_none() -> None:
    """cb_config=None must produce args.cb_config=None."""
    scenario = _make_scenario()
    task = scenario.tasks[0]
    ns = build_args_namespace(scenario, task, "target_genai", Path("/out"), None)
    assert ns.cb_config is None


def test_inline_dataset_clears_dataset_arg() -> None:
    """Inline datasets must set args.dataset=None (handled by runner as test_data)."""
    scenario = _make_scenario(
        tasks=[
            {
                "id": "t1",
                "type": "text",
                "base": "base",
                "targets": ["target_genai"],
                "dataset": "ds_inline",
            }
        ]
    )
    task = scenario.tasks[0]
    ns = build_args_namespace(scenario, task, "target_genai", Path("/out"), None)
    assert ns.dataset is None, f"Expected args.dataset=None for inline dataset, got {ns.dataset!r}"


def test_vlm_params_propagated() -> None:
    """VlmParams must propagate to the correct Namespace attributes."""
    from whowhatbench.scenario.schema import TaskConfig

    scenario = _make_scenario()
    task_data = {
        "id": "vlm_task",
        "type": "visual-text",
        "base": "base",
        "targets": ["target_genai"],
        "dataset": "ds_builtin",
        "vlm": {
            "video_frames_num": 8,
            "pruning_ratio": 25,
            "relevance_weight": 0.7,
            "omit_chat_template": True,
        },
    }
    task = TaskConfig.model_validate(task_data)
    ns = build_args_namespace(scenario, task, "target_genai", Path("/out"), None)
    assert ns.video_frames_num == 8
    assert ns.pruning_ratio == 25
    assert abs(ns.relevance_weight - 0.7) < 1e-9
    assert ns.omit_chat_template is True


def test_chat_params_propagated() -> None:
    """ChatParams must propagate to the correct Namespace attributes."""
    from whowhatbench.scenario.schema import TaskConfig

    scenario = _make_scenario()
    task_data = {
        "id": "chat_task",
        "type": "text-chat",
        "base": "base",
        "targets": ["target_genai"],
        "dataset": "ds_builtin",
        "chat": {
            "omit_chat_template": True,
            "num_assistant_tokens": 5,
            "assistant_confidence_threshold": 0.9,
            "empty_adapters": True,
        },
    }
    task = TaskConfig.model_validate(task_data)
    ns = build_args_namespace(scenario, task, "target_genai", Path("/out"), None)
    assert ns.omit_chat_template is True
    assert ns.num_assistant_tokens == 5
    assert abs(ns.assistant_confidence_threshold - 0.9) < 1e-9
    assert ns.empty_adapters is True


def test_generation_image_params_propagated() -> None:
    """Image/video generation params must map to the correct Namespace attributes."""
    from whowhatbench.scenario.schema import TaskConfig

    scenario = _make_scenario()
    task_data = {
        "id": "t2i_task",
        "type": "text-to-image",
        "base": "base",
        "targets": ["target_genai"],
        "dataset": "ds_builtin",
        "generation": {
            "num_inference_steps": 20,
            "image_size": 512,
            "seed": 7,
        },
    }
    task = TaskConfig.model_validate(task_data)
    ns = build_args_namespace(scenario, task, "target_genai", Path("/out"), None)
    assert ns.num_inference_steps == 20
    assert ns.image_size == 512
    assert ns.seed == 7


def test_csv_dataset_clears_dataset_arg() -> None:
    """CSV datasets (loaded as test_data by runner) must also set args.dataset=None."""
    scenario = Scenario.model_validate(
        {
            "schema_version": 1,
            "name": "csv-test",
            "models": {
                "base": {"path": "org/base", "backend": "hf"},
                "target": {"path": "/ov/t", "backend": "genai"},
            },
            "datasets": {
                "ds_csv": {"type": "csv", "path": "/data/prompts.csv", "field": "text"},
            },
            "tasks": [
                {
                    "id": "t1",
                    "type": "text",
                    "base": "base",
                    "targets": ["target"],
                    "dataset": "ds_csv",
                }
            ],
        }
    )
    task = scenario.tasks[0]
    ns = build_args_namespace(scenario, task, "target", Path("/out"), None)
    assert ns.dataset is None


def _parse_drift_sentinel(source: str) -> set[str]:
    """Extract attribute names from the DRIFT SENTINEL block in args_builder.py.

    The sentinel is a comment block that lists every attribute the namespace
    must expose. Parsing it here (instead of duplicating the list in the test)
    is what makes this an enforcement mechanism: if the comment and the code
    drift apart, the test fails.
    """
    sentinel_match = re.search(
        r"#\s*DRIFT SENTINEL.*?\n(?P<body>(?:#.*\n)+)",
        source,
    )
    assert sentinel_match is not None, "DRIFT SENTINEL block not found in args_builder.py"
    body = sentinel_match.group("body")

    # Strip the lead-in line ("Attributes that must be set:") if present, then
    # collect every comma-separated identifier across the remaining comment lines.
    body = re.sub(r"Attributes that must be set:\s*", "", body)
    text = " ".join(line.lstrip("#").strip() for line in body.splitlines())
    return {token.strip() for token in text.split(",") if re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", token.strip())}


def test_build_args_namespace_has_all_required_attributes(tmp_path: Path) -> None:
    """Drift sentinel enforcement — every attribute named in the sentinel comment
    must actually be present on the Namespace returned by build_args_namespace.

    The list is parsed live from args_builder.py rather than duplicated here, so
    any change to the sentinel is immediately reflected in the test's expectation.
    Catches the case where parse_args() in wwb.py or the schema gains a new field
    that the sentinel acknowledges but build_args_namespace forgets to set.
    """
    source = inspect.getsource(args_builder_module)
    sentinel_attrs = _parse_drift_sentinel(source)

    # Sanity: the sentinel itself must list a non-trivial number of attributes,
    # otherwise the regex parsed nothing and the test would silently pass.
    assert len(sentinel_attrs) >= 40, f"Parsed sentinel only yielded {len(sentinel_attrs)} attrs — regex likely broken"

    scenario = make_scenario()
    ns = _build(scenario, output_dir=tmp_path)
    actual_attrs = set(vars(ns).keys())

    missing = sentinel_attrs - actual_attrs
    assert not missing, (
        f"Namespace is missing attributes listed in the DRIFT SENTINEL: {sorted(missing)}. "
        f"Either build_args_namespace() must set them or the sentinel comment must be updated."
    )
