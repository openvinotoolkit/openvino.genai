# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
import logging
import json
import sys

from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM, OVWeightQuantizationConfig

from test_cli_image import get_similarity
from conftest import convert_text_model, run_wwb


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model_id = "facebook/opt-125m"

# awq/gptq models are skipped for now: 180586
awq_model_id = "TitanML/tiny-mixtral-AWQ-4bit"
# model load failed - ticket: 178940, 180586
gptq_model_id = "ybelkada/opt-125m-gptq-4bit"


def _convert_base(model_id, temp_path):
    from optimum.exporters.openvino.convert import export_tokenizer
    from huggingface_hub import snapshot_download

    model_local = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_local)
    base_model = OVModelForCausalLM.from_pretrained(model_local)
    base_model.save_pretrained(temp_path)
    tokenizer.save_pretrained(temp_path)
    export_tokenizer(tokenizer, temp_path)


def _convert_int8(model_id, temp_path):
    from optimum.exporters.openvino.convert import export_tokenizer
    from huggingface_hub import snapshot_download

    model_local = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_local)
    target_model = OVModelForCausalLM.from_pretrained(
        model_local, quantization_config=OVWeightQuantizationConfig(bits=8)
    )
    target_model.save_pretrained(temp_path)
    tokenizer.save_pretrained(temp_path)
    export_tokenizer(tokenizer, temp_path)


base_model_path = convert_text_model(model_id, "opt125m", _convert_base)
target_model_path = convert_text_model(model_id, "opt125m_int8", _convert_int8)


@pytest.mark.skipif((sys.platform == "darwin"), reason='173169')
def test_text_target_model():
    run_wwb(
        [
            "--base-model",
            base_model_path,
            "--target-model",
            target_model_path,
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--model-type",
            "text",
            "--short-prompt",
        ]
    )


@pytest.fixture
def test_text_gt_data(tmp_path):
    temp_file_name = tmp_path / "gt.csv"
    run_wwb(
        [
            "--base-model",
            base_model_path,
            "--gt-data",
            temp_file_name,
            "--dataset",
            "EleutherAI/lambada_openai,en",
            "--dataset-field",
            "text",
            "--split",
            "test",
            "--num-samples",
            "2",
            "--device",
            "CPU",
        ]
    )
    data = pd.read_csv(temp_file_name)
    assert len(data["questions"].values) == 2


def test_text_output_directory(tmp_path):
    temp_file_name = tmp_path / "gt.csv"
    output = run_wwb(
        [
            "--base-model",
            base_model_path,
            "--gt-data",
            temp_file_name,
            "--target-model",
            target_model_path,
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--output",
            tmp_path,
            "--short-prompt",
        ]
    )
    assert "Metrics for model" in output
    assert (tmp_path / "metrics_per_question.csv").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "target.csv").exists()

    measurement_without_models = run_wwb([
        "--gt-data",
        temp_file_name,
        "--target-data",
        tmp_path / "target.csv",
        "--num-samples",
        "2",
        "--device",
        "CPU",
    ])
    assert "Metrics for model" in measurement_without_models


def test_text_verbose():
    output = run_wwb(
        [
            "--base-model",
            base_model_path,
            "--target-model",
            target_model_path,
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--short-prompt",
            "--verbose",
        ]
    )
    assert "## Diff:" in output


def test_text_language(tmp_path):
    temp_file_name = tmp_path / "gt.csv"
    run_wwb(
        [
            "--base-model",
            "Qwen/Qwen2-0.5B",
            "--gt-data",
            temp_file_name,
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--language",
            "cn",
            "--short-prompt",
        ]
    )
    data = pd.read_csv(temp_file_name)
    assert "马克" in data["prompts"].values[0]


@pytest.mark.parametrize(
    ("model_id"),
    [(model_id)],
)
def test_text_hf_model(model_id, tmp_path):
    temp_file_name = tmp_path / "gt.csv"
    run_wwb(
        [
            "--base-model",
            model_id,
            "--gt-data",
            temp_file_name,
            "--num-samples",
            "1",
            "--device",
            "CPU",
            "--short-prompt",
            "--hf",
        ]
    )
    data = pd.read_csv(temp_file_name)
    assert len(data["prompts"].values) == 1


def test_text_genai_model():
    output = run_wwb(
        [
            "--base-model",
            base_model_path,
            "--target-model",
            target_model_path,
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--genai",
            "--short-prompt",
        ]
    )
    assert "Metrics for model" in output
    assert "## Reference text" not in output


def test_text_genai_cb_model(tmp_path):
    if sys.platform == "darwin":
        pytest.xfail(
            "Continuous batching backend requires PagedAttention operation support, which is available on x86_64 or ARM64 platforms only"
        )
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        config = {
            "dynamic_split_fuse": True,
            "use_cache_eviction": True,
            "cache_eviction_config":
            {
                "start_size": 32,
                "recent_size": 32,
                "max_cache_size": 96
            }
        }
        json.dump(config, f)

    ov_config_path = tmp_path / "ov_config.json"
    with open(ov_config_path, "w") as f:
        config = {
            "KV_CACHE_PRECISION": "f16",
            "ATTENTION_BACKEND": "PA"
        }
        json.dump(config, f)

    output = run_wwb(
        [
            "--base-model",
            base_model_path,
            "--target-model",
            target_model_path,
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--genai",
            "--cb-config",
            config_path,
            "--ov-config",
            ov_config_path,
            "--short-prompt",
        ]
    )
    assert "Metrics for model" in output
    assert "## Reference text" not in output
    assert "INFO:whowhatbench.model_loaders:OpenVINO Config: {'KV_CACHE_PRECISION': 'f16', 'ATTENTION_BACKEND': 'PA'}" in output


def test_text_genai_json_string_config():
    if sys.platform == "darwin":
        pytest.xfail(
            "Continuous batching backend requires PagedAttention operation support, which is available on x86_64 or ARM64 platforms only"
        )

    cb_json_string = "{\"max_num_batched_tokens\": 4096}"
    ov_json_string = "{\"KV_CACHE_PRECISION\":\"f16\", \"ATTENTION_BACKEND\": \"PA\"}"

    output = run_wwb(
        [
            "--base-model",
            base_model_path,
            "--target-model",
            target_model_path,
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--genai",
            "--cb-config",
            cb_json_string,
            "--short-prompt",
            "--ov-config",
            ov_json_string,
        ]
    )

    # Test with WWB log info to make sure the configurations are passed from strings to the GenAI APIs
    assert "INFO:whowhatbench.wwb:cb_config: {'max_num_batched_tokens': 4096}" in output
    assert "INFO:whowhatbench.model_loaders:OpenVINO Config: {'KV_CACHE_PRECISION': 'f16', 'ATTENTION_BACKEND': 'PA'}" in output


sd_main_model_id = "xf2022/tiny-random-qwen3-layer10"
sd_draft_model_id = "xf2022/tiny-random-qwen3-eagle3"


def _convert_draft(model_id, temp_path):
    from optimum.exporters.openvino.convert import export_tokenizer
    from huggingface_hub import snapshot_download

    model_local = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_local, trust_remote_code=True)
    base_model = OVModelForCausalLM.from_pretrained(model_local, trust_remote_code=True, ignore_mismatched_sizes=True)
    base_model.save_pretrained(temp_path)
    tokenizer.save_pretrained(temp_path)
    export_tokenizer(tokenizer, temp_path)


@pytest.fixture(scope="module")
def sd_main_model_path():
    return convert_text_model(sd_main_model_id, "tiny-random-qwen3-layer10", _convert_base)


@pytest.fixture(scope="module")
def sd_draft_model_path():
    return convert_text_model(sd_draft_model_id, "tiny-random-qwen3-eagle3", _convert_draft)


SD_SIMILARITY_THRESHOLD = 0.9


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Speculative decoding requires PagedAttention, not supported on macOS"
)
def test_text_genai_sd_generation_config_json_string(tmp_path, sd_main_model_path, sd_draft_model_path):
    """Test --sd-generation-config with JSON string for speculative decoding."""
    temp_file_name = tmp_path / "gt.csv"
    sd_config_string = '{"num_assistant_tokens": 5}'

    run_wwb(
        [
            "--base-model",
            sd_main_model_path,
            "--gt-data",
            str(temp_file_name),
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--genai",
            "--short-prompt",
        ]
    )

    output = run_wwb(
        [
            "--target-model",
            sd_main_model_path,
            "--gt-data",
            str(temp_file_name),
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--genai",
            "--short-prompt",
            "--draft-model",
            sd_draft_model_path,
            "--sd-generation-config",
            sd_config_string,
        ]
    )
    assert "sd_generation_config:" in output
    assert "Metrics for model" in output
    similarity = get_similarity(output)
    assert similarity >= SD_SIMILARITY_THRESHOLD


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Speculative decoding requires PagedAttention, not supported on macOS"
)
def test_text_genai_sd_generation_config_json_file(tmp_path, sd_main_model_path, sd_draft_model_path):
    """Test --sd-generation-config with JSON file for speculative decoding."""
    temp_file_name = tmp_path / "gt.csv"
    config_path = tmp_path / "sd_config.json"
    with open(config_path, "w") as f:
        json.dump({"num_assistant_tokens": 5}, f)

    run_wwb(
        [
            "--base-model",
            sd_main_model_path,
            "--gt-data",
            str(temp_file_name),
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--genai",
            "--short-prompt",
        ]
    )

    output = run_wwb(
        [
            "--target-model",
            sd_main_model_path,
            "--gt-data",
            str(temp_file_name),
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--genai",
            "--short-prompt",
            "--draft-model",
            sd_draft_model_path,
            "--sd-generation-config",
            str(config_path),
        ]
    )
    assert "sd_generation_config:" in output
    assert "Metrics for model" in output
    similarity = get_similarity(output)
    assert similarity >= SD_SIMILARITY_THRESHOLD


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Speculative decoding requires PagedAttention, not supported on macOS"
)
def test_text_genai_sd_generation_config_unsupported_key(tmp_path, sd_main_model_path, sd_draft_model_path):
    """Test --sd-generation-config warns on unsupported keys."""
    temp_file_name = tmp_path / "gt.csv"
    sd_config_string = '{"num_assistant_tokens": 5, "unsupported_key": 99}'

    run_wwb(
        [
            "--base-model",
            sd_main_model_path,
            "--gt-data",
            str(temp_file_name),
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--genai",
            "--short-prompt",
        ]
    )

    output = run_wwb(
        [
            "--target-model",
            sd_main_model_path,
            "--gt-data",
            str(temp_file_name),
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--genai",
            "--short-prompt",
            "--draft-model",
            sd_draft_model_path,
            "--sd-generation-config",
            sd_config_string,
        ]
    )
    assert "not supported" in output


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Speculative decoding requires PagedAttention, not supported on macOS"
)
def test_text_genai_sd_generation_config_override(tmp_path, sd_main_model_path, sd_draft_model_path):
    """Test --sd-generation-config overrides --num-assistant-tokens from cmdline."""
    temp_file_name = tmp_path / "gt.csv"
    sd_config_string = '{"num_assistant_tokens": 7}'

    run_wwb(
        [
            "--base-model",
            sd_main_model_path,
            "--gt-data",
            str(temp_file_name),
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--genai",
            "--short-prompt",
        ]
    )

    output = run_wwb(
        [
            "--target-model",
            sd_main_model_path,
            "--gt-data",
            str(temp_file_name),
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--genai",
            "--short-prompt",
            "--draft-model",
            sd_draft_model_path,
            "--num-assistant-tokens",
            "3",
            "--sd-generation-config",
            sd_config_string,
        ]
    )
    # JSON should override cmdline: num_assistant_tokens=7 takes precedence over --num-assistant-tokens=3
    assert "num_assistant_tokens (final): 7" in output
    assert "Metrics for model" in output
    similarity = get_similarity(output)
    assert similarity >= SD_SIMILARITY_THRESHOLD


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Speculative decoding requires PagedAttention, not supported on macOS"
)
def test_text_genai_sd_generation_config_topk(tmp_path, sd_main_model_path, sd_draft_model_path):
    """Test --sd-generation-config with EAGLE3 Top-K parameters."""
    temp_file_name = tmp_path / "gt.csv"
    sd_config_string = '{"num_assistant_tokens": 6, "branching_factor": 3, "tree_depth": 2}'

    run_wwb(
        [
            "--base-model",
            sd_main_model_path,
            "--gt-data",
            str(temp_file_name),
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--genai",
            "--short-prompt",
        ]
    )

    output = run_wwb(
        [
            "--target-model",
            sd_main_model_path,
            "--gt-data",
            str(temp_file_name),
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--genai",
            "--short-prompt",
            "--draft-model",
            sd_draft_model_path,
            "--sd-generation-config",
            sd_config_string,
        ]
    )
    assert "sd_generation_config:" in output
    assert "Metrics for model" in output
    similarity = get_similarity(output)
    assert similarity >= SD_SIMILARITY_THRESHOLD


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Speculative decoding requires PagedAttention, not supported on macOS"
)
def test_text_genai_sd_generation_config_topk_file(tmp_path, sd_main_model_path, sd_draft_model_path):
    """Test --sd-generation-config Top-K via JSON file."""
    temp_file_name = tmp_path / "gt.csv"
    config_path = tmp_path / "sd_topk_config.json"
    with open(config_path, "w") as f:
        json.dump({"num_assistant_tokens": 6, "branching_factor": 3, "tree_depth": 2}, f)

    run_wwb(
        [
            "--base-model",
            sd_main_model_path,
            "--gt-data",
            str(temp_file_name),
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--genai",
            "--short-prompt",
        ]
    )

    output = run_wwb(
        [
            "--target-model",
            sd_main_model_path,
            "--gt-data",
            str(temp_file_name),
            "--num-samples",
            "2",
            "--device",
            "CPU",
            "--genai",
            "--short-prompt",
            "--draft-model",
            sd_draft_model_path,
            "--sd-generation-config",
            str(config_path),
        ]
    )
    assert "sd_generation_config:" in output
    assert "Metrics for model" in output
    similarity = get_similarity(output)
    assert similarity >= SD_SIMILARITY_THRESHOLD


@pytest.mark.parametrize(
    ("model_id"),
    [("optimum-intel-internal-testing/tiny-random-Phi3ForCausalLM")],
)
def test_text_chat_model(model_id, tmp_path):
    if sys.platform == "darwin":
        pytest.xfail("Ticket 183495")

    SIMILARITY_THRESHOLD = 0.9
    temp_file_name = tmp_path / "gt.csv"
    chat_model_path = convert_text_model(model_id, model_id.split("/")[1], _convert_base)

    run_wwb(
        [
            "--base-model",
            model_id,
            "--gt-data",
            temp_file_name,
            "--num-samples",
            "1",
            "--device",
            "CPU",
            "--model-type",
            "text-chat",
            "--hf",
            "--max_new_tokens",
            "10",
        ]
    )

    outputs_path = tmp_path / "optimum"
    output = run_wwb(
        [
            "--target-model",
            chat_model_path,
            "--gt-data",
            temp_file_name,
            "--num-samples",
            "1",
            "--device",
            "CPU",
            "--model-type",
            "text-chat",
            "--output",
            outputs_path,
            "--max_new_tokens",
            "10",
        ]
    )
    assert "Metrics for model" in output
    assert (outputs_path / "metrics_per_question.csv").exists()
    assert (outputs_path / "metrics.csv").exists()
    assert (outputs_path / "target.csv").exists()

    similarity = get_similarity(output)
    assert similarity >= SIMILARITY_THRESHOLD

    outputs_path = tmp_path / "genai"
    output = run_wwb(
        [
            "--target-model",
            chat_model_path,
            "--gt-data",
            temp_file_name,
            "--num-samples",
            "1",
            "--device",
            "CPU",
            "--model-type",
            "text-chat",
            "--genai",
            "--output",
            outputs_path,
            "--max_new_tokens",
            "10",
        ]
    )
    assert "Metrics for model" in output
    assert (outputs_path / "metrics_per_question.csv").exists()
    assert (outputs_path / "metrics.csv").exists()
    assert (outputs_path / "target.csv").exists()

    similarity = get_similarity(output)
    assert similarity >= SIMILARITY_THRESHOLD
