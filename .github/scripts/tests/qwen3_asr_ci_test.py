"""Verify Qwen3-ASR test selection in the CI matrices."""

import sys
from pathlib import Path

import pytest
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import build_test_matrices as matrices  # noqa: E402


@pytest.mark.parametrize("platform", ["mac", "windows"])
@pytest.mark.parametrize("component", ["llm_bench", "automatic_speech_recognition"])
def test_qwen_benchmark_selected_independently(platform, component):
    path = matrices.MATRICES_DIR / platform / "wheel_tests.yml"
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))
    selected = matrices.filter_matrix({component: {"test": True}}, entries)
    test = next(entry for entry in selected if entry["name"] == "LLM benchmark Qwen3-ASR")
    assert "samples and llm_bench_qwen3_asr" in test["cmd"]


@pytest.mark.parametrize("platform", ["linux", "manylinux", "windows"])
def test_qwen_npu_tests_run_on_supported_platforms(platform):
    path = matrices.MATRICES_DIR / platform / "wheel_tests.yml"
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))
    selected = matrices.filter_matrix({"automatic_speech_recognition": {"test": True}}, entries)
    test = next(entry for entry in selected if "test_asr_pipeline_npu.py" in entry["cmd"])
    assert 'python -c "import qwen_asr"' in test["cmd"]


def test_linux_qwen_benchmark_triggers_for_either_component():
    path = SCRIPTS_DIR.parent / "workflows" / "linux.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["genai_tools_tests"]["steps"]
    step = next(item for item in steps if item.get("name") == "Test LLM Benchmark Tools with Qwen3-ASR model")
    assert step["if"] == (
        "${{ fromJSON(needs.smart_ci.outputs.affected_components).llm_bench.test || "
        "fromJSON(needs.smart_ci.outputs.affected_components).automatic_speech_recognition.test }}"
    )
    assert "test_llm_bench_speech_to_text.py" not in step["run"]
    assert "pip install qwen-asr==0.0.6" in step["run"]


def test_linux_cpp_tests_include_asr_and_shared_capacity():
    path = SCRIPTS_DIR.parent / "workflows" / "linux.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["genai_tools_tests"]["steps"]
    step = next(item for item in steps if item.get("name") == "Qwen3-ASR and LM capacity gtests")
    assert "*Qwen3ASR*:LMEncodingCapacityTest.*" in step["run"]
    for component in ("automatic_speech_recognition", "LLM", "visual_language"):
        assert f".{component}.test" in step["if"]
    assert "!fromJSON(needs.smart_ci.outputs.affected_components).continuous_batching" in step["if"]
