"""Tests for .github/scripts/build_test_matrices.py and the matrix definition files."""

import json
import sys
from pathlib import Path

import pytest
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import build_test_matrices as bm  # noqa: E402

MATRICES_DIR = bm.MATRICES_DIR
COMPONENTS_YML = SCRIPTS_DIR.parent / "components.yml"

PLATFORMS = sorted(path.name for path in MATRICES_DIR.iterdir() if path.is_dir())
MATRIX_FILES = sorted(MATRICES_DIR.glob("*/*.yml"))


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def matrix_file_id(path: Path) -> str:
    return str(path.relative_to(MATRICES_DIR))


def known_component_tokens() -> set[str]:
    """Component names declared in components.yml (keys and revalidate lists), lower-cased."""
    data = load_yaml(COMPONENTS_YML)
    tokens: set[str] = set()
    for key, value in data.items():
        tokens.add(key.lower())
        revalidate = value.get("revalidate")
        if isinstance(revalidate, list):
            tokens.update(token.lower() for token in revalidate)
    return tokens


def run_main(monkeypatch, tmp_path, platform: str, affected: dict) -> dict:
    output_file = tmp_path / "github_output.txt"
    monkeypatch.setenv("AFFECTED_COMPONENTS", json.dumps(affected))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setattr(sys, "argv", ["build_test_matrices.py", "--platform", platform])
    bm.main()

    outputs = {}
    for line in output_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, value = line.split("=", 1)
        outputs[name] = json.loads(value)
    return outputs


@pytest.mark.parametrize(
    "affected,expected",
    [
        ({"LLM": {"test": True}}, True),
        ({"LLM": {"test": False}}, False),
        ({"LLM": {}}, False),
        ({}, False),
        ({"LLM": True}, True),
        ({"LLM": False}, False),
    ],
)
def test_is_affected(affected, expected):
    assert bm.is_affected(affected, "LLM") == expected


def test_filter_matrix_keeps_matching_and_strips_components():
    entries = [
        {"name": "a", "cmd": "x", "timeout": 1, "components": ["LLM", "GGUF"]},
        {"name": "b", "cmd": "y", "timeout": 2, "components": ["RAG"]},
    ]
    result = bm.filter_matrix({"GGUF": {"test": True}}, entries)
    assert result == [{"name": "a", "cmd": "x", "timeout": 1}]


def test_filter_matrix_excludes_when_no_component_affected():
    entries = [{"name": "a", "cmd": "x", "components": ["LLM"]}]
    assert bm.filter_matrix({"RAG": {"test": True}}, entries) == []


@pytest.mark.parametrize("platform", PLATFORMS)
def test_main_output_names_match_file_stems_and_empty_input(monkeypatch, tmp_path, platform):
    outputs = run_main(monkeypatch, tmp_path, platform, {})
    expected_names = {path.stem for path in (MATRICES_DIR / platform).glob("*.yml")}
    assert set(outputs) == expected_names
    assert all(entries == [] for entries in outputs.values())


@pytest.mark.parametrize("platform", PLATFORMS)
def test_main_all_affected_includes_every_entry(monkeypatch, tmp_path, platform):
    files = list((MATRICES_DIR / platform).glob("*.yml"))
    all_components = {component for file in files for entry in load_yaml(file) for component in entry["components"]}
    affected = {component: {"test": True} for component in all_components}

    outputs = run_main(monkeypatch, tmp_path, platform, affected)

    for file in files:
        assert len(outputs[file.stem]) == len(load_yaml(file))
        assert all("components" not in entry for entry in outputs[file.stem])


@pytest.mark.parametrize("path", MATRIX_FILES, ids=matrix_file_id)
def test_matrix_file_entry_schema(path):
    entries = load_yaml(path)
    assert isinstance(entries, list) and entries, f"{path} must be a non-empty list"

    is_samples = path.stem == "samples_tests"
    for entry in entries:
        assert entry.get("name"), f"missing name in {path}"
        assert entry.get("cmd"), f"missing cmd in {entry['name']} ({path})"
        assert isinstance(entry.get("components"), list) and entry["components"], (
            f"missing components in {entry['name']} ({path})"
        )
        if is_samples:
            assert entry.get("marker"), f"missing marker in {entry['name']} ({path})"
        else:
            assert isinstance(entry.get("timeout"), int), f"missing timeout in {entry['name']} ({path})"


@pytest.mark.parametrize("path", MATRIX_FILES, ids=matrix_file_id)
def test_matrix_entry_names_are_unique(path):
    names = [entry["name"] for entry in load_yaml(path)]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("path", MATRIX_FILES, ids=matrix_file_id)
def test_matrix_components_are_declared_in_components_yml(path):
    known = known_component_tokens()
    for entry in load_yaml(path):
        for component in entry["components"]:
            assert component.lower() in known, f"Unknown component '{component}' in {matrix_file_id(path)}"
