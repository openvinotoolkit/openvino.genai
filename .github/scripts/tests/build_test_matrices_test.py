"""Tests for .github/scripts/build_test_matrices.py and the matrix definition files."""

import json
import re
import sys
from fnmatch import fnmatchcase
from pathlib import Path

import pytest
import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

import build_test_matrices as bm  # noqa: E402

MATRICES_DIR = bm.MATRICES_DIR
COMPONENTS_YML = SCRIPTS_DIR.parent / "components.yml"
LABELER_YML = SCRIPTS_DIR.parent / "labeler.yml"

PLATFORMS = sorted(path.name for path in MATRICES_DIR.iterdir() if path.is_dir())
MATRIX_FILES = sorted(MATRICES_DIR.glob("*/*.yml"))


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def matrix_file_id(path: Path) -> str:
    return str(path.relative_to(MATRICES_DIR))


def known_component_tokens() -> set[str]:
    """Component names declared in components.yml (keys and revalidate lists)."""
    data = load_yaml(COMPONENTS_YML)
    tokens: set[str] = set()
    for key, value in data.items():
        tokens.add(key)
        revalidate = value.get("revalidate")
        if isinstance(revalidate, list):
            tokens.update(revalidate)
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


@pytest.mark.parametrize(
    "files,expected",
    [
        (["tests/python_tests/gguf_reference/CMakeLists.txt"], False),
        (["tests/python_tests/gguf_reference/generate.cpp"], False),
        (["CMakeLists.txt"], True),
        (["src/cpp/CMakeLists.txt"], True),
        (["tests/cpp/CMakeLists.txt"], True),
        (["cmake/features.cmake"], True),
        (["src/python/compare_pyi.cmake"], True),
        (["requirements-build.txt"], True),
        (["pyproject.toml"], True),
        (["tests/python_tests/gguf_reference/CMakeLists.txt", "src/cpp/CMakeLists.txt"], True),
    ],
)
def test_gguf_reference_build_does_not_trigger_global_build(files, expected):
    rules = load_yaml(LABELER_YML)["category: build"]

    def matches(path, pattern):
        if pattern.startswith("!"):
            return not matches(path, pattern[1:])
        return fnmatchcase(path, pattern) or (
            pattern.startswith("**/") and fnmatchcase(path, pattern.removeprefix("**/"))
        )

    # Labeler v4's "any" requires all patterns to match at least one changed file.
    assert (
        any(
            all(matches(path, pattern) for pattern in ([rule] if isinstance(rule, str) else rule["any"]))
            for rule in rules
            for path in files
        )
        == expected
    )


@pytest.mark.parametrize("platform", PLATFORMS)
def test_gguf_only_selects_existing_gguf_jobs(monkeypatch, tmp_path, platform):
    outputs = run_main(monkeypatch, tmp_path, platform, {"GGUF": {"build": True, "test": True}})
    expected_names = {"GGUF Reader tests"}
    if platform in {"linux", "manylinux"}:
        expected_names.add("GGUF Frontend tiny models")
    assert {entry["name"] for entry in outputs["wheel_tests"]} == expected_names
    assert outputs.get("samples_tests", []) == []


def test_gguf_gtest_filter_covers_gguf_test_files():
    gguf_gtest_filter = "*GGUF*"
    suites = set()
    for path in (SCRIPTS_DIR.parents[1] / "tests" / "cpp").glob("test_gguf_*.cpp"):
        text = path.read_text(encoding="utf-8")
        suites.update(re.findall(r"^TEST(?:_F|_P)?\((\w+),", text, re.MULTILINE))
        suites.update(re.findall(r"^INSTANTIATE_TEST_SUITE_P\((\w+),", text, re.MULTILINE))
    assert suites
    assert all(fnmatchcase(suite, gguf_gtest_filter) for suite in suites)


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
            assert component in known, f"Unknown component '{component}' in {matrix_file_id(path)}"
