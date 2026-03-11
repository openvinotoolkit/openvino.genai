import pytest
import logging
import sys
import os
from pathlib import Path

from conftest import convert_model, run_wwb
from test_cli_image import get_similarity


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_test(model_id, model_type, optimum_threshold, genai_threshold, tmp_path):
    if sys.platform == 'darwin':
        pytest.xfail("Ticket 173169")
    if sys.platform == "win32":
        pytest.xfail("Ticket 178790")

    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = convert_model(model_id)

    # Collect reference with HF model
    run_wwb([
        "--base-model",
        model_id,
        "--num-samples",
        "1",
        "--gt-data",
        GT_FILE,
        "--device",
        "CPU",
        "--model-type",
        model_type,
        "--hf",
    ])

    # test Optimum
    output = run_wwb([
        "--target-model",
        MODEL_PATH,
        "--num-samples",
        "1",
        "--gt-data",
        GT_FILE,
        "--device",
        "CPU",
        "--model-type",
        model_type,
    ])
    if optimum_threshold is not None:
        similarity = get_similarity(output)
        assert similarity >= optimum_threshold

    # test GenAI
    output = run_wwb([
        "--target-model",
        MODEL_PATH,
        "--num-samples",
        "1",
        "--gt-data",
        GT_FILE,
        "--device",
        "CPU",
        "--model-type",
        model_type,
        "--genai",
        "--output",
        tmp_path,
    ])
    if genai_threshold is not None:
        similarity = get_similarity(output)
        assert similarity >= genai_threshold

    # test w/o models
    run_wwb([
        "--target-data",
        tmp_path / "target.csv",
        "--num-samples",
        "1",
        "--gt-data",
        GT_FILE,
        "--device",
        "CPU",
        "--model-type",
        model_type,
        "--genai",
    ])


@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [
        ("optimum-intel-internal-testing/tiny-random-llava", "visual-text"),
    ],
)
def test_vlm_basic(model_id, model_type, tmp_path):
    run_test(model_id, model_type, None, None, tmp_path)


@pytest.mark.nanollava
@pytest.mark.parametrize(
    ("model_id", "model_type", "optimum_threshold", "genai_threshold"),
    [
        ("qnguyen3/nanoLLaVA", "visual-text", 0.99, 0.88),
    ],
)
def test_vlm_nanollava(model_id, model_type, optimum_threshold, genai_threshold, tmp_path):
    run_test(model_id, model_type, optimum_threshold, genai_threshold, tmp_path)


@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [
        ("optimum-intel-internal-testing/tiny-random-qwen2vl", "visual-video-text"),
        ("optimum-intel-internal-testing/tiny-random-llava-next-video", "visual-video-text"),
    ],
)
def test_vlm_video(model_id, model_type, tmp_path):
    run_test(model_id, model_type, 0.8, 0.8, tmp_path)


def _read_answers_from_target_csv(csv_path: Path) -> list[str]:
    import csv

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "answers" not in reader.fieldnames:
            raise AssertionError(f"Unexpected CSV schema in {csv_path}, fields: {reader.fieldnames}")
        return [row.get("answers", "") for row in reader]


def _split_csv_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",")]
    parts = [p for p in parts if p]
    return parts or None


def test_vlm_genai_lora_changes_output(tmp_path):
    """E2E: VLM GenAI output differs with vs without LoRA.

    This is intentionally opt-in because it requires local VLM + LoRA files.
    Enable via `WWB_VLM_LORA_E2E=true`.

    Optional env:
    - `WWB_VLM_MODEL_DIR`: OpenVINO VLM model directory
    - `WWB_VLM_LORA_PATH`: single LoRA adapter safetensors
    - `WWB_VLM_LORA_PATHS`: comma-separated adapter list (overrides `WWB_VLM_LORA_PATH`)
    - `WWB_VLM_LORA_ALPHAS`: comma-separated float list
    """

    if os.environ.get("WWB_VLM_LORA_E2E", "false").lower() != "true":
        pytest.skip("Set WWB_VLM_LORA_E2E=true to run this test")

    if sys.platform == "darwin":
        pytest.xfail("Ticket 173169")
    if sys.platform == "win32":
        pytest.xfail("Ticket 178790")

    repo_root = Path(__file__).resolve().parents[3]
    default_model_candidates = [
        repo_root / "vlm" / "Qwen2.5-VL-7B-Instruct",
        repo_root / "Qwen2.5-VL-7B-Instruct_FP32",
        repo_root / "Qwen3-VL-4B-Instruct_FP32",
    ]

    env_model_dir = os.environ.get("WWB_VLM_MODEL_DIR")
    if env_model_dir:
        model_dir = Path(env_model_dir)
    else:
        model_dir = next((p for p in default_model_candidates if p.exists()), None)

    if model_dir is None or not Path(model_dir).exists():
        pytest.skip("Missing VLM model dir; set WWB_VLM_MODEL_DIR")

    raw_paths = os.environ.get("WWB_VLM_LORA_PATHS")
    adapters = _split_csv_list(raw_paths)
    if adapters is None:
        single = os.environ.get("WWB_VLM_LORA_PATH")
        if single:
            adapters = [single]
        else:
            adapters = [str(repo_root / "vlm" / "qwen2.5-vl-lora-diagrams" / "adapter_model.safetensors")]

    adapter_paths = [Path(p) for p in adapters]
    missing = [p for p in adapter_paths if not p.exists()]
    if missing:
        pytest.skip(f"Missing LoRA adapter files: {missing}")

    raw_alphas = _split_csv_list(os.environ.get("WWB_VLM_LORA_ALPHAS"))
    if raw_alphas is None:
        alphas = [4.0] * len(adapter_paths)
    else:
        alphas = [float(v) for v in raw_alphas]
    if len(alphas) != len(adapter_paths):
        raise ValueError("WWB_VLM_LORA_ALPHAS must match WWB_VLM_LORA_PATHS length")

    gt_file = tmp_path / "gt.csv"
    out_no_lora = tmp_path / "out_no_lora"
    out_lora = tmp_path / "out_lora"
    out_no_lora.mkdir(parents=True, exist_ok=True)
    out_lora.mkdir(parents=True, exist_ok=True)

    # 1) Generate GT once (required by WWB CLI).
    run_wwb(
        [
            "--base-model",
            str(model_dir),
            "--num-samples",
            "1",
            "--gt-data",
            gt_file,
            "--device",
            "CPU",
            "--model-type",
            "visual-text",
            "--genai",
            "--max_new_tokens",
            "32",
        ]
    )

    # 2) Baseline generation (no LoRA) -> target.csv
    run_wwb(
        [
            "--target-model",
            str(model_dir),
            "--num-samples",
            "1",
            "--gt-data",
            gt_file,
            "--device",
            "CPU",
            "--model-type",
            "visual-text",
            "--genai",
            "--max_new_tokens",
            "32",
            "--output",
            out_no_lora,
        ]
    )

    # 3) LoRA generation -> target.csv
    args_with_lora = [
        "--target-model",
        str(model_dir),
        "--num-samples",
        "1",
        "--gt-data",
        gt_file,
        "--device",
        "CPU",
        "--model-type",
        "visual-text",
        "--genai",
        "--max_new_tokens",
        "32",
        "--output",
        out_lora,
        "--adapters",
        *[str(p) for p in adapter_paths],
        "--alphas",
        *[str(a) for a in alphas],
    ]
    run_wwb(args_with_lora)

    baseline_csv = out_no_lora / "target.csv"
    lora_csv = out_lora / "target.csv"
    assert baseline_csv.exists(), f"Missing baseline predictions: {baseline_csv}"
    assert lora_csv.exists(), f"Missing LoRA predictions: {lora_csv}"

    baseline_answers = _read_answers_from_target_csv(baseline_csv)
    lora_answers = _read_answers_from_target_csv(lora_csv)
    assert len(baseline_answers) == len(lora_answers)
    assert any(a != b for a, b in zip(baseline_answers, lora_answers)), (
        "Expected at least one answer to differ between baseline and LoRA runs; "
        "try a different adapter or increase alpha"
    )
