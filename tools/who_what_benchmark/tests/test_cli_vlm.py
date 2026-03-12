import pytest
import logging
from pathlib import Path
import subprocess  # nosec B404
import sys

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


def _download_hf_files_to_cache(repo_id: str, cache_dir: Path, filenames: list[str]):
    from ov_utils import AtomicDownloadManager, retry_request

    dest_dir = Path(cache_dir)

    def download_to_local_dir(local_dir: Path) -> None:
        for filename in filenames:
            command = [
                "huggingface-cli",
                "download",
                repo_id,
                filename,
                "--local-dir",
                str(local_dir),
            ]

            def _run_download() -> None:
                subprocess.run(command, check=True, text=True, capture_output=True)

            retry_request(_run_download)

    # If destination exists (e.g. shared CI cache), make sure all required files are present.
    # This test previously cached only adapter_model.safetensors, but peft also requires
    # adapter_config.json next to it.
    if dest_dir.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
        missing = [name for name in filenames if not (dest_dir / name).exists()]
        if missing:
            import shutil
            import uuid

            temp_dir = dest_dir.parent / f".tmp_{dest_dir.name}_{uuid.uuid4().hex[:8]}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            try:
                download_to_local_dir(temp_dir)
                for filename in missing:
                    src = temp_dir / filename
                    if not src.exists():
                        raise AssertionError(f"Download failed: {src}")
                    dst = dest_dir / filename
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    src.replace(dst)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        manager = AtomicDownloadManager(dest_dir)
        manager.execute(download_to_local_dir)

    for filename in filenames:
        downloaded = dest_dir / filename
        if not downloaded.exists():
            raise AssertionError(f"Download failed: {downloaded}")

    return dest_dir


def run_test_with_lora(
    model_id: str,
    model_type: str,
    lora_repo_id: str,
    lora_cache_subdir: str,
    hf_alpha: float,
    genai_alpha: float,
    tmp_path,
    *,
    genai_threshold: float,
):
    if sys.platform == "darwin":
        pytest.xfail("Ticket 173169")
    if sys.platform == "win32":
        pytest.xfail("Ticket 178790")

    gt_file = tmp_path / "gt.csv"
    model_path = convert_model(model_id)

    from ov_utils import get_ov_cache_dir

    lora_filenames = ["adapter_model.safetensors", "adapter_config.json"]
    lora_cache_dir = get_ov_cache_dir() / "test_data" / lora_cache_subdir
    lora_adapter_dir = _download_hf_files_to_cache(lora_repo_id, lora_cache_dir, lora_filenames)
    lora_adapter_file = lora_adapter_dir / "adapter_model.safetensors"
    assert lora_adapter_file.exists(), f"LoRA adapter wasn't downloaded: {lora_adapter_file}"

    # 1) Generate GT using HF + LoRA.
    run_wwb(
        [
            "--base-model",
            model_id,
            "--num-samples",
            "1",
            "--gt-data",
            gt_file,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--hf",
            "--adapters",
            str(lora_adapter_dir),
            "--alphas",
            str(hf_alpha),
            "--max_new_tokens",
            "32",
        ]
    )
    assert gt_file.exists(), f"GT wasn't generated: {gt_file}"

    # 2) Target: GenAI + LoRA
    outputs_genai = tmp_path / "genai_lora"
    out_genai = run_wwb(
        [
            "--target-model",
            model_path,
            "--num-samples",
            "1",
            "--gt-data",
            gt_file,
            "--device",
            "CPU",
            "--model-type",
            model_type,
            "--genai",
            "--max_new_tokens",
            "32",
            "--output",
            outputs_genai,
            "--adapters",
            str(lora_adapter_file),
            "--alphas",
            str(genai_alpha),
        ]
    )

    assert (outputs_genai / "target.csv").exists()
    assert (outputs_genai / "metrics_per_question.csv").exists()
    assert (outputs_genai / "metrics.csv").exists()
    assert "Metrics for model" in out_genai
    similarity_genai = get_similarity(out_genai)

    assert similarity_genai >= genai_threshold


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


@pytest.mark.parametrize(
    (
        "model_id",
        "model_type",
        "lora_repo_id",
        "hf_alpha",
        "genai_alpha",
        "genai_threshold",
    ),
    [
        (
            "Qwen/Qwen2-VL-2B-Instruct",
            "visual-text",
            "saim1212/qwen2b-lora-100",
            1.0,
            2.0,
            0.99,
        ),
    ],
)
def test_vlm_genai_lora(
    model_id,
    model_type,
    lora_repo_id,
    hf_alpha,
    genai_alpha,
    genai_threshold,
    tmp_path,
):
    run_test_with_lora(
        model_id=model_id,
        model_type=model_type,
        lora_repo_id=lora_repo_id,
        lora_cache_subdir="wwb_qwen2b_lora_100",
        hf_alpha=hf_alpha,
        genai_alpha=genai_alpha,
        tmp_path=tmp_path,
        genai_threshold=genai_threshold,
    )
