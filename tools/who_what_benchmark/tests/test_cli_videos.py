import sys
import pytest
from test_cli_image import get_similarity
from conftest import convert_model, run_wwb


@pytest.mark.xfail(sys.platform == "darwin", reason="Not enough memory on macOS CI runners. Ticket CVS-179749")
@pytest.mark.xfail(sys.platform == "win32", reason="Access violation in OVLTXPipeline on Windows. Ticket CVS-179750")
@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [("optimum-intel-internal-testing/tiny-random-ltx-video", "text-to-video")],
)
def test_video_model_genai(model_id, model_type, tmp_path):
    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = convert_model(model_id)

    run_wwb(
        [
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
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
        ]
    )
    assert GT_FILE.exists()
    assert (tmp_path / "reference").exists()

    output = run_wwb(
        [
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
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
            "--output",
            tmp_path,
        ]
    )

    assert "Metrics for model" in output
    similarity = get_similarity(output)
    assert similarity >= 0.88
    assert (tmp_path / "target").exists()

    # test w/o models
    run_wwb(
        [
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
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
        ]
    )


@pytest.mark.xfail(sys.platform == "darwin", reason="Not enough memory on macOS CI runners. Ticket CVS-179749")
@pytest.mark.xfail(sys.platform == "win32", reason="Access violation in OVLTXPipeline on Windows. Ticket CVS-179750")
@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [("optimum-intel-internal-testing/tiny-random-ltx-video", "text-to-video")],
)
def test_video_model_genai_with_taylorseer(model_id, model_type, tmp_path):
    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = convert_model(model_id)

    run_wwb(
        [
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
            "--num-inference-steps",
            "4",
            "--video-frames-num",
            "9",
        ]
    )
    assert GT_FILE.exists()
    assert (tmp_path / "reference").exists()

    # Test with full TaylorSeer config
    output = run_wwb(
        [
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
            "--num-inference-steps",
            "4",
            "--video-frames-num",
            "9",
            "--output",
            tmp_path,
            "--taylorseer-config",
            '{"cache_interval": 3, "disable_cache_before_step": 4, "disable_cache_after_step": -1}',
        ]
    )

    assert "Metrics for model" in output
    assert "TaylorSeer config:" in output
    similarity = get_similarity(output)
    assert similarity >= 0.97


def run_test_with_lora(model_id, model_type, tmp_path, *, genai_threshold):
    if sys.platform == "darwin":
        pytest.xfail("Not enough memory on macOS CI runners. Ticket CVS-179749")
    if sys.platform == "win32":
        pytest.xfail("Access violation in OVLTXPipeline on Windows. Ticket CVS-179750")

    from ov_utils import get_ov_cache_dir, download_hf_files_to_cache

    gt_file = tmp_path / "gt.csv"
    model_path = convert_model(model_id)

    lora_cache_dir = get_ov_cache_dir() / "test_data" / "ltx_tiny_dummy_lora"
    lora_dir = download_hf_files_to_cache(
        "goyaladitya05/openvino-genai-test-files",
        lora_cache_dir,
        ["ltx_tiny_dummy_lora.safetensors"],
    )
    lora_file = lora_dir / "ltx_tiny_dummy_lora.safetensors"
    assert lora_file.exists(), f"LoRA adapter wasn't downloaded: {lora_file}"

    # 1) Generate GT using HF + LoRA
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
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
            "--adapters",
            str(lora_file),
            "--alphas",
            "0.9",
            "--hf",
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
            "--num-inference-steps",
            "2",
            "--video-frames-num",
            "9",
            "--adapters",
            str(lora_file),
            "--alphas",
            "0.9",
            "--output",
            outputs_genai,
        ]
    )

    assert "Metrics for model" in out_genai
    similarity = get_similarity(out_genai)
    assert similarity >= genai_threshold


@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [("optimum-intel-internal-testing/tiny-random-ltx-video", "text-to-video")],
)
def test_video_model_genai_with_lora(model_id, model_type, tmp_path):
    run_test_with_lora(model_id, model_type, tmp_path, genai_threshold=0.88)
