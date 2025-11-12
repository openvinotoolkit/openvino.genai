import sys
import pytest

from conftest import convert_model, run_wwb


def remove_artifacts(artifacts_path: Path, file_type="outputs"):
    logger.info(f"Remove {file_type}")
    shutil.rmtree(artifacts_path)


@pytest.mark.parametrize(
    ("model_id", "model_type"),
    [
        pytest.param(
            "BAAI/bge-small-en-v1.5",
            "text-embedding",
            marks=pytest.mark.xfail(sys.platform == "darwin", reason="Hangs. Ticket 175534", run=False),
        ),
        ("Qwen/Qwen3-Embedding-0.6B", "text-embedding"),
    ],
)
def test_embeddings_basic(model_id, model_type, tmp_path):
    if sys.platform == "win32":
        pytest.xfail("Ticket 178790")

    GT_FILE = tmp_path / "gt.csv"
    MODEL_PATH = convert_model(model_id)

    # Collect reference with HF model
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
            "--hf",
        ]
    )

    outputs_path = tmp_path / "optimum"
    # test Optimum
    run_wwb(
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
        ]
    )

    assert (outputs_path / "target").exists()
    assert (outputs_path / "target.csv").exists()
    assert (outputs_path / "metrics_per_question.csv").exists()
    assert (outputs_path / "metrics.csv").exists()
    assert "Metrics for model" in outputs

    similarity = get_similarity(outputs)
    assert similarity >= SIMILARITY_THRESHOLD

    remove_artifacts(outputs_path)

    outputs_path = tmp_path / "genai"
    # test GenAI
    run_wwb(
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
            "--output",
            tmp_path,
        ]
    )

    assert (outputs_path / "target").exists()
    assert (outputs_path / "target.csv").exists()
    assert (outputs_path / "metrics_per_question.csv").exists()
    assert (outputs_path / "metrics.csv").exists()
    assert "Metrics for model" in outputs

    similarity = get_similarity(outputs)
    assert similarity >= SIMILARITY_THRESHOLD

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
            "--genai",
        ]
    )


@pytest.mark.parametrize(
    ("model_id", "model_type", "batch_size"),
    [
        ("Qwen/Qwen3-Embedding-0.6B", "text-embedding", 1),
        ("Qwen/Qwen3-Embedding-0.6B", "text-embedding", 12),
    ],
)
def test_embeddings_with_batch(model_id, model_type, batch_size, tmp_path):
    if sys.platform == "win32":
        pytest.xfail("Ticket 178790")
    GT_FILE = tmp_path / f"gt_batch_{batch_size}.csv"
    MODEL_PATH = convert_model(model_id)

    # Collect reference with HF model
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
            "--embeds_batch_size",
            str(batch_size),
            "--hf",
        ]
    )

    # test Optimum
    run_wwb(
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
            "--embeds_batch_size",
            str(batch_size),
        ]
    )

    # test GenAI
    run_wwb(
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
            "--output",
            tmp_path,
            "--embeds_batch_size",
            str(batch_size),
        ]
    )

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
            "--genai",
            "--embeds_batch_size",
            str(batch_size),
        ]
    )
