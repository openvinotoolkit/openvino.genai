import sys
import pytest

from conftest import convert_model, run_wwb


@pytest.mark.parametrize(
    ("model_id"),
    [
        pytest.param(
            "BAAI/bge-small-en-v1.5",
            marks=pytest.mark.xfail(sys.platform == "darwin", reason="Hangs. Ticket 175534", run=False),
        ),
        ("Qwen/Qwen3-Embedding-0.6B"),
    ],
)
def test_embeddings_basic(model_id, tmp_path):
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
            "text-embedding",
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
            "text-embedding",
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
            "text-embedding",
            "--genai",
            "--output",
            tmp_path,
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
            "text-embedding",
            "--genai",
        ]
    )


@pytest.mark.parametrize(
    ("model_id", "batch_size"),
    [
        ("Qwen/Qwen3-Embedding-0.6B", 1),
        ("Qwen/Qwen3-Embedding-0.6B", 12),
    ],
)
def test_embeddings_with_batch(model_id, batch_size, tmp_path):
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
            "text-embedding",
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
            "text-embedding",
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
            "text-embedding",
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
            "text-embedding",
            "--genai",
            "--embeds_batch_size",
            str(batch_size),
        ]
    )
