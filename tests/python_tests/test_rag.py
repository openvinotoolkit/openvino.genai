# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import gc
from pathlib import Path
from openvino_genai import TextEmbeddingPipeline
from utils.hugging_face import download_and_convert_embeddings_models
from langchain_community.embeddings import OpenVINOBgeEmbeddings
from typing import Literal

TEST_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "mixedbread-ai/mxbai-embed-xsmall-v1",
]

TEXT_DATASET = f"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aliquam tempus mollis suscipit. Pellentesque id suscipit magna. Pellentesque condimentum magna vel nisi condimentum suscipit. Interdum et malesuada fames ac ante ipsum primis in faucibus. Duis a urna ac eros accumsan commodo non non magna. Aenean mattis commodo urna, ac interdum turpis semper eu. Ut ut pharetra quam. Suspendisse erat tortor, vulputate sit amet quam eu, accumsan facilisis est. Ut varius nibh quis suscipit tempor. Pellentesque luctus, turpis id hendrerit condimentum, urna augue ultricies elit, vitae lacinia mauris enim id lacus.\
Sed blandit in odio sed sagittis. Donec et turpis tincidunt nisl suscipit sodales id et eros. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Praesent tincidunt quam feugiat, luctus lacus semper, pharetra lorem. Aenean tincidunt, mi id porttitor fringilla, nisi nibh posuere dui, a vulputate libero ex in velit. Nunc non mauris faucibus nibh mattis venenatis. Integer maximus lectus eu mollis sodales. Donec sed varius tortor. Morbi sagittis at ex at semper. Sed in euismod mi, at viverra ante.\
Nulla luctus accumsan varius. Cras ut tempor est, vitae vehicula velit. In consectetur mi eget elit tempus, at pellentesque felis auctor. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Cras tempor feugiat congue. Interdum et malesuada fames ac ante ipsum primis in faucibus. Suspendisse ac leo ut felis fermentum sagittis rutrum iaculis lectus. Aenean a lorem lobortis, suscipit leo id, malesuada velit. Phasellus vitae eros placerat, sodales ex eget, porta orci. Aliquam erat volutpat. Nam eget nisi at nibh euismod tempus vel at massa. Vivamus malesuada dui quis nibh congue facilisis."


@pytest.fixture(scope="class", autouse=True)
def run_gc_after_test():
    """
    Fixture to run garbage collection after each test class.
    This is a workaround to minimize memory consumption during tests and allow the use of less powerful CI runners.
    """
    yield
    gc.collect()


@pytest.fixture(scope="module")
def dataset_documents(chunk_size=200):
    return [
        TEXT_DATASET[i : i + chunk_size]
        for i in range(0, len(TEXT_DATASET), chunk_size)
    ]


def run_genai(
    models_path: Path,
    documents: list[str],
    config: TextEmbeddingPipeline.Config | None = None,
    task: Literal["embed_documents", "embed_query"] = "embed_documents",
):
    if not config:
        config = TextEmbeddingPipeline.Config()

    pipeline = TextEmbeddingPipeline(models_path, "CPU", config)

    if config.batch_size:
        documents = documents[: config.batch_size]

    if task == "embed_documents":
        return pipeline.embed_documents(documents)
    else:
        return pipeline.embed_query(documents[0])


def run_langchain(
    models_path: Path,
    documents: list[str],
    config: TextEmbeddingPipeline.Config | None = None,
    task: Literal["embed_documents", "embed_query"] = "embed_documents",
):
    if not config:
        config = TextEmbeddingPipeline.Config()

    encode_kwargs = {
        "normalize_embeddings": config.normalize,
        # batch size affects the result
        "batch_size": len(documents),
    }
    if config.pooling_type == TextEmbeddingPipeline.PoolingType.MEAN:
        encode_kwargs["mean_pooling"] = True

    ov_embeddings = OpenVINOBgeEmbeddings(
        model_name_or_path=str(models_path),
        model_kwargs={"device": "CPU"},
        encode_kwargs=encode_kwargs,
    )

    # align instructions
    ov_embeddings.embed_instruction = config.embed_instruction or ""
    ov_embeddings.query_instruction = config.query_instruction or ""

    if config.batch_size:
        documents = documents[: config.batch_size]

    if task == "embed_documents":
        return ov_embeddings.embed_documents(documents)
    else:
        return ov_embeddings.embed_query(documents[0])


def run_pipeline_with_ref(
    models_path: Path,
    documents: list[str],
    config: TextEmbeddingPipeline.Config | None = None,
    task: Literal["embed_documents", "embed_query"] = "embed_documents",
):
    genai_result = run_genai(models_path, documents, config, task)
    langchain_result = run_langchain(models_path, documents, config, task)

    np_genai_result = np.array(genai_result)
    np_langchain_result = np.array(langchain_result)

    max_error = np.abs(np_genai_result - np_langchain_result).max()
    print(f"Max error: {max_error}")
    assert (
        np.abs(np_genai_result - np_langchain_result).max() < 2e-6
    ), f"Max error: {max_error}"


@pytest.mark.parametrize(
    "download_and_convert_embeddings_models", ["BAAI/bge-small-en-v1.5"], indirect=True
)
@pytest.mark.precommit
def test_constructors(download_and_convert_embeddings_models):
    _, _, models_path = download_and_convert_embeddings_models

    TextEmbeddingPipeline(models_path, "CPU")
    TextEmbeddingPipeline(models_path, "CPU", TextEmbeddingPipeline.Config())
    TextEmbeddingPipeline(
        models_path,
        "CPU",
        TextEmbeddingPipeline.Config(),
        PERFORMANCE_HINT_NUM_REQUESTS=2,
    )
    TextEmbeddingPipeline(
        models_path,
        "CPU",
        normalize=True,
        pooling_type=TextEmbeddingPipeline.PoolingType.MEAN,
    )
    TextEmbeddingPipeline(
        models_path,
        "CPU",
        normalize=True,
        pooling_type=TextEmbeddingPipeline.PoolingType.MEAN,
        PERFORMANCE_HINT_NUM_REQUESTS=2,
    )


@pytest.mark.parametrize(
    "download_and_convert_embeddings_models", TEST_MODELS, indirect=True
)
@pytest.mark.parametrize(
    "config",
    [
        TextEmbeddingPipeline.Config(normalize=False),
        TextEmbeddingPipeline.Config(
            normalize=False, pooling_type=TextEmbeddingPipeline.PoolingType.MEAN
        ),
        TextEmbeddingPipeline.Config(normalize=True),
        TextEmbeddingPipeline.Config(
            normalize=True, pooling_type=TextEmbeddingPipeline.PoolingType.MEAN
        ),
        TextEmbeddingPipeline.Config(
            normalize=False,
            embed_instruction="Represent this document for searching relevant passages: ",
        ),
        TextEmbeddingPipeline.Config(max_length=256, pad_to_max_length=True, batch_size=4),
    ],
    ids=[
        "cls_pooling",
        "mean_pooling",
        "cls_pooling + normalize",
        "mean_pooling + normalize",
        "embed_instruction",
        "max_length + pad_to_max_length + batch_size",
    ],
)
@pytest.mark.precommit
def test_embed_documents(
    download_and_convert_embeddings_models, dataset_documents, config
):
    _, _, models_path = download_and_convert_embeddings_models
    run_pipeline_with_ref(models_path, dataset_documents, config, "embed_documents")


@pytest.mark.parametrize(
    "download_and_convert_embeddings_models", TEST_MODELS, indirect=True
)
@pytest.mark.parametrize(
    "config",
    [
        TextEmbeddingPipeline.Config(normalize=False),
        TextEmbeddingPipeline.Config(
            normalize=False, pooling_type=TextEmbeddingPipeline.PoolingType.MEAN
        ),
        TextEmbeddingPipeline.Config(normalize=True),
        TextEmbeddingPipeline.Config(
            normalize=True, pooling_type=TextEmbeddingPipeline.PoolingType.MEAN
        ),
        TextEmbeddingPipeline.Config(
            normalize=False,
            query_instruction="Represent this query for searching relevant passages: ",
        ),
    ],
    ids=[
        "cls_pooling",
        "mean_pooling",
        "cls_pooling + normalize",
        "mean_pooling + normalize",
        "query_instruction",
    ],
)
@pytest.mark.precommit
def test_embed_query(download_and_convert_embeddings_models, dataset_documents, config):
    _, _, models_path = download_and_convert_embeddings_models
    run_pipeline_with_ref(models_path, dataset_documents[:1], config, "embed_query")
