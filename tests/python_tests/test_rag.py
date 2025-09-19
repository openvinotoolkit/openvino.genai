# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import gc
from pathlib import Path
from openvino_genai import TextEmbeddingPipeline, TextRerankPipeline
from utils.hugging_face import download_and_convert_embeddings_models, download_and_convert_rerank_model
from langchain_core.documents.base import Document
from langchain_community.embeddings import OpenVINOBgeEmbeddings
from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
from typing import Literal, Union
import sys
import platform

EMBEDDINGS_TEST_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "mixedbread-ai/mxbai-embed-xsmall-v1",
]

RERANK_TEST_MODELS = [
    "cross-encoder/ms-marco-TinyBERT-L2-v2",  # sigmoid applied
    # "answerdotai/ModernBERT-base",  # 2 classes output, softmax applied. Skip until langchain OpenVINORerank supports it.
]

TEXT_DATASET = f"The commercial PC market is propelled by premium\
computing solutions that drive user productivity and help\
service organizations protect and maintain devices.\
Corporations must empower mobile and hybrid workers\
while extracting value from artificial intelligence (AI) to\
improve business outcomes. Moreover, both public and\
private sectors must address sustainability initiatives\
pertaining to the full life cycle of computing fleets. An\
inflection point in computing architecture is needed to stay\
ahead of evolving requirements.\
Introducing Intel® Core™ Ultra Processors\
Intel® Core™ Ultra processors shape the future of\
commercial computing in four major ways:\
Power Efficiency\
The new product line features a holistic approach to powerefficiency that benefits mobile work. Substantial changes to\
the microarchitecture, manufacturing process, packaging\
technology, and power management software result in up to\
40% lower processor power consumption for modern tasks\
such as video conferencing with a virtual camera. \
Artificial Intelligence\
Intel Core Ultra processors incorporate an AI-optimized\
architecture that supports new user experiences and the\
next wave of commercial applications. The CPU, GPU, and\
the new neural processing unit (NPU) are all capable of\
executing AI tasks as directed by application developers.\
For example, elevated mobile collaboration is possible with\
support for AI assisted background blur, noise suppression,\
eye tracking, and picture framing. Intel Core Ultra\
processors are capable of up to 2.5x the AI inference\
performance per watt as compared to Intel’s previous\
mobile processor offering.2\
"


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
    return [TEXT_DATASET[i : i + chunk_size] for i in range(0, len(TEXT_DATASET), chunk_size)]


def run_text_embedding_genai(
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


def run_text_embedding_langchain(
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


EmbeddingResult = Union[list[list[float]], list[list[int]], list[float], list[int]]
MAX_EMBEDDING_ERROR = 2e-6


def validate_embedding_results(result_1: EmbeddingResult, result_2: EmbeddingResult):
    np_result_1 = np.array(result_1)
    np_result_2 = np.array(result_2)

    max_error = np.abs(np_result_1 - np_result_2).max()
    assert max_error < MAX_EMBEDDING_ERROR, f"Max error: {max_error} is greater than allowed {MAX_EMBEDDING_ERROR}"


def run_text_embedding_pipeline_with_ref(
    models_path: Path,
    documents: list[str],
    config: TextEmbeddingPipeline.Config | None = None,
    task: Literal["embed_documents", "embed_query"] = "embed_documents",
):
    genai_result = run_text_embedding_genai(models_path, documents, config, task)
    langchain_result = run_text_embedding_langchain(models_path, documents, config, task)

    validate_embedding_results(genai_result, langchain_result)


def assert_rerank_results(result_1: list[tuple[int, float]], result_2: list[tuple[int, float]]):
    assert len(result_1) == len(result_2), f"Results length mismatch: {len(result_1)} != {len(result_2)}"
    for pair_1, pair_2 in zip(result_1, result_2):
        assert pair_1[0] == pair_2[0], f"Document IDs do not match: {pair_1[0]} != {pair_2[0]}"
        assert abs(pair_1[1] - pair_2[1]) < 1e-6, f"Scores do not match for document ID {pair_1[0]}: " f"{pair_1[1]} != {pair_2[1]}"


def run_text_rerank_langchain(
    models_path: Path,
    query: str,
    documents: list[str],
    config: TextRerankPipeline.Config | None = None,
):
    if not config:
        config = TextRerankPipeline.Config()

    reranker = OpenVINOReranker(model_name_or_path=str(models_path), top_n=config.top_n)

    langchain_documents = [Document(page_content=text) for text in documents]

    reranked_documents = reranker.compress_documents(documents=langchain_documents, query=query)

    return [(doc.metadata["id"], float(doc.metadata.get("relevance_score", -1))) for doc in reranked_documents]


def run_text_rerank_genai(
    models_path: Path,
    query: str,
    documents: list[str],
    config: TextRerankPipeline.Config | None = None,
):
    if not config:
        config = TextRerankPipeline.Config()

    reranker = TextRerankPipeline(models_path, "CPU", config=config)

    sync_result = reranker.rerank(
        query=query,
        texts=documents,
    )

    reranker.start_rerank_async(query, documents)
    async_result = reranker.wait_rerank()

    assert_rerank_results(sync_result, async_result)
    return async_result


def run_text_rerank_pipeline_with_ref(
    models_path: Path,
    query: str,
    documents: list[str],
    config: TextRerankPipeline.Config | None = None,
):
    genai_result = run_text_rerank_genai(models_path, query, documents, config)
    langchain_result = run_text_rerank_langchain(models_path, query, documents, config)

    assert_rerank_results(genai_result, langchain_result)


@pytest.mark.parametrize("download_and_convert_embeddings_models", ["BAAI/bge-small-en-v1.5"], indirect=True)
@pytest.mark.precommit
def test_embedding_constructors(download_and_convert_embeddings_models):
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


@pytest.mark.parametrize("download_and_convert_embeddings_models", EMBEDDINGS_TEST_MODELS, indirect=True)
@pytest.mark.parametrize(
    "config",
    [
        TextEmbeddingPipeline.Config(normalize=False),
        TextEmbeddingPipeline.Config(normalize=False, pooling_type=TextEmbeddingPipeline.PoolingType.MEAN),
        TextEmbeddingPipeline.Config(normalize=True),
        TextEmbeddingPipeline.Config(normalize=True, pooling_type=TextEmbeddingPipeline.PoolingType.MEAN),
        TextEmbeddingPipeline.Config(
            normalize=False,
            embed_instruction="Represent this document for searching relevant passages: ",
        ),
    ],
    ids=[
        "cls_pooling",
        "mean_pooling",
        "cls_pooling + normalize",
        "mean_pooling + normalize",
        "embed_instruction",
    ],
)
@pytest.mark.precommit
def test_embed_documents(download_and_convert_embeddings_models, dataset_documents, config):
    if (sys.platform == "linux"
            and "bge-small-en-v1.5" in str(download_and_convert_embeddings_models)
            and config.normalize
            and config.pooling_type == TextEmbeddingPipeline.PoolingType.CLS):
        pytest.xfail("Random segmentation fault. Ticket 172306")
    _, _, models_path = download_and_convert_embeddings_models
    run_text_embedding_pipeline_with_ref(models_path, dataset_documents, config, "embed_documents")


@pytest.mark.parametrize("download_and_convert_embeddings_models", EMBEDDINGS_TEST_MODELS, indirect=True)
@pytest.mark.parametrize(
    "config",
    [
        TextEmbeddingPipeline.Config(normalize=False),
        TextEmbeddingPipeline.Config(normalize=False, pooling_type=TextEmbeddingPipeline.PoolingType.MEAN),
        TextEmbeddingPipeline.Config(normalize=True),
        TextEmbeddingPipeline.Config(normalize=True, pooling_type=TextEmbeddingPipeline.PoolingType.MEAN),
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
    run_text_embedding_pipeline_with_ref(models_path, dataset_documents[:1], config, "embed_query")


@pytest.fixture(scope="module")
def dataset_embeddings_genai_default_config_refs(download_and_convert_embeddings_models, dataset_documents):
    _, _, models_path = download_and_convert_embeddings_models
    return run_text_embedding_genai(models_path, dataset_documents, None, "embed_documents")


@pytest.mark.parametrize("download_and_convert_embeddings_models", ["mixedbread-ai/mxbai-embed-xsmall-v1"], indirect=True)
@pytest.mark.parametrize(
    "config",
    [
        TextEmbeddingPipeline.Config(batch_size=4),
        TextEmbeddingPipeline.Config(max_length=50),
        TextEmbeddingPipeline.Config(max_length=50, batch_size=3),
        TextEmbeddingPipeline.Config(max_length=50, pad_to_max_length=True),
        TextEmbeddingPipeline.Config(batch_size=3, pad_to_max_length=True),
        TextEmbeddingPipeline.Config(max_length=50, pad_to_max_length=True, batch_size=4),
        TextEmbeddingPipeline.Config(max_length=64, pad_to_max_length=True, batch_size=1),
    ],
)
@pytest.mark.precommit
def test_fixed_shapes_configs(download_and_convert_embeddings_models, dataset_documents, config, dataset_embeddings_genai_default_config_refs):
    _, _, models_path = download_and_convert_embeddings_models

    docs_to_embed = dataset_documents[: config.batch_size] if config.batch_size else dataset_documents
    result = run_text_embedding_genai(models_path, docs_to_embed, config, "embed_documents")

    refs_to_validate = dataset_embeddings_genai_default_config_refs[: config.batch_size] if config.batch_size else dataset_embeddings_genai_default_config_refs
    validate_embedding_results(refs_to_validate, result)


@pytest.mark.parametrize("download_and_convert_embeddings_models", ["mixedbread-ai/mxbai-embed-xsmall-v1"], indirect=True)
@pytest.mark.parametrize(
    "config",
    [
        TextEmbeddingPipeline.Config(batch_size=0),
        # more than documents in dataset (9)
        TextEmbeddingPipeline.Config(batch_size=10),
        TextEmbeddingPipeline.Config(max_length=0),
        # more than model's max_position_embeddings (4096)
        TextEmbeddingPipeline.Config(max_length=4097),
    ],
)
@pytest.mark.xfail()
@pytest.mark.precommit
def test_fixed_shapes_configs_xfail(download_and_convert_embeddings_models, dataset_documents, config, dataset_embeddings_genai_default_config_refs):
    _, _, models_path = download_and_convert_embeddings_models

    docs_to_embed = dataset_documents[: config.batch_size] if config.batch_size else dataset_documents
    result = run_text_embedding_genai(models_path, docs_to_embed, config, "embed_documents")

    refs_to_validate = dataset_embeddings_genai_default_config_refs[: config.batch_size] if config.batch_size else dataset_embeddings_genai_default_config_refs
    validate_embedding_results(refs_to_validate, result)


@pytest.mark.parametrize("download_and_convert_embeddings_models", ["mixedbread-ai/mxbai-embed-xsmall-v1"], indirect=True)
@pytest.mark.parametrize(
    "config",
    [
        TextEmbeddingPipeline.Config(max_length=64, pad_to_max_length=True, batch_size=1),
        TextEmbeddingPipeline.Config(max_length=50, pad_to_max_length=True, batch_size=4),
    ],
)
@pytest.mark.precommit
@pytest.mark.skipif(
    sys.platform == "darwin" or platform.machine() in ["aarch64", "arm64", "ARM64"],
    reason="NPU plugin is available only on Linux and Windows x86_64",
)
def test_npu_fallback(download_and_convert_embeddings_models, dataset_documents, config, dataset_embeddings_genai_default_config_refs):
    _, _, models_path = download_and_convert_embeddings_models

    NPU_FALLBACK_PROPERTIES = {"NPU_USE_NPUW": "YES", "NPUW_DEVICES": "CPU", "NPUW_ONLINE_PIPELINE": "NONE"}

    pipeline = TextEmbeddingPipeline(models_path, "NPU", config, **NPU_FALLBACK_PROPERTIES)
    docs_to_embed = dataset_documents[: config.batch_size] if config.batch_size else dataset_documents
    result = pipeline.embed_documents(docs_to_embed)

    refs_to_validate = dataset_embeddings_genai_default_config_refs[: config.batch_size] if config.batch_size else dataset_embeddings_genai_default_config_refs
    validate_embedding_results(refs_to_validate, result)


@pytest.mark.parametrize("download_and_convert_rerank_model", [RERANK_TEST_MODELS[0]], indirect=True)
@pytest.mark.precommit
def test_rerank_constructors(download_and_convert_rerank_model):
    _, _, models_path = download_and_convert_rerank_model

    TextRerankPipeline(models_path, "CPU")
    TextRerankPipeline(models_path, "CPU", TextRerankPipeline.Config())
    TextRerankPipeline(
        models_path,
        "CPU",
        TextRerankPipeline.Config(),
        PERFORMANCE_HINT_NUM_REQUESTS=2,
    )
    TextRerankPipeline(
        models_path,
        "CPU",
        top_n=2,
    )
    TextRerankPipeline(
        models_path,
        "CPU",
        top_n=2,
        PERFORMANCE_HINT_NUM_REQUESTS=2,
    )


@pytest.mark.parametrize("download_and_convert_rerank_model", RERANK_TEST_MODELS, indirect=True)
@pytest.mark.parametrize("query", ["What are the main features of Intel Core Ultra processors?"])
@pytest.mark.parametrize(
    "config",
    [
        TextRerankPipeline.Config(),
        TextRerankPipeline.Config(top_n=10),
    ],
    ids=[
        "top_n=default",
        "top_n=10",
    ],
)
@pytest.mark.precommit
def test_rerank_documents(download_and_convert_rerank_model, dataset_documents, query, config):
    _, _, models_path = download_and_convert_rerank_model
    run_text_rerank_pipeline_with_ref(models_path, query, dataset_documents, config)
