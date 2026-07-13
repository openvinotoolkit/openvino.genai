# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import gc
from pathlib import Path
import requests
import openvino as ov
import openvino_genai
from openvino_genai import EmbeddingPipeline, TextEmbeddingPipeline, TextRerankPipeline, VideoMetadata
from utils.hugging_face import download_and_convert_model, download_and_convert_model_class, OVConvertedModelSchema
from langchain_core.documents.base import Document
from langchain_community.embeddings import OpenVINOBgeEmbeddings
from langchain_community.document_compressors.openvino_rerank import OpenVINOReranker
from typing import Literal
import sys
from optimum.intel import OVModelForFeatureExtraction, OVModelForSequenceClassification
from PIL import Image
from torch import Tensor
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from utils.constants import NPUW_CPU_PROPERTIES
from utils.ov_genai_pipelines import should_skip_npuw_tests
from utils.qwen3_reranker_utils import qwen3_reranker_format_queries, qwen3_reranker_format_document
from samples.conftest import TEST_FILES

EMBEDDINGS_TEST_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "mixedbread-ai/mxbai-embed-xsmall-v1",
]

MULTIMODAL_EMBEDDINGS_TEST_MODELS = [
    # "Qwen/Qwen3-VL-Embedding-2B",
    "optimum-intel-internal-testing/tiny-random-qwen3-vl-embedding"
]

# nomic_bert is a bidirectional RoPE + gated-MLP BERT-family encoder (not a generative
# architecture). It is not yet listed in EMBEDDINGS_TEST_MODELS (large download vs. the
# tiny models above), but TextEmbeddingPipeline serves it out-of-the-box with MEAN pooling.
NOMIC_EMBED_TEXT_MODEL = "nomic-ai/nomic-embed-text-v1.5"

RERANK_TEST_MODELS = [
    "cross-encoder/ms-marco-TinyBERT-L2-v2",  # sigmoid applied
    # "answerdotai/ModernBERT-base",  # 2 classes output, softmax applied. Skip until langchain OpenVINORerank supports it.
]

QWEN3_RERANK_SEQ_CLS = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
QWEN3_RERANK = "Qwen/Qwen3-Reranker-0.6B"

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


def test_embedding_pipeline_public_api():
    assert hasattr(openvino_genai, "EmbeddingPipeline")
    assert EmbeddingPipeline.__name__ == "EmbeddingPipeline"
    assert hasattr(EmbeddingPipeline, "embed")
    assert "prompt" in EmbeddingPipeline.embed.__doc__
    assert hasattr(openvino_genai, "EmbedResult")
    assert hasattr(openvino_genai.EmbedResult, "embeddings")


@pytest.fixture(scope="module")
def rerank_model(request) -> OVConvertedModelSchema:
    model_id = request.param
    return download_and_convert_model_class(model_id, OVModelForSequenceClassification)


@pytest.fixture(scope="module")
def emb_model(request) -> OVConvertedModelSchema:
    model_id = request.param
    return download_and_convert_model_class(model_id, OVModelForFeatureExtraction)


@pytest.fixture(scope="module")
def multimodal_emb_model(request) -> OVConvertedModelSchema:
    model_id = request.param
    return download_and_convert_model_class(model_id, OVModelForFeatureExtraction)


@pytest.fixture(scope="module")
def multimodal_emb_hf_components(multimodal_emb_model: OVConvertedModelSchema):
    processor = AutoProcessor.from_pretrained(multimodal_emb_model.model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(multimodal_emb_model.model_id, trust_remote_code=True).eval()
    return processor, model


@pytest.fixture(scope="module")
def llm_model(request: pytest.FixtureRequest) -> OVConvertedModelSchema:
    tokenizer_kwargs = {
        "padding_side": "left"
    }
    return download_and_convert_model(request.param, **tokenizer_kwargs)


@pytest.fixture(autouse=True)
def run_gc_after_test():
    """
    Fixture to run garbage collection after each test.
    This is a workaround to minimize memory consumption
    during tests and allow the use of less powerful CI runners.
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
    device: str = "CPU",
    properties: dict | None = None,
):
    if not config:
        config = TextEmbeddingPipeline.Config()

    if properties:
        pipeline = TextEmbeddingPipeline(models_path, device, config, **properties)
    else:
        pipeline = TextEmbeddingPipeline(models_path, device, config)

    if config.batch_size:
        documents = documents[: config.batch_size]

    if task == "embed_documents":
        return pipeline.embed_documents(documents)
    else:
        return pipeline.embed_query(documents[0])


@pytest.fixture(scope="module")
def cat_image_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    image_path = tmp_path_factory.mktemp("rag_test_data") / "cat"
    if not image_path.exists():
        response = requests.get(TEST_FILES["cat"], timeout=30)
        response.raise_for_status()
        image_path.write_bytes(response.content)
    return image_path


def get_image_as_array(image_path: Path) -> np.ndarray:
    return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)


def get_image_tensor(image_path: Path) -> ov.Tensor:
    return ov.Tensor(get_image_as_array(image_path))


def get_video_as_array(image_path: Path) -> np.ndarray:
    image = get_image_as_array(image_path)
    return np.stack([image] * 4, axis=0)


def get_video_tensor(image_path: Path) -> ov.Tensor:
    return ov.Tensor(get_video_as_array(image_path))


def get_video_metadata() -> VideoMetadata:
    video_metadata = VideoMetadata()
    video_metadata.frames_indices = [0, 1, 2, 3]
    return video_metadata


def assert_embedding_tensor(result, batch_size: int):
    assert isinstance(result.embeddings, ov.Tensor)
    assert result.embeddings.shape[0] == batch_size
    assert result.embeddings.shape[1] > 0


def run_multimodal_embedding_hf(
    hf_processor,
    hf_model,
    text: str,
    image: np.ndarray | None = None,
    prompt: str | None = None,
) -> np.ndarray:
    images = None
    if image is not None or prompt is not None:
        content = [{"type": "text", "text": text}]
        if image is not None:
            image = Image.fromarray(image)
            content.append({"type": "image", "image": image})
            images = [image]

        conversation = [
            {"role": "user", "content": content},
        ]
        if prompt is not None:
            conversation.insert(0, {"role": "system", "content": [{"type": "text", "text": prompt}]})
        text = hf_processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)

    kwargs = {"text": [text], "return_tensors": "pt", "padding": True}
    if images is not None:
        kwargs["images"] = images

    with torch.no_grad():
        inputs = hf_processor(**kwargs)
        outputs = hf_model(**inputs)

    sequence_lengths = inputs["attention_mask"].sum(dim=1) - 1
    pooled = outputs.last_hidden_state[torch.arange(outputs.last_hidden_state.shape[0]), sequence_lengths]
    return F.normalize(pooled.to(torch.float32), p=2, dim=-1).cpu().numpy()


def run_multimodal_embedding_sentence_transformers(
    model_id: str,
    text: str,
    image: np.ndarray | None = None,
    prompt: str | None = None,
) -> np.ndarray:
    import sentence_transformers

    model = sentence_transformers.SentenceTransformer(model_id)
    model_input = {"text": text}
    if image is not None:
        model_input["image"] = Image.fromarray(image)
    return model.encode([model_input], prompt=prompt, convert_to_numpy=True).astype(np.float32)


def run_multimodal_embedding_transformers(
    model_id: str,
    text: str,
    image: np.ndarray | None = None,
    prompt: str = "Represent the user's input.",
) -> np.ndarray:
    processor = AutoProcessor.from_pretrained(model_id, padding_side="right")
    model = AutoModel.from_pretrained(model_id)

    content = [{"type": "text", "text": text}]
    if image is not None:
        content.append({"type": "image", "image": Image.fromarray(image)})
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
        {"role": "user", "content": content},
    ]
    templated_text = processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
    inputs = processor(
        text=[templated_text],
        images=[Image.fromarray(image)] if image is not None else None,
        padding=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    sequence_lengths = inputs["attention_mask"].sum(dim=1) - 1
    pooled = outputs.last_hidden_state[torch.arange(outputs.last_hidden_state.shape[0]), sequence_lengths]
    return F.normalize(pooled.to(torch.float32), p=2, dim=-1).cpu().numpy()


def assert_embedding_matches_hf_cosine(result, reference: np.ndarray, max_cosine_diff: float):
    actual = np.array(result.embeddings.data, dtype=np.float32).reshape(result.embeddings.shape)
    actual_norm = np.linalg.norm(actual)
    reference_norm = np.linalg.norm(reference)
    cosine_similarity = float(actual.flatten() @ reference.flatten() / (actual_norm * reference_norm))
    cosine_difference = 1.0 - cosine_similarity
    message = (
        f"EmbeddingPipeline output does not match HF reference by cosine\n"
        f"actual shape: {actual.shape}, reference shape: {reference.shape}\n"
        f"cosine similarity: {cosine_similarity}\n"
        f"cosine difference: {cosine_difference}\n"
        f"actual first 8: {actual.flatten()[:8].tolist()}\n"
        f"reference first 8: {reference.flatten()[:8].tolist()}"
    )
    assert cosine_difference < max_cosine_diff, message


@pytest.mark.parametrize("emb_model", [EMBEDDINGS_TEST_MODELS[0]], indirect=True)
def test_embedding_pipeline_prompt_api_reaches_cpp(emb_model):
    pipeline = EmbeddingPipeline(emb_model.models_path, "CPU")

    result = pipeline.embed("What is OpenVINO?")
    assert result.embeddings.shape[0] == 1
    assert result.embeddings.shape[1] > 0

    batch_result = pipeline.embed(["What is OpenVINO?", "What is OpenVINO GenAI?"])
    assert batch_result.embeddings.shape[0] == 2
    assert batch_result.embeddings.shape[1] == result.embeddings.shape[1]

    with pytest.raises(
        RuntimeError, match="TextEmbeddingPipeline fallback is active and does not support image/video input"
    ):
        pipeline.embed("What is OpenVINO?", images=[ov.Tensor(np.zeros((1, 1, 1, 1), dtype=np.float32))])


@pytest.mark.parametrize("multimodal_emb_model", MULTIMODAL_EMBEDDINGS_TEST_MODELS, indirect=True)
def test_qwen3_vl_embedding_text_and_prompt(multimodal_emb_model):
    pipeline = EmbeddingPipeline(multimodal_emb_model.models_path, "CPU")
    text = "What is OpenVINO?"
    prompt = "Represent the user's input."

    result = pipeline.embed(text, prompt=prompt)
    assert_embedding_tensor(result, 1)


@pytest.mark.parametrize("multimodal_emb_model", MULTIMODAL_EMBEDDINGS_TEST_MODELS, indirect=True)
def test_qwen3_vl_embedding_text_batch_consistency(multimodal_emb_model):
    """Test that batch processing gives same results as individual processing."""
    pipeline = EmbeddingPipeline(multimodal_emb_model.models_path, "CPU")
    texts = ["What is OpenVINO?", "How does deep learning work?"]
    prompt = "Represent the user's input."

    # Process individually
    result1 = pipeline.embed(texts[0], prompt=prompt)
    result2 = pipeline.embed(texts[1], prompt=prompt)

    # Process as batch
    result_batch = pipeline.embed(texts, prompt=prompt)

    # Verify shapes
    assert result_batch.embeddings.shape == (2, result1.embeddings.shape[1])

    # Verify results match
    batch_data = np.array(result_batch.embeddings.data).reshape(result_batch.embeddings.shape)
    assert np.allclose(result1.embeddings.data, batch_data[0], rtol=1e-4, atol=1e-4)
    assert np.allclose(result2.embeddings.data, batch_data[1], rtol=1e-4, atol=1e-4)


@pytest.mark.xfail(reason="Ticket - CVS-189808")
@pytest.mark.parametrize("multimodal_emb_model", MULTIMODAL_EMBEDDINGS_TEST_MODELS, indirect=True)
def test_qwen3_vl_embedding_text_and_image(multimodal_emb_model, multimodal_emb_hf_components, cat_image_path):
    pipeline = EmbeddingPipeline(multimodal_emb_model.models_path, "CPU")
    image_array = get_image_as_array(cat_image_path)
    image = get_image_tensor(cat_image_path)
    image_text_prompt = " المرأ playing with her dog on a beach at sunset."
    prompt = "Represent the user's input."
    hf_processor, hf_model = multimodal_emb_hf_components

    result = pipeline.embed(image_text_prompt, images=[image], prompt=prompt)
    assert_embedding_tensor(result, 1)
    hf_result = run_multimodal_embedding_hf(hf_processor, hf_model, image_text_prompt, image=image_array, prompt=prompt)
    assert_embedding_matches_hf_cosine(result, hf_result, max_cosine_diff=0.05)


@pytest.mark.xfail(reason="Ticket - CVS-189808")
@pytest.mark.parametrize("multimodal_emb_model", MULTIMODAL_EMBEDDINGS_TEST_MODELS, indirect=True)
def test_qwen3_vl_embedding_text_and_image_sentence_transformers(
    multimodal_emb_model,
    cat_image_path,
):
    pipeline = EmbeddingPipeline(multimodal_emb_model.models_path, "CPU")
    image_array = get_image_as_array(cat_image_path)
    image = get_image_tensor(cat_image_path)
    text = "A woman playing with her dog on a beach at sunset."
    prompt = "Represent the user's input."
    # prompt = ""

    result = pipeline.embed(text, images=[image], prompt=prompt)
    assert_embedding_tensor(result, 1)

    sentence_transformers_result = run_multimodal_embedding_sentence_transformers(
        multimodal_emb_model.model_id,
        text,
        image=image_array,
        prompt=prompt,
    )
    assert_embedding_matches_hf_cosine(result, sentence_transformers_result, max_cosine_diff=0.06)


@pytest.mark.xfail(reason="Ticket - CVS-189808")
@pytest.mark.parametrize("model_id", MULTIMODAL_EMBEDDINGS_TEST_MODELS)
def test_qwen3_vl_embedding_sentence_transformers_matches_transformers(
    model_id,
    cat_image_path,
):
    image_array = get_image_as_array(cat_image_path)
    text = "A woman"
    prompt = "hi"

    sentence_transformers_result = run_multimodal_embedding_sentence_transformers(
        model_id,
        text,
        image=image_array,
        prompt=prompt,
    )
    transformers_result = run_multimodal_embedding_transformers(
        model_id,
        text,
        image=image_array,
        prompt=prompt,
    )
    assert_embedding_matches_hf_cosine(
        openvino_genai.EmbedResult(ov.Tensor(sentence_transformers_result)),
        transformers_result,
        max_cosine_diff=0.05,
    )


@pytest.mark.parametrize("multimodal_emb_model", MULTIMODAL_EMBEDDINGS_TEST_MODELS, indirect=True)
def test_qwen3_vl_embedding_text_and_video(multimodal_emb_model, cat_image_path):
    pipeline = EmbeddingPipeline(multimodal_emb_model.models_path, "CPU")
    video = get_video_tensor(cat_image_path)
    video_metadata = get_video_metadata()
    text = "Represent this video."

    result = pipeline.embed(text, videos=[video], videos_metadata=[video_metadata])
    assert_embedding_tensor(result, 1)


@pytest.mark.parametrize("multimodal_emb_model", MULTIMODAL_EMBEDDINGS_TEST_MODELS, indirect=True)
def test_qwen3_vl_embedding_three_texts_image_and_video(multimodal_emb_model, cat_image_path):
    pipeline = EmbeddingPipeline(multimodal_emb_model.models_path, "CPU")
    image = get_image_tensor(cat_image_path)
    video = get_video_tensor(cat_image_path)
    video_metadata = get_video_metadata()
    texts = ["Represent OpenVINO.", "Represent this image.", "Represent this video."]

    result = pipeline.embed(
        texts,
        images=[image],
        videos=[video],
        videos_metadata=[video_metadata],
        prompt="Represent the user's input.",
    )
    assert_embedding_tensor(result, len(texts))


@pytest.mark.parametrize("emb_model", [EMBEDDINGS_TEST_MODELS[0]], indirect=True)
def test_embedding_pipeline_matches_text_embedding_pipeline(emb_model):
    text = "What is OpenVINO?"
    embedding_pipeline = EmbeddingPipeline(emb_model.models_path, "CPU")
    text_embedding_pipeline = TextEmbeddingPipeline(emb_model.models_path, "CPU")

    embedding_result = embedding_pipeline.embed(text)
    text_embedding_result = text_embedding_pipeline.embed_documents([text])

    np.testing.assert_allclose(
        embedding_result.embeddings.data,
        np.asarray(text_embedding_result, dtype=np.float32),
        atol=MAX_EMBEDDING_ERROR,
        rtol=0,
    )


@pytest.mark.parametrize("emb_model", ["BAAI/bge-small-en-v1.5"], indirect=True)
def test_embedding_pipeline_prompt_matches_embed_instruction(emb_model):
    text = "What is OpenVINO?"
    prompt = "Represent this document for searching relevant passages: "

    embedding_pipeline = EmbeddingPipeline(emb_model.models_path, "CPU")
    text_embedding_pipeline = TextEmbeddingPipeline(
        emb_model.models_path, "CPU", TextEmbeddingPipeline.Config(embed_instruction=prompt)
    )

    embedding_result = embedding_pipeline.embed(text, embedding_prompt=prompt)
    text_embedding_result = text_embedding_pipeline.embed_documents([text])

    np.testing.assert_allclose(
        embedding_result.embeddings.data,
        np.asarray(text_embedding_result, dtype=np.float32),
        atol=MAX_EMBEDDING_ERROR,
        rtol=0,
    )


@pytest.fixture(scope="module")
def nomic_embed_model() -> OVConvertedModelSchema:
    try:
        return download_and_convert_model_class(NOMIC_EMBED_TEXT_MODEL, OVModelForFeatureExtraction)
    except ValueError as e:
        # nomic_bert OpenVINO export support (NomicBertOpenVINOConfig) is only available once
        # huggingface/optimum-intel#1864 is merged and released. Until then, CI installs a
        # stock optimum-intel from PyPI that does not recognize this architecture. Skip
        # cleanly rather than failing so this test starts running automatically once the
        # dependency lands.
        if "unsupported architecture" in str(e) or "nomic_bert" in str(e):
            pytest.skip(
                f"nomic_bert export not yet supported by installed optimum-intel "
                f"(depends on huggingface/optimum-intel#1864): {e}"
            )
        raise


def test_nomic_bert_text_embedding_pipeline_matches_hf(nomic_embed_model):
    """
    nomic_bert (bidirectional RoPE + gated-MLP encoder, feature-extraction/embedding model)
    has no generative pipeline of its own. This test confirms that the generic,
    architecture-agnostic TextEmbeddingPipeline (MEAN pooling + normalize, matching the
    model's native sentence-transformers config) already serves it correctly, so no new
    GenAI pipeline class is required for this architecture.
    """
    docs = [
        "search_document: OpenVINO is a toolkit for optimizing deep learning models.",
        "search_document: The quick brown fox jumps over the lazy dog.",
    ]

    tokenizer = AutoTokenizer.from_pretrained(NOMIC_EMBED_TEXT_MODEL)
    hf_model = AutoModel.from_pretrained(NOMIC_EMBED_TEXT_MODEL, trust_remote_code=True).eval()
    encoded = tokenizer(docs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        hf_out = hf_model(**encoded)
    mask = encoded["attention_mask"].unsqueeze(-1).expand(hf_out.last_hidden_state.size()).float()
    pooled = (hf_out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    hf_embeddings = F.normalize(pooled, p=2, dim=1).numpy()

    config = TextEmbeddingPipeline.Config(pooling_type=TextEmbeddingPipeline.PoolingType.MEAN, normalize=True)
    pipeline = TextEmbeddingPipeline(nomic_embed_model.models_path, "CPU", config)
    genai_embeddings = np.asarray(pipeline.embed_documents(docs), dtype=np.float32)

    np.testing.assert_allclose(genai_embeddings, hf_embeddings, atol=MAX_EMBEDDING_ERROR, rtol=0)


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


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        batch_dim = torch.arange(batch_size, device=last_hidden_states.device)
        result = last_hidden_states[batch_dim, sequence_lengths]
        return result


# from transformers Qwen3-Embedding-0.6B model card: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#transformers-usage
def run_qwen3_embedding_optimum(
    model: OVModelForFeatureExtraction,
    tokenizer,
    documents: list[str],
    padding_side: Literal["left", "right"] = "left",
):
    encoded = tokenizer(
        documents,
        padding=True,
        truncation=True,
        padding_side=padding_side,
        return_tensors="pt",
    )
    outputs = model(**encoded)
    return last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])


EmbeddingResult = list[list[float]] | list[list[int]] | list[float] | list[int]
MAX_EMBEDDING_ERROR = 2e-6 if sys.platform != "darwin" else 0.02  # ARM64 macs have different results


def validate_embedding_results(
    result_1: EmbeddingResult, result_2: EmbeddingResult, threshold: float = MAX_EMBEDDING_ERROR
):
    __tracebackhide__ = True
    np_result_1 = np.array(result_1)
    np_result_2 = np.array(result_2)

    max_error = np.abs(np_result_1 - np_result_2).max()
    assert max_error < threshold, f"Max error: {max_error} is greater than allowed {threshold}"


def run_text_embedding_pipeline_with_ref(
    models_path: Path,
    documents: list[str],
    config: TextEmbeddingPipeline.Config | None = None,
    task: Literal["embed_documents", "embed_query"] = "embed_documents",
):
    __tracebackhide__ = True
    genai_result = run_text_embedding_genai(models_path, documents, config, task)
    langchain_result = run_text_embedding_langchain(models_path, documents, config, task)

    validate_embedding_results(genai_result, langchain_result)


def assert_rerank_results(result_1: list[tuple[int, float]], result_2: list[tuple[int, float]]):
    __tracebackhide__ = True
    score_diff_max = 1e-6 if sys.platform != "darwin" else 2e-4  # ARM64 macs have different results
    assert len(result_1) == len(result_2), f"Results length mismatch: {len(result_1)} != {len(result_2)}"
    for pair_1, pair_2 in zip(result_1, result_2):
        assert pair_1[0] == pair_2[0], f"Document IDs do not match: {pair_1[0]} != {pair_2[0]}"
        assert abs(pair_1[1] - pair_2[1]) < score_diff_max, f"Scores do not match for document ID {pair_1[0]}: " f"{pair_1[1]} != {pair_2[1]}"


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


def run_qwen3_rerank_optimum(
    model: OVModelForSequenceClassification,
    tokenizer,
    query: str,
    documents: list[str],
    config: TextRerankPipeline.Config | None = None,
):
    if not config:
        config = TextRerankPipeline.Config()

    concatenated = [query + doc for doc in documents]
    inputs = tokenizer(
        concatenated,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    logits = model(**inputs).logits

    # support seq-cls reranker
    if logits.shape[1] == 1:
        scores = logits.squeeze().sigmoid()
    # original postprocessing
    else:
        token_false_id = tokenizer.convert_tokens_to_ids("no")
        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        batch_scores = logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp()

    with_ids = list(enumerate(scores.tolist()))
    sorted_by_score = sorted(with_ids, key=lambda x: x[1], reverse=True)
    return sorted_by_score[: config.top_n]


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


@pytest.mark.parametrize(
    "emb_model", 
    ["Qwen/Qwen3-Embedding-0.6B"], 
    indirect=True,
)
@pytest.mark.parametrize(
    "config",
    [
        TextEmbeddingPipeline.Config(
            normalize=False, 
            pooling_type=TextEmbeddingPipeline.PoolingType.LAST_TOKEN, 
            padding_side="left"
        ),
        TextEmbeddingPipeline.Config(
            normalize=False, 
            pooling_type=TextEmbeddingPipeline.PoolingType.LAST_TOKEN
        ),
    ],
)
@pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 174635")
def test_qwen3_embedding(emb_model, dataset_documents, config):
    embeddings_opt = run_qwen3_embedding_optimum(
        emb_model.opt_model, 
        emb_model.hf_tokenizer, 
        dataset_documents, 
        config.padding_side,
    )
    embeddings_genai = run_text_embedding_genai(
        emb_model.models_path, 
        dataset_documents, 
        config, 
        "embed_documents",
    )
    validate_embedding_results(embeddings_genai, embeddings_opt.tolist())


@pytest.mark.parametrize(
    "emb_model",
    ["Qwen/Qwen3-Embedding-0.6B"],
    indirect=True,
)
@pytest.mark.parametrize(
    ("config", "chunk_size", "threshold", "task"),
    [
        # Chunk disabled
        (
            TextEmbeddingPipeline.Config(
                batch_size=1,
                max_length=192,
                normalize=False,
                pad_to_max_length=False,
                pooling_type=TextEmbeddingPipeline.PoolingType.CLS,
                padding_side="right",
            ),
            0,
            3e-4,
            "embed_documents",
        ),
        (
            TextEmbeddingPipeline.Config(
                batch_size=1,
                max_length=192,
                normalize=False,
                pad_to_max_length=False,
                pooling_type=TextEmbeddingPipeline.PoolingType.LAST_TOKEN,
                padding_side="right",
            ),
            0,
            3e-4,
            "embed_documents",
        ),
        (
            TextEmbeddingPipeline.Config(
                batch_size=1,
                max_length=192,
                normalize=False,
                pad_to_max_length=False,
                pooling_type=TextEmbeddingPipeline.PoolingType.MEAN,
                padding_side="right",
            ),
            0,
            3e-4,
            "embed_documents",
        ),
        # Chunk enabled
        # 33 tokens handled by a chunk of 128
        (
            TextEmbeddingPipeline.Config(
                batch_size=1,
                max_length=192,
                normalize=False,
                pad_to_max_length=False,
                pooling_type=TextEmbeddingPipeline.PoolingType.CLS,
                padding_side="right",
            ),
            128,
            3e-4,
            "embed_documents",
        ),
        (
            TextEmbeddingPipeline.Config(
                batch_size=1,
                max_length=192,
                normalize=False,
                pad_to_max_length=False,
                pooling_type=TextEmbeddingPipeline.PoolingType.LAST_TOKEN,
                padding_side="right",
            ),
            128,
            3e-4,
            "embed_documents",
        ),
        (
            TextEmbeddingPipeline.Config(
                batch_size=1,
                max_length=192,
                normalize=False,
                pad_to_max_length=False,
                pooling_type=TextEmbeddingPipeline.PoolingType.MEAN,
                padding_side="right",
            ),
            128,
            3e-4,
            "embed_documents",
        ),
        # 33 tokens handled by 3 chunks of 16
        (
            TextEmbeddingPipeline.Config(
                batch_size=1,
                max_length=180,
                normalize=False,
                pad_to_max_length=False,
                pooling_type=TextEmbeddingPipeline.PoolingType.CLS,
                padding_side="right",
            ),
            16,
            6e-3,
            "embed_documents",
        ),
        (
            TextEmbeddingPipeline.Config(
                batch_size=1,
                max_length=180,
                normalize=False,
                pad_to_max_length=False,
                pooling_type=TextEmbeddingPipeline.PoolingType.LAST_TOKEN,
                padding_side="right",
            ),
            16,
            6e-3,
            "embed_documents",
        ),
        (
            TextEmbeddingPipeline.Config(
                batch_size=1,
                max_length=180,
                normalize=False,
                pad_to_max_length=False,
                pooling_type=TextEmbeddingPipeline.PoolingType.MEAN,
                padding_side="right",
            ),
            16,
            6e-3,
            "embed_documents",
        ),
        # normalize = True, 33 tokens handled by 3 chunks of 16
        (
            TextEmbeddingPipeline.Config(
                batch_size=1,
                max_length=192,
                normalize=True,
                pad_to_max_length=False,
                pooling_type=TextEmbeddingPipeline.PoolingType.CLS,
                padding_side="right",
            ),
            16,
            7e-5,
            "embed_documents",
        ),
        (
            TextEmbeddingPipeline.Config(
                batch_size=1,
                max_length=192,
                normalize=True,
                pad_to_max_length=False,
                pooling_type=TextEmbeddingPipeline.PoolingType.LAST_TOKEN,
                padding_side="right",
            ),
            16,
            7e-5,
            "embed_documents",
        ),
        (
            TextEmbeddingPipeline.Config(
                batch_size=1,
                max_length=192,
                normalize=True,
                pad_to_max_length=False,
                pooling_type=TextEmbeddingPipeline.PoolingType.MEAN,
                padding_side="right",
            ),
            16,
            7e-5,
            "embed_documents",
        ),
        # embed_query
        (
            TextEmbeddingPipeline.Config(
                batch_size=1,
                max_length=192,
                normalize=False,
                pad_to_max_length=False,
                pooling_type=TextEmbeddingPipeline.PoolingType.CLS,
                padding_side="right",
            ),
            16,
            6e-3,
            "embed_query",
        ),
        (
            TextEmbeddingPipeline.Config(
                batch_size=1,
                max_length=192,
                normalize=False,
                pad_to_max_length=False,
                pooling_type=TextEmbeddingPipeline.PoolingType.LAST_TOKEN,
                padding_side="right",
            ),
            16,
            6e-3,
            "embed_query",
        ),
        (
            TextEmbeddingPipeline.Config(
                batch_size=1,
                max_length=192,
                normalize=False,
                pad_to_max_length=False,
                pooling_type=TextEmbeddingPipeline.PoolingType.MEAN,
                padding_side="right",
            ),
            16,
            6e-3,
            "embed_query",
        ),
    ],
)
@pytest.mark.xfail(reason="Ticket - 186607")
@pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 174635")
def test_qwen3_embedding_npu(emb_model, dataset_documents, config, chunk_size, threshold, task):
    NPU_FALLBACK_PROPERTIES = {"NPUW_DEVICES": "CPU", "NPUW_F16IC": "False", "NPUW_LLM_PREFILL_CHUNK_SIZE": chunk_size}

    embeddings_genai_cpu = run_text_embedding_genai(
        emb_model.models_path, dataset_documents, config, task, device="CPU"
    )
    embeddings_genai_npu = run_text_embedding_genai(
        emb_model.models_path, dataset_documents, config, task, device="NPU", properties=NPU_FALLBACK_PROPERTIES
    )
    validate_embedding_results(embeddings_genai_npu, embeddings_genai_cpu, threshold)


@pytest.mark.parametrize("emb_model", ["BAAI/bge-small-en-v1.5"], indirect=True)
def test_embedding_constructors(emb_model):
    models_path = emb_model.models_path

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

    EmbeddingPipeline(models_path, "CPU")
    EmbeddingPipeline(models_path, "CPU", text_embedding_config=TextEmbeddingPipeline.Config())
    EmbeddingPipeline(
        models_path,
        "CPU",
        text_embedding_config=TextEmbeddingPipeline.Config(),
        PERFORMANCE_HINT_NUM_REQUESTS=2,
    )
    EmbeddingPipeline(
        models_path,
        "CPU",
        normalize=True,
        pooling_type=TextEmbeddingPipeline.PoolingType.MEAN,
    )
    EmbeddingPipeline(
        models_path,
        "CPU",
        normalize=True,
        pooling_type=TextEmbeddingPipeline.PoolingType.MEAN,
        PERFORMANCE_HINT_NUM_REQUESTS=2,
    )


@pytest.mark.parametrize("emb_model", EMBEDDINGS_TEST_MODELS, indirect=True)
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
def test_embed_documents(emb_model, dataset_documents, config):
    if (
        sys.platform == "linux"
        and "bge-small-en-v1.5" in str(emb_model)
        and config.normalize
        and config.pooling_type == TextEmbeddingPipeline.PoolingType.CLS
    ):
        pytest.xfail("Random segmentation fault. Ticket 172306")
    models_path = emb_model.models_path
    run_text_embedding_pipeline_with_ref(models_path, dataset_documents, config, "embed_documents")


@pytest.mark.parametrize("emb_model", EMBEDDINGS_TEST_MODELS, indirect=True)
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
def test_embed_query(emb_model, dataset_documents, config):
    models_path = emb_model.models_path
    run_text_embedding_pipeline_with_ref(models_path, dataset_documents[:1], config, "embed_query")


@pytest.fixture(scope="module")
def dataset_embeddings_genai_default_config_refs(emb_model, dataset_documents):
    models_path = emb_model.models_path
    return run_text_embedding_genai(models_path, dataset_documents, None, "embed_documents")


@pytest.mark.parametrize("emb_model", ["mixedbread-ai/mxbai-embed-xsmall-v1"], indirect=True)
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
def test_fixed_shapes_configs(emb_model, dataset_documents, config, dataset_embeddings_genai_default_config_refs):
    models_path = emb_model.models_path

    docs_to_embed = dataset_documents[: config.batch_size] if config.batch_size else dataset_documents
    result = run_text_embedding_genai(models_path, docs_to_embed, config, "embed_documents")

    refs_to_validate = dataset_embeddings_genai_default_config_refs[: config.batch_size] if config.batch_size else dataset_embeddings_genai_default_config_refs
    validate_embedding_results(refs_to_validate, result)


@pytest.mark.parametrize("emb_model", ["mixedbread-ai/mxbai-embed-xsmall-v1"], indirect=True)
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
def test_fixed_shapes_configs_xfail(emb_model, dataset_documents, config, dataset_embeddings_genai_default_config_refs):
    models_path = emb_model.models_path

    docs_to_embed = dataset_documents[: config.batch_size] if config.batch_size else dataset_documents
    result = run_text_embedding_genai(models_path, docs_to_embed, config, "embed_documents")

    refs_to_validate = dataset_embeddings_genai_default_config_refs[: config.batch_size] if config.batch_size else dataset_embeddings_genai_default_config_refs
    validate_embedding_results(refs_to_validate, result)


@pytest.mark.parametrize("emb_model", ["mixedbread-ai/mxbai-embed-xsmall-v1"], indirect=True)
@pytest.mark.parametrize(
    "config",
    [
        TextEmbeddingPipeline.Config(max_length=64, pad_to_max_length=True, batch_size=1),
        TextEmbeddingPipeline.Config(max_length=50, pad_to_max_length=True, batch_size=4),
    ],
)
@pytest.mark.skipif(**should_skip_npuw_tests())
def test_npu_fallback(emb_model, dataset_documents, config, dataset_embeddings_genai_default_config_refs):
    models_path = emb_model.models_path

    pipeline = TextEmbeddingPipeline(models_path, "NPU", config, **NPUW_CPU_PROPERTIES)
    docs_to_embed = dataset_documents[: config.batch_size] if config.batch_size else dataset_documents
    result = pipeline.embed_documents(docs_to_embed)

    refs_to_validate = dataset_embeddings_genai_default_config_refs[: config.batch_size] if config.batch_size else dataset_embeddings_genai_default_config_refs
    validate_embedding_results(refs_to_validate, result)


@pytest.mark.parametrize("rerank_model", [RERANK_TEST_MODELS[0]], indirect=True)
def test_rerank_constructors(rerank_model):
    models_path = rerank_model.models_path

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


@pytest.mark.parametrize("rerank_model", RERANK_TEST_MODELS, indirect=True)
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
def test_rerank_documents(rerank_model, dataset_documents, query, config):
    models_path = rerank_model.models_path
    run_text_rerank_pipeline_with_ref(models_path, query, dataset_documents, config)


# aligned with https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls#updated-transformers-usage
@pytest.mark.parametrize("rerank_model", [QWEN3_RERANK_SEQ_CLS], indirect=True)
@pytest.mark.parametrize("query", ["Which planet is known as the Red Planet?"])
@pytest.mark.parametrize("task", ["Given a web search query, retrieve relevant passages that answer the query"])
@pytest.mark.parametrize(
    "documents",
    [
        [
            "Venus is often called Earth's twin because of its similar size and proximity.",
            "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
            "Jupiter, the largest planet in our solar system, has a prominent red spot.",
            "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
        ]
    ],
)
@pytest.mark.parametrize(
    "config",
    [
        TextRerankPipeline.Config(top_n=4),
    ],
    ids=[
        "top_n=4",
    ],
)
@pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 174635")
def test_qwen3_seq_cls_rerank_documents(rerank_model: OVConvertedModelSchema, query, task, documents, config):
    formatted_query = qwen3_reranker_format_queries(query, task)
    formatted_documents = [qwen3_reranker_format_document(doc) for doc in documents]

    opt_result = run_qwen3_rerank_optimum(
        rerank_model.opt_model,
        rerank_model.hf_tokenizer,
        formatted_query,
        formatted_documents,
        config,
    )
    genai_result = run_text_rerank_genai(
        rerank_model.models_path,
        formatted_query,
        formatted_documents,
        config,
    )

    assert_rerank_results(opt_result, genai_result)

@pytest.mark.parametrize("llm_model", [QWEN3_RERANK], indirect=True)
@pytest.mark.parametrize("query", ["Which planet is known as the Red Planet?"])
@pytest.mark.parametrize("task", ["Given a web search query, retrieve relevant passages that answer the query"])
@pytest.mark.parametrize(
    "documents",
    [
        [
            "Venus is often called Earth's twin because of its similar size and proximity.",
            "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
            "Jupiter, the largest planet in our solar system, has a prominent red spot.",
            "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
        ]
    ],
)
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(
            TextRerankPipeline.Config(top_n=4),
            marks=pytest.mark.skip(
                reason="Qwen3 Reranker different default tokenizer padding side on Win vs Linux: 177405"
            ),
        ),
        TextRerankPipeline.Config(top_n=4, padding_side="left"),
    ],
    ids=[
        "top_n=4",
        "top_n=4, padding_side=left",
    ],
)
@pytest.mark.xfail(condition=(sys.platform == "darwin"), reason="Ticket - 174635")
def test_qwen3_rerank_documents(llm_model: OVConvertedModelSchema, query, task, documents, config):
    formatted_query = qwen3_reranker_format_queries(query, task)
    formatted_documents = [qwen3_reranker_format_document(doc) for doc in documents]

    opt_result = run_qwen3_rerank_optimum(
        llm_model.opt_model,
        llm_model.hf_tokenizer,
        formatted_query,
        formatted_documents,
        config,
    )
    genai_result = run_text_rerank_genai(
        llm_model.models_path,
        formatted_query,
        formatted_documents,
        config,
    )

    assert_rerank_results(opt_result, genai_result)
