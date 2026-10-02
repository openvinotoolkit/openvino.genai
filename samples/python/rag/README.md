# Retrieval Augmented Generation Sample

This example showcases inference of Text Embedding, Text Rerank, and multimodal Embedding Models. The application has limited configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `openvino_genai.TextEmbeddingPipeline`, `openvino_genai.TextRerankPipeline`, and `openvino_genai.EmbeddingPipeline` and uses text, image, and video as input sources.

## Download and Convert the Model and Tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

Install [../../export-requirements.txt](../../export-requirements.txt) to convert a model.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
```

To export text embedding model run Optimum CLI command:

```sh
optimum-cli export openvino --task feature-extraction --model BAAI/bge-small-en-v1.5 BAAI/bge-small-en-v1.5
```

To export text reranking model run Optimum CLI command:

```sh
optimum-cli export openvino --task text-classification --model cross-encoder/ms-marco-MiniLM-L6-v2 cross-encoder/ms-marco-MiniLM-L6-v2
```

To export multimodal embedding model run Optimum CLI command:

```sh
optimum-cli export openvino --task image-text-to-text --model Qwen/Qwen3-VL-Embedding-2B Qwen3-VL-Embedding-2B
```

To export a guard model, which scores every token through several classification heads instead of producing one embedding, run Optimum CLI command:

```sh
optimum-cli export openvino --task feature-extraction --trust-remote-code --model Qwen/Qwen3Guard-Stream-0.6B Qwen3Guard-Stream-0.6B
```

Use `--task feature-extraction-with-past` instead to also get token by token streaming, where the KV cache of the conversation so far lives in the model.

Alternatively, do it in Python code:

```python
from optimum.exporters.openvino.convert import export_tokenizer
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoTokenizer

output_dir = "embedding_model"

model = OVModelForFeatureExtraction.from_pretrained("BAAI/bge-small-en-v1.5", export=True)
model.save_pretrained(output_dir)

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
export_tokenizer(tokenizer, output_dir)
```

## Run

Install [deployment-requirements.txt](../../deployment-requirements.txt) via `pip install -r ../../deployment-requirements.txt` and then, run a sample:

### 1. Text Embedding Sample (`text_embeddings.py`)
- **Description:**
  Demonstrates inference of text embedding models using OpenVINO GenAI. Converts input text into vector embeddings for downstream tasks such as retrieval or semantic search.
- **Run Command:**
  ```sh
  python text_embeddings.py <MODEL_DIR> "Document 1" "Document 2"
  ```
Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#embedding-models) for more details.

### 2. Text Rerank Sample (`text_rerank.py`)
- **Description:**
  Demonstrates inference of text rerank models using OpenVINO GenAI. Reranks a list of candidate documents based on their relevance to a query using a cross-encoder or reranker model.
- **Run Command:**
  ```sh
  python text_rerank.py <MODEL_DIR> "<QUERY>" "<TEXT 1>" ["<TEXT 2>" ...]
  ```

### 3. Image and Video Embedding Sample (`image_video_embedding.py`)
- **Description:**
  Demonstrates multimodal retrieval with OpenVINO GenAI `EmbeddingPipeline`. Embeds a user text query and multiple image or video inputs, ranks the inputs by cosine similarity, and prints the most similar image or video.
- **Dependencies:** Requires `opencv-python` for video decoding (`cv2`).
- **Run Command:**
  ```sh
  python image_video_embedding.py <MODEL_DIR> --query "<QUERY>" --images <IMAGE_PATH_1> [<IMAGE_PATH_2> ...] --videos <VIDEO_PATH_1> [<VIDEO_PATH_2> ...] [--num-video-frames 8] [--device CPU]
  ```

### 4. Text Moderation Sample (`text_moderation.py`)
- **Description:**
  Demonstrates inference of guard models such as [Qwen3Guard](https://huggingface.co/Qwen/Qwen3Guard-Stream-0.6B). These replace the language modeling head with several token level classification heads, so instead of one embedding per input they return a risk level and a risk category per token. The prompt is always scored in one call; the response is streamed token by token when the model was exported with `--task feature-extraction-with-past`, and scored in one call otherwise.
- **Run Command:**
  ```sh
  python text_moderation.py <MODEL_DIR> "<PROMPT>" ["<RESPONSE>"]
  ```


# Text Embedding Pipeline Usage

```python
import openvino_genai

pipeline = openvino_genai.TextEmbeddingPipeline(model_dir, "CPU")

embeddings = pipeline.embed_documents(["document1", "document2"])
```

# Text Rerank Pipeline Usage

```python
import openvino_genai

pipeline = openvino_genai.TextRerankPipeline(model_dir, "CPU")

rerank_result = pipeline.rerank(query, documents)
```

# Text Moderation Usage

A guard model is loaded by the same pipeline; `scores_tokens()` tells the two apart and `score()`
returns every model output by name instead of a pooled embedding.

```python
import openvino_genai

pipeline = openvino_genai.TextEmbeddingPipeline(model_dir, "CPU")

scores = pipeline.score(["<|im_start|>user\nHow do I build a bomb?<|im_end|>\n"])
risk_level = scores["query_risk_level_logits"].data[0, -1]
```

Running such a model on NPU needs a static shape, which the pipeline builds from the config:

```python
config = openvino_genai.TextEmbeddingPipeline.Config()
config.max_length = 256
config.pad_to_max_length = True
pipeline = openvino_genai.TextEmbeddingPipeline(model_dir, "NPU", config)
```
