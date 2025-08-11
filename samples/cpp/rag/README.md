# Retrieval Augmented Generation Sample

This example showcases inference of Text Embedding and Text Rerank Models. The application has limited configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `ov::genai::TextEmbeddingPipeline` and `ov::genai::TextRerankPipeline` and uses text as an input source.

## Download and Convert the Model and Tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

Install [../../export-requirements.txt](../../export-requirements.txt) to convert a model.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
```

To export text embedding model run Optimum CLI command:

```sh
optimum-cli export openvino --trust-remote-code --model BAAI/bge-small-en-v1.5 BAAI/bge-small-en-v1.5
```

To export text reranking model run Optimum CLI command:

```sh
optimum-cli export openvino --trust-remote-code --model cross-encoder/ms-marco-MiniLM-L6-v2 cross-encoder/ms-marco-MiniLM-L6-v2
```


## Run

Follow [Get Started with Samples](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/get-started-demos.html) to run the sample.

### 1. Text Embedding Sample (`text_embeddings.cpp`)
- **Description:**
  Demonstrates inference of text embedding models using OpenVINO GenAI. Converts input text into vector embeddings for downstream tasks such as retrieval or semantic search.
- **Run Command:**
  ```sh
  text_embeddings <MODEL_DIR> "Document 1" "Document 2"
  ```
Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#text-embeddings-models) for more details.

### 2. Text Rerank Sample (`text_rerank.cpp`)
- **Description:**
  Demonstrates inference of text rerank models using OpenVINO GenAI. Reranks a list of candidate documents based on their relevance to a query using a cross-encoder or reranker model.
- **Run Command:**
  ```sh
  text_rerank <MODEL_DIR> '<QUERY>' '<TEXT 1>' ['<TEXT 2>' ...]
  ```


# Text Embedding Pipeline Usage

```c++
#include "openvino/genai/rag/text_embedding_pipeline.hpp"

ov::genai::TextEmbeddingPipeline pipeline(models_path, device, config);
std::vector<ov::genai::EmbeddingResult> embeddings = pipeline.embed_documents(documents);
```

# Text Rerank Pipeline Usage

```c++
#include "openvino/genai/rag/text_rerank_pipeline.hpp"

ov::genai::TextRerankPipeline pipeline(models_path, device, config);
std::vector<std::pair<size_t, float>> rerank_result = pipeline.rerank(query, documents);
```
