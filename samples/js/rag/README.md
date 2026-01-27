# Retrieval Augmented Generation Sample

This example showcases inference of Text Embedding and Text Rerank Models. The application has limited configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `TextEmbeddingPipeline` and `TextRerankPipeline`, which use text as an input source.

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

## Run

Compile GenAI JavaScript bindings archive first using [the instructions](../../../src/js/README.md#build-bindings).

Run `npm install` and the example will be ready to run.

### 1. Text Embedding Sample (`text_embeddings.js`)
- **Description:**
  Demonstrates inference of text embedding models using OpenVINO GenAI. Converts input text into vector embeddings for downstream tasks such as retrieval or semantic search.
- **Run Command:**
  ```sh
  node text_embeddings.js <MODEL_DIR> "Document 1" "Document 2"
  ```
Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#text-embeddings-models) for more details.

### 2. Text Rerank Sample (`text_rerank.js`)
- **Description:**
  Demonstrates inference of text rerank models using OpenVINO GenAI. Reranks a list of candidate documents based on their relevance to a query using a cross-encoder or reranker model.
- **Run Command:**
  ```sh
  node text_rerank.js <MODEL_DIR> "<QUERY>" "<TEXT 1>" ["<TEXT 2>" ...]
  ```

# Text Embedding Pipeline Usage

```js
import { TextEmbeddingPipeline } from 'openvino-genai-node';

const pipeline = await TextEmbeddingPipeline(model_dir, "CPU");

const embeddings = await pipeline.embedDocuments(["document1", "document2"]);
```

# Text Rerank Pipeline Usage

```js
import { TextRerankPipeline } from 'openvino-genai-node';

const pipeline = await TextRerankPipeline(modelPath, { device: "CPU" });

const rerankResult = await pipeline.rerank(query, documents);
```
