# Retrieval Augmented Generation Sample

This example showcases inference of Text Embedding Models. The application has limited configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `ov::genai::TextEmbeddingPipeline` and uses text as an input source.

## Download and Convert the Model and Tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

Install [../../export-requirements.txt](../../export-requirements.txt) to convert a model.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
optimum-cli export openvino --trust-remote-code --model BAAI/bge-small-en-v1.5 BAAI/bge-small-en-v1.5
```

## Run

Follow [Get Started with Samples](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/get-started-demos.html) to run the sample.

`text_embeddings BAAI/bge-small-en-v1.5 "Document 1" "Document 2"`

Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#text-embeddings-models) for more details.

# Text Embedding Pipeline Usage

```c++
#include "openvino/genai/rag/text_embedding_pipeline.hpp"

ov::genai::TextEmbeddingPipeline pipeline(models_path, device, config);
std::vector<ov::genai::EmbeddingResult> embeddings = pipeline.embed_documents(documents);
```
