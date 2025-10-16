# Retrieval Augmented Generation Sample

This example showcases inference of Text Embedding Models. The application limited configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `TextEmbeddingPipeline` and uses text as an input source.

## Download and Convert the Model and Tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

Install [../../export-requirements.txt](../../export-requirements.txt) to convert a model.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
optimum-cli export openvino --trust-remote-code --model BAAI/bge-small-en-v1.5 BAAI/bge-small-en-v1.5
```

## Sample Descriptions
### Common information

Compile GenAI JavaScript bindings archive first using [the instructions](../../../src/js/README.md#build-bindings).

Run `npm install` and the example will be ready to run.

Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#text-embeddings-models) for more details.

### Text embedding (`text_embeddings`)
- **Description:** This is basic text embedding using a causal embedding model.
- **Recommended models:** BAAI/bge-small-en-v1.5, etc
- **Main Feature:** Demonstrates simple text embedding.
- **Run Command:**
  ```bash
  node text_embeddings.js model_dir text1 text2
  ```