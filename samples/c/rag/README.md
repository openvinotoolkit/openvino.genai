# Retrieval Augmented Generation C Samples

This example showcases inference of Retrieval Augmented Generation models from C. The application has limited configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The samples feature C wrappers around `ov::genai::TextEmbeddingPipeline` (and, in a future PR, `ov::genai::TextRerankPipeline`) and use text as an input source.

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

## Run

Follow [Get Started with Samples](https://docs.openvino.ai/2026/get-started/learn-openvino/openvino-samples/get-started-demos.html) to run the sample.

### Text Embeddings Sample (`text_embeddings_c.c`)
- **Description:**
  Demonstrates inference of text embedding models using the OpenVINO GenAI C API. Converts input text into vector embeddings for downstream tasks such as retrieval or semantic search.
- **Run Command:**
  ```sh
  text_embeddings_c <MODEL_DIR> "Document 1" "Document 2"
  ```
Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#text-embeddings-models) for more details.

## Support and Contribution
- For troubleshooting, please refer to the [troubleshooting](https://openvinotoolkit.github.io/openvino.genai/docs/guides/troubleshooting/) section.
- To report a bug or request a feature, [create a GitHub issue](https://github.com/openvinotoolkit/openvino.genai/issues).
- Contributions are welcome! Please see the [Contribution Guide](https://github.com/openvinotoolkit/openvino.genai/blob/master/.github/CONTRIBUTING.md) for details.
