# Retrieval Augmented Generation Sample

This example showcases inference of Text Embedding Models. The application limited configuration configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `openvino_genai.TextEmbeddingPipeline` and uses text as an input source.

## Download and Convert the Model and Tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

Install [../../export-requirements.txt](../../export-requirements.txt) to convert a model.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
```

Then, run the export with Optimum CLI:

```sh
optimum-cli export openvino --trust-remote-code --model BAAI/bge-small-en-v1.5 BAAI/bge-small-en-v1.5
```

Alternatively, do it in Python code:

```python
from optimum.exporters.openvino.convert import export_tokenizer
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoTokenizer

output_dir = "embedding_model"

model = OVModelForFeatureExtraction.from_pretrained("BAAI/bge-small-en-v1.5", export=True, trust_remote_code=True)
model.save_pretrained(output_dir)

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
export_tokenizer(tokenizer, output_dir)
```

## Run

Install [deployment-requirements.txt](../../deployment-requirements.txt) via `pip install -r ../../deployment-requirements.txt` and then, run a sample:

`python text_embeddings.py BAAI/bge-small-en-v1.5 "Document 1" "Document 2"`

Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#text-embeddings-models) for more details.

# Text Embedding Pipeline Usage

```python
import openvino_genai

pipeline = openvino_genai.TextEmbeddingPipeline(model_dir, "CPU")

embeddings = pipeline.embed_documents(["document1", "document2"])
```
