---
sidebar_position: 1
sidebar_label: Node.js
description: Node.js bindings provide JavaScript/TypeScript API.
---

# Node.js Bindings for OpenVINO™ GenAI

OpenVINO GenAI provides Node.js bindings that enable you to use generative AI pipelines in JavaScript and TypeScript applications.

:::warning API Coverage
Node.js bindings currently provide a subset of the full OpenVINO GenAI API available in C++ and Python. The focus is on core text generation (`LLMPipeline`), vision language models (`VLMPipeline`), text embedding (`TextEmbeddingPipeline`), and text reranking (`TextRerankPipeline`) functionality.
:::

## Supported Pipelines and Features

Node.js bindings currently support:

- `LLMPipeline`: Text generation with Large Language Models
  - Chat mode with conversation history
  - Streaming support
  - Batch generation
  - Multiple sampling strategies (greedy, beam search)
  - Structured output
  - ReAct agent support
- `VLMPipeline`: Vision Language Model inference for multimodal tasks
  - Process images and videos with text prompts
  - Chat mode with conversation history
  - Streaming support
- `TextEmbeddingPipeline`: Generate text embeddings for semantic search and RAG applications
- `TextRerankPipeline`: Rerank documents by semantic relevance for RAG applications
  - Configurable top-n results
- `Tokenizer`: Fast tokenization / detokenization and chat prompt formatting
  - Encode strings into token id and attention mask tensors
  - Decode token sequences
  - Apply chat template
  - Access special tokens (BOS/EOS/PAD)
  - Supports paired input

## Installation

To install OpenVINO GenAI for Node.js, refer to the [Install Guide](https://docs.openvino.ai/2025/get-started/install-openvino.html).

## Quick Start

:::tip Model Preparation
Before using LLMPipeline, you need to convert your model to OpenVINO IR format.
See [Model Preparation](/docs/category/model-preparation) for details.
:::

After installation, you can start using OpenVINO GenAI in your Node.js projects:

```js
import { LLMPipeline } from "openvino-genai-node";

async function main() {
  const modelPath = "/path/to/ov/model";
  const device = "CPU";
  const pipe = await LLMPipeline(modelPath, device);

  const input = "What is OpenVINO?";
  const config = { max_new_tokens: 100 };

  for await (const chunk of pipe.stream(input, config)) {
    process.stdout.write(chunk);
  }
}

main();
```

## Next Steps

- Check out [Code Samples](/docs/samples)
- Review [Supported Models](/docs/supported-models)
- Explore [Use Cases](/docs/category/use-cases)
- Browse the [Node.js bindings source](https://github.com/openvinotoolkit/openvino.genai/tree/master/src/js)
- View the [NPM package](https://www.npmjs.com/package/openvino-genai-node)

## Troubleshooting

### Module Not Found Errors

If you encounter errors like `Cannot find module 'openvino-genai-node'`:

1. Verify installation: `npm list openvino-genai-node`
2. Check Node.js version: `node --version`
3. Ensure ES modules are enabled: add `"type": "module"` in your `package.json`

### Version Compatibility Issues

If you encounter errors related to shared libraries or ABI compatibility:

1. Ensure both `openvino-node` and `openvino-genai-node` are the same version
2. If building from source, rebuild both OpenVINO and OpenVINO GenAI bindings
3. Check that your system meets the requirements for your platform

For more help, refer to the [OpenVINO GenAI GitHub repository](https://github.com/openvinotoolkit/openvino.genai) or open an issue.
