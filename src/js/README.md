# OpenVINO™ GenAI for Node.js

<div align="center">

![OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai/blob/master/site/static/img/openvino-genai-logo-gradient.svg?raw=1)

[<b>Getting Started</b>](#getting-started) •
[<b>Quick install</b>](#quick-install) •
[<b>AI Scenarios</b>](#supported-generative-ai-scenarios) •
[<b>Documentation</b>](https://openvinotoolkit.github.io/openvino.genai/docs/bindings/node-js/) •
[<b>Supported Models</b>](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/)

[![GitHub Release](https://img.shields.io/github/v/release/openvinotoolkit/openvino.genai?color=green)](https://github.com/openvinotoolkit/openvino.genai/releases)
[![NPM Version](https://img.shields.io/npm/v/openvino-genai-node)](https://www.npmjs.com/package/openvino-genai-node?activeTab=versions)
[![OS](https://img.shields.io/badge/OS-Linux_|_Windows_|_macOS-blue)](#supported-platforms)
[![NPM License](https://img.shields.io/npm/l/openvino-genai-node)](#license)

![](https://github.com/openvinotoolkit/openvino.genai/blob/master/site/static/img/openvino-genai-workflow.svg?raw=1)

</div>

**openvino-genai-node** brings [**OpenVINO™ GenAI**](https://openvinotoolkit.github.io/openvino.genai/) with the most popular **Generative AI** model pipelines to **Node.js**. Run LLMs, VLMs, image and speech generation, and RAG locally with hardware-accelerated inference on top of [**OpenVINO Runtime**](https://www.npmjs.com/package/openvino-node).

This library is friendly to PC and laptop execution, and optimized for resource consumption. Prebuilt native addons, the OpenVINO runtime and tokenization are fetched during `npm install`. You do not need a separate OpenVINO SDK install for typical use.

## Key Features and Benefits:
 - 📦 Pre-built Generative AI Pipelines: Ready-to-use pipelines for text generation (LLMs), visual language models (VLMs), image generation (Diffusers), speech recognition (Whisper), and speech generation (SpeechT5). See all supported [AI scenarios](#supported-generative-ai-scenarios).
 - 👣 Minimal Footprint: Smaller binary size and reduced memory footprint compared to other frameworks.
 - 📥 Plug-and-play install: Prebuilt native addons and OpenVINO runtime are downloaded on `npm install` — no manual OpenVINO installation for typical use.
 - 🖥️ In-process inference: Run models directly from Node.js in your application process — no separate inference service or access tokens.
 - 🚀 Performance Optimization: Hardware-specific optimizations for CPU, GPU, and NPU devices. See [optimization techniques](https://openvinotoolkit.github.io/openvino.genai/docs/category/optimization-techniques) in the GenAI docs.
 - 👨‍💻 Programming Language Support: APIs aligned with Python/C++ where possible — async functions and ESM.
 - 🗜️ Model Compression: Support for 8-bit and 4-bit weight compression, including embedding layers.
 - 🎓 Advanced Inference Capabilities: In-place KV-cache, dynamic quantization, speculative sampling, and more.
 - 🎨 Wide Model Compatibility: Support for popular models including Llama, Mistral, Phi, Qwen, Stable Diffusion, Flux, Whisper, and others. Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/) for more details.

## Quick install

```bash
npm install openvino-genai-node
```

## Requirements

Node.js **≥ 21**. Refer to the [supported platforms](#supported-platforms) for more details.

## Supported platforms

| **OS** | **x64** | **arm64** |
| ------ | ------- | --------- |
| **Windows** | ✅ | ❌ |
| **Linux** | ✅ | ✅ |
| **macOS** | ❌ | ✅ |

Prebuilt binaries are downloaded for your OS/arch during `npm install`. If a platform is unsupported, installation or loading the native addon will fail — build from source per [BUILD.md](https://github.com/openvinotoolkit/openvino.genai/blob/master/src/js/BUILD.md).

## Getting started

### Model preparation

Models must be converted to **OpenVINO IR** (for example with [Optimum Intel](https://github.com/huggingface/optimum-intel)). See the [GenAI documentation](https://openvinotoolkit.github.io/openvino.genai/docs/guides/model-preparation/convert-to-openvino) for model preparation and supported architectures.

### Minimal example (LLM)

Import a pipeline from `openvino-genai-node`, point it at a prepared model directory, then call `generate` or `stream`:

```js
import { LLMPipeline } from "openvino-genai-node";

const modelPath = "/path/to/ov/model"; // directory with OpenVINO IR + tokenizer files
const device = "CPU"; // or "GPU", "NPU" where supported
const config = { max_new_tokens: 100 };

const pipe = await LLMPipeline(modelPath, device);

// One-shot generation
const result = await pipe.generate("What is OpenVINO?", config);
console.log(result);

// Streaming tokens to stdout
for await (const chunk of pipe.stream("What is OpenVINO?", config)) {
  process.stdout.write(chunk);
}
```

More runnable examples: [**samples/js**](https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/js).

## Supported Generative AI Scenarios

The OpenVINO™ GenAI Node.js package supports the following scenarios (details and model lists are in the linked use-case docs):
- [Text generation using Large Language Models (LLMs)](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/text-generation/) - Chat with local Llama, Phi, Qwen and other models
- [Image processing using Visual Language Model (VLMs)](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/) - Analyze images/videos with LLaVa, MiniCPM-V and other models
- [Image generation using Diffusers](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-generation/) - Generate images with Stable Diffusion & Flux models
- [Speech recognition using Whisper](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/speech-recognition/) - Convert speech to text using Whisper models
- [Speech generation using SpeechT5](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/speech-generation/) - Convert text to speech using SpeechT5 TTS models
- [Semantic search using Text Embedding](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/text-embedding) - Compute embeddings for documents and queries to enable efficient retrieval in RAG workflows
- [Text Rerank for Retrieval-Augmented Generation (RAG)](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/text-rerank) - Analyze the relevance and accuracy of documents and queries for your RAG workflows

## Contributing

Contributions are welcome in the [openvino.genai](https://github.com/openvinotoolkit/openvino.genai) repository.

- Read the project [Contributing guide](https://github.com/openvinotoolkit/openvino.genai/blob/master/.github/CONTRIBUTING.md) (PR process, code quality, branching).
- **Node.js bindings:** build and test locally — [BUILD.md](https://github.com/openvinotoolkit/openvino.genai/blob/master/src/js/BUILD.md) (native addon + TypeScript wrapper in `src/js`).
- **JavaScript changes:** from `src/js`, run `npm run lint`, `npm run build`, and `npm test` (tests require Python 3.10+ and model setup described in BUILD.md).
- **Samples:** extend or fix examples under [samples/js](https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/js).
- Open issues or pull requests on GitHub; use the repository PR template and fork-based workflow described in the contributing guide.

## FAQ

### Can I use OpenVINO GenAI in the browser?

Not with **openvino-genai-node** — this package targets Node.js (server-side) only. For in-browser inference, use [Transformers.js](https://www.npmjs.com/package/@huggingface/transformers).

### Can I use OpenVINO GenAI for Node.js as an ML inference server?

No. **openvino-genai-node** does not expose REST or gRPC APIs; it runs models in-process inside your Node.js application. For a dedicated model-serving layer with HTTP/gRPC, deploy [**OpenVINO Model Server (OVMS)**](https://docs.openvino.ai/2026/model-server/ovms_what_is_openvino_model_server.html) and call it from Node (or other clients).

## License

The OpenVINO™ GenAI repository is licensed under [Apache License Version 2.0](https://github.com/openvinotoolkit/openvino.genai/blob/master/LICENSE).
By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.
