# OpenVINO™ GenAI Node.js bindings (preview)

## DISCLAIMER

This is preview version, do not use it in production!

## Quick Start

Install the **genai-node** package:
```bash
npm install genai-node
```

Use the **genai-node** package:
```js
import { LLMPipeline } from 'genai-node';

const pipe = await LLMPipeline(MODEL_PATH, device);

const input = 'What is the meaning of life?';
const config = { 'max_new_tokens': 100 };

await pipe.startChat();
const result = await pipe.generate(input, config, streamer);
await pipe.finishChat();

// Output all generation result
console.log(result);

function streamer(subword) {
  process.stdout.write(subword);
}
```

### Build from source

To build binaries from source, follow the instructions in the [BUILD.md](https://github.com/openvinotoolkit/openvino.genai/blob/master/src/js/BUILD.md) file.

### Requirements

- Node.js v21+
- Tested on Ubuntu and Windows, another OS didn't tested yet
