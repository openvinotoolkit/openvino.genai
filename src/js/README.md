# OpenVINO™ GenAI Node.js bindings (preview)

## DISCLAIMER

This is preview version, do not use it in production!

## Install and Run

### Requirements

- Node.js v21+
- Tested on Ubuntu, another OS didn't tested yet

### Build Bindings

TODO: Add instructions

### Perform Test Run

- To run sample you should have prepared model.
  Use this instruction [to download model](https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/cpp/chat_sample/README.md#download-and-convert-the-model-and-tokenizers)
- Go to [samples/js](../../samples/js/)
- Run `node app.js`, you should see: `User Prompt: ...`

### Using as npm Dependency

To use this package locally use `npm link` in this directory
and `npm link genai-node` in the folder where you want add this package as dependency

To extract this package and use it as distributed npm package run `npm package`.
This command creates archive that you may use in your projects.
