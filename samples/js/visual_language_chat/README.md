# JS vlm_chat_sample that supports VLM models

This example showcases inference of text-generation Vision Language Models (VLMs): `miniCPM-V-2_6` and other models with the same signature. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `openvino-genai-node.VLMPipeline` and configures it for the chat scenario.

There are two sample files:
 - [`visual_language_chat.js`](./visual_language_chat.js) demonstrates basic usage of the VLM pipeline which supports accelerated inference using prompt lookup decoding.

## Install JS dependencies

Install Node.js dependencies from `samples/js`:

```sh
cd samples/js
npm install
```

## Download and convert the model and tokenizers

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

Install [../../export-requirements.txt](../../export-requirements.txt) to convert a model.

```sh
pip install --upgrade-strategy eager -r ../export-requirements.txt
```

Then, run the export with Optimum CLI:

```sh
optimum-cli export openvino --model openbmb/MiniCPM-V-2_6 --trust-remote-code MiniCPM-V-2_6
```

## Run image-to-text chat sample

[This image](https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/d5fbbd1a-d484-415c-88cb-9986625b7b11) can be used as a sample image.

```sh
cd samples/js
node visual_language_chat/visual_language_chat.js ./miniCPM-V-2_6/ 319483352-d5fbbd1a-d484-415c-88cb-9986625b7b11.jpg
```

See https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md#supported-models for the list of supported models.
