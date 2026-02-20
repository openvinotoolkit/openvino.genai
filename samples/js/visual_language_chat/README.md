# JS vlm_chat_sample that supports VLM models

This example showcases inference of text-generation Vision Language Models (VLMs): `miniCPM-V-2_6` and other models with the same signature. The application doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU. The sample features `openvino-genai-node.VLMPipeline` and configures it for the chat scenario.

There are three sample files:
 - [`visual_language_chat.js`](./visual_language_chat.js) demonstrates basic usage of the VLM pipeline.
 - [`video_to_text_chat.js`](./video_to_text_chat.js) demonstrates video to text usage of the VLM pipeline.

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
pip install --upgrade-strategy eager -r ../../export-requirements.txt
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

## Run video-to-text chat sample

A model that supports video input is required to run this sample, for example `llava-hf/LLaVA-NeXT-Video-7B-hf`.

[This video](https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4) can be used as a sample video.

```sh
cd samples/js
node visual_language_chat/video_to_text_chat.js ./LLaVA-NeXT-Video-7B-hf/ sample_demo_1.mp4
```

Supported models with video input are listed in [this section](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/#use-image-or-video-tags-in-prompt).

Discrete GPUs (dGPUs) usually provide better performance compared to CPUs. It is recommended to run larger models on a dGPU with 32GB+ RAM.
Modify the source code to change the device for inference to the GPU.
