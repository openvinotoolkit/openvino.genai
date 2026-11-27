# Classification Sample

This example showcases `openvino_genai.ClassificationPipeline`, which scores a prompt with optional images
against the labels of a classification model. The application has limited configuration options to encourage
the reader to explore and modify the source code. For example, change the device for inference to GPU.

## Download and Convert the Model

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
optimum-cli export openvino --task text-classification --model openbmb/MiniCPM-OCR-Jev --weight-format int4 MiniCPM-OCR-Jev
```

## Run

Follow [Get Started with Samples](https://docs.openvino.ai/2026/get-started/learn-openvino/openvino-samples/get-started-demos.html) to run the sample.

```sh
python classification.py MiniCPM-OCR-Jev '<PROMPT>' original.png rendered.png
```

The prompt is passed to the model as is. For MiniCPM-OCR-Jev, build it as described in the
[model card](https://huggingface.co/openbmb/MiniCPM-OCR-Jev) and pass the original region and the rendered
OCR text as images. The score of label `1` is the probability that the OCR text is correct.

Refer to the [Supported Models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#classification-models) for more details.
