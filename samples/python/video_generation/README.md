# OpenVINO GenAI Video Generation Python Samples

These samples showcase the use of OpenVINO's inference capabilities for video generation tasks. The sample features `openvino_genai.Text2VideoPipeline` for generating videos from text prompts using models like LTX-Video.
The applications don't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU.

## Table of Contents
1. [Download and Convert the Model](#download-and-convert-the-model)
2. [Sample Descriptions](#sample-descriptions)
3. [Troubleshooting](#troubleshooting)
4. [Support and Contribution](#support-and-contribution)

## Download and Convert the Model

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.
Install [../../export-requirements.txt](../../export-requirements.txt) if model conversion is required.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
```

Then, run the export with Optimum CLI:

```sh
optimum-cli export openvino --model Lightricks/LTX-Video --task text-to-video --weight-format int8 ltx_video_ov/INT8
```

Alternatively, do it in Python code. If NNCF is installed, the model will be compressed to INT8 automatically.

```python
from optimum.intel.openvino import OVLTXPipeline

output_dir = "ltx_video_ov/INT8"

pipeline = OVLTXPipeline.from_pretrained("Lightricks/LTX-Video", export=True, compile=False, load_in_8bit=True)
pipeline.save_pretrained(output_dir)
```

## Sample Descriptions

### Common Information

Follow [Get Started with Samples](https://docs.openvino.ai/2025/get-started/learn-openvino/openvino-samples/get-started-demos.html) to get common information about OpenVINO samples.
Follow [build instruction](../../../src/docs/BUILD.md) to build GenAI samples.

GPUs usually provide better performance compared to CPUs. Modify the source code to change the device for inference to the GPU.

Install [../../deployment-requirements.txt](../../deployment-requirements.txt) to run samples:
```sh
pip install --upgrade-strategy eager -r ../../deployment-requirements.txt
```

### Text to Video Sample (`text2video.py`)

- **Description:**
  Basic video generation using a text-to-video model. This sample demonstrates how to generate videos from text prompts using the OpenVINO GenAI Text2VideoPipeline. The LTX-Video model is recommended for this sample.

  Recommended models: Lightricks/LTX-Video

- **Main Feature:** Generate videos from text descriptions with customizable parameters.

- **Run Command:**
  ```bash
  python text2video.py model_dir prompt [--device DEVICE] [--output OUTPUT]
  ```

  Example:
  ```bash
  python text2video.py ./ltx_video_ov/INT8 "A woman with long brown hair and light skin smiles at another woman with long blonde hair"
  ```

The sample will generate a video file `genai_video.avi` in the current directory.

Users can modify the source code to experiment with different generation parameters:
- Change width or height of generated video
- Change number of frames
- Generate multiple videos per prompt
- Adjust number of inference steps
- Play with guidance scale (improves quality when > 1)
- Add negative prompt when guidance scale > 1
- Adjust frame rate

#### Run with threaded callback

You can also implement a callback function that runs in a separate thread. This allows for parallel processing, enabling you to interrupt generation early if intermediate results are satisfactory or to add logs.

Please find the template of the callback usage below:

```python
pipe = openvino_genai.Text2VideoPipeline(model_dir, device)

def callback(step, num_steps, latent):
   print(f"Video generation step: {step + 1} / {num_steps}")
   if your_condition:  # return True if you want to interrupt video generation
      return True
   return False

video = pipe.generate(
   prompt,
   callback=callback
).video
```

## Troubleshooting

### LTX-Video Model Constraints

> [!NOTE]
> The LTX-Video model works best on:
> - Resolutions divisible by 32 (e.g., 480x704, 512x512, 720x1280)
> - Number of frames divisible by 8 + 1 (e.g., 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121, 161, 257)
> - At least 2 inference steps (1 step may produce artifacts)
> - Best quality achieved with resolutions under 720x1280 and number of frames below 257

### OpenCV Installation

If you encounter issues with OpenCV when running the samples, ensure it's properly installed:

```sh
pip install opencv-python==4.12.0.88
```

This dependency is included in [../../deployment-requirements.txt](../../deployment-requirements.txt).

### Unicode characters encoding error on Windows

Example error:
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u25aa' in position 0: character maps to <undefined>
```

If you encounter the error described in the example when sample is printing output to the Windows console, it is likely due to the default Windows encoding not supporting certain Unicode characters. To resolve this:
1. Enable Unicode characters for Windows cmd - open `Region` settings from `Control panel`. `Administrative`->`Change system locale`->`Beta: Use Unicode UTF-8 for worldwide language support`->`OK`. Reboot.
2. Enable UTF-8 mode by setting environment variable `PYTHONIOENCODING="utf8"`.

## Support and Contribution
- For troubleshooting, consult the [OpenVINO documentation](https://docs.openvino.ai).
- To report issues or contribute, visit the [GitHub repository](https://github.com/openvinotoolkit/openvino.genai).
