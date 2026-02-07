# OpenVINO GenAI Video Generation C++ Samples

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

### Text to Video Sample (`text2video.cpp`)

- **Description:**
  Basic video generation using a text-to-video model. This sample demonstrates how to generate videos from text prompts using the OpenVINO GenAI Text2VideoPipeline. The LTX-Video model is recommended for this sample.

  Recommended models: Lightricks/LTX-Video

- **Main Feature:** Generate videos from text descriptions with customizable parameters.

- **Run Command:**
  ```bash
  ./text2video model_dir prompt
  ```

  Example:
  ```bash
  ./text2video ltx_video_ov/INT8 "A woman with long brown hair and light skin smiles at another woman with long blonde hair"
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

```cpp
ov::genai::Text2VideoPipeline pipe(models_path, device);

auto callback = [&](size_t step, size_t num_steps, ov::Tensor& latent) -> bool {
   std::cout << "Generation step: " << step + 1 << " / " << num_steps << std::endl;
   ov::Tensor video = pipe.decode(latent).video; // get intermediate video tensor
   if (your_condition) // return true if you want to interrupt video generation
      return true;
   return false;
};

ov::Tensor video = pipe.generate(prompt,
   /* other generation properties */
   ov::genai::callback(callback)
).video;
```

## Troubleshooting

### LTX-Video Model Constraints

> [!NOTE]
> The LTX-Video model works best on:
> - Resolutions divisible by 32 (e.g., 480x704, 512x512, 720x1280)
> - Number of frames divisible by 8 + 1 (e.g., 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121, 161, 257)
> - At least 2 inference steps (1 step may produce artifacts)
> - Best quality achieved with resolutions under 720x1280 and number of frames below 257

## Support and Contribution
- For troubleshooting, consult the [OpenVINO documentation](https://docs.openvino.ai).
- To report issues or contribute, visit the [GitHub repository](https://github.com/openvinotoolkit/openvino.genai).
