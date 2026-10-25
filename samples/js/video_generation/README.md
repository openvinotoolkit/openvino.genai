# Video Generation JavaScript Sample

This directory contains a JavaScript sample that shows how to generate a video from a text prompt with LTX-Video and LTX-2 family models via the `openvino-genai-node` package. The sample doesn't have many configuration options to encourage the reader to explore and modify the source code. For example, change the device for inference to GPU.

Sample files:
 - [`text2video.js`](./text2video.js) demonstrates basic usage of the `Text2VideoPipeline` (text-to-video) with a step callback, and saves every generated frame as a BMP file using `bmp-js`. For models that generate audio (LTX-2 family), it also saves the audio track as a WAV file using `node-wav`.

Users can change the sample code and play with the following generation parameters:

- Change width or height of the generated video
- Change the number of frames (`num_frames`) and the frame rate (`frame_rate`)
- Generate multiple videos per prompt (`num_videos_per_prompt`)
- Adjust the number of inference steps (`num_inference_steps`)
- Change the negative prompt or the guidance scale (`guidance_scale`)

## Download and convert the model

The `--upgrade-strategy eager` option is needed to ensure `optimum-intel` is upgraded to the latest version.

It's not required to install [../../export-requirements.txt](../../export-requirements.txt) for deployment if the model has already been exported.

```sh
pip install --upgrade-strategy eager -r ../../export-requirements.txt
```

Then, run the export with Optimum CLI:

```sh
optimum-cli export openvino --model Lightricks/LTX-Video --task text-to-video --weight-format int8 ltx_video_ov/INT8
```

LTX-2.3 also generates a synchronized audio track:

```sh
optimum-cli export openvino --model diffusers/LTX-2.3-Diffusers --task text-to-video --weight-format int8 --trust-remote-code ltx_2_3_ov/INT8
```

## Run

From the `samples/js` directory, install dependencies (if not already done):

```bash
npm install
```

If you use the master branch, you may need to [build openvino-genai-node from source](../../../src/js/README.md#build-bindings) first.

### Run the text-to-video sample:

```bash
node video_generation/text2video.js ltx_video_ov/INT8 "A woman with long brown hair smiles at another woman with long blonde hair" 25
```

The optional last argument is the number of frames to generate (161 by default). The frames are saved as `frame_000.bmp`, `frame_001.bmp`, and so on in the `genai_video_frames` directory. To combine them into a video, you can use a tool such as `ffmpeg`:

```bash
ffmpeg -framerate 25 -i genai_video_frames/frame_%03d.bmp genai_video.mp4
```

For models that generate audio (LTX-2 family), the sample also writes `genai_audio.wav`. To combine the frames and the audio into one file:

```bash
ffmpeg -framerate 25 -i genai_video_frames/frame_%03d.bmp -i genai_audio.wav genai_video.mp4
```
