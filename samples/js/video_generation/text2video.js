// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { basename, join } from "node:path";
import { mkdir, rm, writeFile } from "node:fs/promises";
import yargs from "yargs/yargs";
import { hideBin } from "yargs/helpers";
import { encode } from "node-wav";
import { addon as ov } from "openvino-node";
import { Text2VideoPipeline } from "openvino-genai-node";
import { saveAsBMP } from "../image_utils.js";

async function main() {
  const argv = await yargs(hideBin(process.argv))
    .scriptName(basename(process.argv[1]))
    .command(
      "$0 <model_dir> <prompt> [num_frames]",
      "Run Text2Video pipeline and save the generated frames as BMP files and the audio as a WAV file",
      (yargsBuilder) =>
        yargsBuilder
          .positional("model_dir", {
            type: "string",
            describe: "Path to the converted video generation model directory",
            demandOption: true,
          })
          .positional("prompt", {
            type: "string",
            describe: "Prompt to generate the video from",
            demandOption: true,
          })
          .positional("num_frames", {
            type: "number",
            describe: "Number of frames to generate",
            default: 161,
          })
    )
    .strict()
    .help()
    .parse();

  const device = "CPU"; // GPU can be used as well
  const pipeline = await Text2VideoPipeline(argv.model_dir, device);

  function callback(step, numSteps) {
    console.log(`Generation step ${step + 1} / ${numSteps}`);
    return false;
  }

  const { video, audio, audioSampleRate, perfMetrics } = await pipeline.generate(argv.prompt, {
    negative_prompt: "worst quality, inconsistent motion, blurry, jittery, distorted",
    height: 480,
    width: 704,
    num_frames: argv.num_frames,
    num_inference_steps: 25,
    num_videos_per_prompt: 1,
    frame_rate: 25,
    guidance_scale: 3,
    callback,
  });

  // The video tensor has shape [num_videos, num_frames, H, W, 3]; save each frame of the first video
  const [, numFrames, height, width, channels] = video.getShape();
  const frameSize = height * width * channels;
  const outputDir = "genai_video_frames";
  // Start from an empty directory, so frames of a previous run don't end up in the video
  await rm(outputDir, { recursive: true, force: true });
  await mkdir(outputDir, { recursive: true });
  for (let i = 0; i < numFrames; i++) {
    // slice() copies the frame, a Tensor can't be created from a view with a nonzero byteOffset
    const frameData = video.data.slice(i * frameSize, (i + 1) * frameSize);
    const frame = new ov.Tensor("u8", [1, height, width, channels], frameData);
    await saveAsBMP(join(outputDir, `frame_${String(i).padStart(3, "0")}.bmp`), frame);
  }
  console.log(`Saved ${numFrames} frames to ${outputDir}`);

  // Models that generate audio (LTX-2) return it as a separate track. To combine the frames and the audio
  // into one video file:
  //   ffmpeg -framerate 25 -i genai_video_frames/frame_%03d.bmp -i genai_audio.wav genai_video.mp4
  if (audioSampleRate) {
    // The audio tensor has shape [num_videos, channels, samples]; save the track of the first video
    const [, numChannels, numSamples] = audio.getShape();
    const channelData = Array.from({ length: numChannels }, (_, c) =>
      audio.data.subarray(c * numSamples, (c + 1) * numSamples),
    );
    await writeFile("genai_audio.wav", encode(channelData, { sampleRate: audioSampleRate, float: true }));
    console.log(
      `Saved genai_audio.wav (${numSamples} samples, ${numChannels} channels @ ${audioSampleRate} Hz)`,
    );
  }

  console.log("Performance metrics:");
  console.log(`  Load time: ${perfMetrics.getLoadTime().toFixed(2)} ms`);
  console.log(`  Generate duration: ${perfMetrics.getGenerateDuration().toFixed(2)} ms`);
  console.log(`  Transformer duration: ${perfMetrics.getTransformerInferDuration().mean.toFixed(2)} ms`);
  console.log(`  VAE decoder duration: ${perfMetrics.getVaeDecoderInferDuration().toFixed(2)} ms`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
