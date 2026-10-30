// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { basename } from "node:path";
import yargs from "yargs/yargs";
import { hideBin } from "yargs/helpers";
import { Text2VideoPipeline } from "openvino-genai-node";
import { saveVideo } from "../video_utils.js";
import { saveAudio } from "../wav_utils.js";

async function main() {
  const argv = await yargs(hideBin(process.argv))
    .scriptName(basename(process.argv[1]))
    .command(
      "$0 <model_dir> <prompt> [num_frames]",
      "Run Text2Video pipeline and save the generated video as an AVI file and the audio as a WAV file",
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

  const frameRate = 25;

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
    callback,
    frame_rate: frameRate,
    guidance_scale: 3,
  });

  saveVideo("genai_video.avi", video, frameRate);
  // Models that generate audio (LTX-2) return it as a separate track. To combine both into one file:
  //   ffmpeg -i genai_video.avi -i genai_audio.wav genai_video.mp4
  if (audioSampleRate) {
    await saveAudio("genai_audio.wav", audio, audioSampleRate);
  }

  console.log("\nPerformance metrics:");
  console.log(`  Load time: ${perfMetrics.getLoadTime().toFixed(2)} ms`);
  console.log(`  Generate duration: ${perfMetrics.getGenerateDuration().toFixed(2)} ms`);
  console.log(`  Transformer duration: ${perfMetrics.getTransformerInferDuration().mean.toFixed(2)} ms`);
  console.log(`  VAE decoder duration: ${perfMetrics.getVaeDecoderInferDuration().toFixed(2)} ms`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
