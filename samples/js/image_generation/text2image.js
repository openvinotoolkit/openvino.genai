// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { basename } from "node:path";
import yargs from "yargs/yargs";
import { hideBin } from "yargs/helpers";
import { Text2ImagePipeline } from "openvino-genai-node";
import { saveAsBMP } from "../image_utils.js";

async function main() {
  const argv = await yargs(hideBin(process.argv))
    .scriptName(basename(process.argv[1]))
    .command(
      "$0 <model_dir> <prompt>",
      "Run Text2Image pipeline and save generated image as BMP file",
      (yargsBuilder) =>
        yargsBuilder
          .positional("model_dir", {
            type: "string",
            describe: "Path to the converted image generation model directory",
            demandOption: true,
          })
          .positional("prompt", {
            type: "string",
            describe: "Prompt to generate images from",
            demandOption: true,
          })
    )
    .strict()
    .help()
    .parse();

  const device = "CPU"; // GPU can be used as well
  const pipeline = await Text2ImagePipeline(argv.model_dir, device);

  function callback(step, numSteps) {
    process.stdout.write(`Step ${step + 1}/${numSteps}\r`);
    return false;
  }

  const imageTensor = await pipeline.generate(
    argv.prompt,
    {
      width: 512,
      height: 512,
      num_inference_steps: 20,
      num_images_per_prompt: 1,
      callback,
    },
  );

  await saveAsBMP("image.bmp", imageTensor);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
