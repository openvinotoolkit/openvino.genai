// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { writeFile } from "node:fs/promises";
import { basename } from "node:path";
import bmp from "bmp-js";
import yargs from "yargs/yargs";
import { hideBin } from "yargs/helpers";
import { Text2ImagePipeline } from "openvino-genai-node";

function toABGR(tensor) {
  const [_, height, width, channels] = tensor.getShape();
  if (channels !== 3) {
    throw new Error(`Expected RGB image tensor, got ${channels} channels.`);
  }

  const rgb = tensor.data instanceof Uint8Array ? tensor.data : Uint8Array.from(tensor.data);
  const rgba = Buffer.allocUnsafe(width * height * 4);

  for (let src = 0, dst = 0; src < rgb.length; src += 3, dst += 4) {
    rgba[dst] = 255;              // A
    rgba[dst + 1] = rgb[src + 2]; // B
    rgba[dst + 2] = rgb[src + 1]; // G
    rgba[dst + 3] = rgb[src];     // R
  }

  return { height, width, rgba };
}

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

  const { height, width, rgba } = toABGR(imageTensor);
  const bmpData = bmp.encode({ width, height, data: rgba });
  await writeFile("image.bmp", bmpData.data);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
