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
            "Decode and save the intermediate image at every denoising step",
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

    const rngSeed = 42;
    const numInferenceSteps = 20;
    const imageWidth = 512;
    const imageHeight = 512;

    const device = "CPU"; // GPU can be used as well
    const pipeline = await Text2ImagePipeline(argv.model_dir, device);

    // The callback decodes the current latent into an image at every denoising step.
    // decode() runs asynchronously, so awaiting it inside the callback keeps the event
    // loop responsive. The frames are saved after generation completes.
    const frames = [];
    async function callback(step, numSteps, latent) {
        frames.push(await pipeline.decode(latent));
        process.stdout.write(`Step ${step + 1}/${numSteps}\r`);
        return false;
    }

    await pipeline.generate(argv.prompt, {
        width: imageWidth,
        height: imageHeight,
        num_inference_steps: numInferenceSteps,
        num_images_per_prompt: 1,
        rng_seed: rngSeed,
        callback,
    });

    for (let step = 0; step < frames.length; step++) {
        await saveAsBMP(`denoising_step_${step}.bmp`, frames[step]);
    }
    console.log(`\nSaved ${frames.length} denoising step images.`);
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
