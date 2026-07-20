// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { basename } from "node:path";
import yargs from "yargs/yargs";
import { hideBin } from "yargs/helpers";
import { InpaintingPipeline } from "openvino-genai-node";
import { readImage, saveAsBMP } from "../image_utils.js";

async function main() {
    const argv = await yargs(hideBin(process.argv))
        .scriptName(basename(process.argv[1]))
        .command(
            "$0 <model_dir> <prompt> <image> <mask>",
            "Run Inpainting pipeline and save generated image as BMP file",
            (yargsBuilder) =>
                yargsBuilder
                    .positional("model_dir", {
                        type: "string",
                        describe: "Path to the converted image generation model directory",
                        demandOption: true,
                    })
                    .positional("prompt", {
                        type: "string",
                        describe: "Prompt that guides image inpainting",
                        demandOption: true,
                    })
                    .positional("image", {
                        type: "string",
                        describe: "Path to the input image (JPEG, PNG or BMP)",
                        demandOption: true,
                    })
                    .positional("mask", {
                        type: "string",
                        describe: "Path to the mask image marking the region to repaint (JPEG, PNG or BMP)",
                        demandOption: true,
                    })
        )
        .strict()
        .help()
        .parse();

    const device = "CPU"; // GPU can be used as well
    const pipeline = await InpaintingPipeline(argv.model_dir, device);

    const imageTensor = await readImage(argv.image, { batched: true });
    const maskTensor = await readImage(argv.mask, { batched: true });

    function callback(step, numSteps) {
        process.stdout.write(`Step ${step + 1}/${numSteps}\r`);
        return false;
    }

    const resultTensor = await pipeline.generate(
        argv.prompt,
        imageTensor,
        maskTensor,
        {
            callback,
        },
    );

    await saveAsBMP("image.bmp", resultTensor);
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
