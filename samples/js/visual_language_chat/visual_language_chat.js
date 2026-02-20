// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { promises as fs } from "node:fs";
import { basename } from "node:path";
import readline from "node:readline/promises";
import { VLMPipeline } from "openvino-genai-node";
import { addon as ov } from "openvino-node";
import sharp from "sharp";
import { hideBin } from "yargs/helpers";
import yargs from "yargs/yargs";

/**
 * Streams text chunks to stdout as they are generated.
 * @param {string} chunk - Text chunk produced by the model.
 * @returns {void}
 */
function streamer(chunk) {
    process.stdout.write(chunk);
}

/**
 * Reads an image from disk and converts it to an OpenVINO tensor.
 * @param {string} filePath - Path to the image file.
 * @returns {Promise<ov.Tensor>} Tensor in HWC layout with type u8.
 */
async function readImage(filePath) {
    try {
        const { data, info } = await sharp(filePath)
            .raw()
            .toBuffer({ resolveWithObject: true });

        return new ov.Tensor("u8", [info.height, info.width, 3], data);
    } catch (err) {
        throw new Error(`Failed to read image: ${filePath}. ${err.message}`);
    }
}

/**
 * Reads one image or all images from a directory and converts them to tensors.
 * @param {string} path - File path or directory path containing images.
 * @returns {Promise<ov.Tensor[]>} List of image tensors sorted by filename.
 */
async function readImages(path) {
    const stat = await fs.stat(path);
    if (!stat.isDirectory()) {
        return [await readImage(path)];
    }

    const entries = await fs.readdir(path);
    const files = entries.map((name) => `${path}/${name}`);

    files.sort((a, b) => a.localeCompare(b));

    const tensors = [];
    for (const file of files) {
        tensors.push(await readImage(file));
    }

    return tensors;
}

/**
 * Entry point for interactive image-to-text chat.
 * @returns {Promise<void>}
 */
async function main() {
    const argv = yargs(hideBin(process.argv))
        .scriptName(basename(process.argv[1]))
        .command(
            "$0 model_dir image_path [device]",
            "Run image-to-text chat",
            (yargsBuilder) => yargsBuilder
                .positional("model_dir", {
                    type: "string",
                    describe: "Path to the VLM model directory",
                    demandOption: true,
                })
                .positional("image_path", {
                    type: "string",
                    describe: "Path to an image file or a directory with images",
                    demandOption: true,
                })
                .positional("device", {
                    type: "string",
                    describe: "Device name (e.g., CPU, GPU)",
                    default: "CPU",
                }),
        )
        .strict()
        .help()
        .parse();

    const modelDir = argv.model_dir;
    const imagePathOrDir = argv.image_path;
    const device = argv.device;

    const images = await readImages(imagePathOrDir);

    const properties = {};
    if (device === "GPU") {
        // Cache compiled models on disk for GPU to save time on the next run.
        // It's not beneficial for CPU.
        properties.CACHE_DIR = "vlm_cache";
    }

    const pipe = await VLMPipeline(modelDir, device, properties);
    const generationConfig = { max_new_tokens: 100 };

    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

    await pipe.startChat();

    try {
        let prompt = await rl.question("question:\n");
        await pipe.generate(prompt, { generationConfig, streamer, images });

        while (true) {
            prompt = await rl.question("\n----------\nquestion:\n");
            await pipe.generate(prompt, { generationConfig, streamer });
        }
    } catch (error) {
        if (error.name === "AbortError") {
            console.log(error.message);
        } else {
            throw error;
        }
    } finally {
        rl.close();
        await pipe.finishChat();
    }
}

main();
