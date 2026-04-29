// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { promises as fs } from "node:fs";
import { basename } from "node:path";
import readline from "node:readline/promises";
import { addon as ov } from "openvino-node";
import jpeg from "jpeg-js";
import { PNG } from "pngjs";
import { ChatHistory, VLMPipeline } from "openvino-genai-node";
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
 * Converts an RGBA pixel buffer to an RGB Uint8Array.
 * @param {Uint8Array} rgba - Source RGBA pixel data.
 * @returns {Uint8Array} RGB pixel data.
 */
function rgbaToRgb(rgba) {
    const rgb = new Uint8Array((rgba.length / 4) * 3);
    for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
        rgb[j] = rgba[i];
        rgb[j + 1] = rgba[i + 1];
        rgb[j + 2] = rgba[i + 2];
    }
    return rgb;
}

/**
 * Reads one image file and converts it to an OpenVINO tensor in HWC RGB layout.
 * @param {string} filePath - Path to a .jpg/.jpeg or .png file.
 * @returns {Promise<ov.Tensor>} Tensor with shape [height, width, 3] and type u8.
 */
async function readImage(filePath) {
    const buf = await fs.readFile(filePath);
    const lower = filePath.toLowerCase();

    if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) {
        const { width, height, data } = jpeg.decode(buf);
        const rgb = rgbaToRgb(data);
        const tensor = new ov.Tensor("u8", [height, width, 3], rgb);
        tensor._buffer = rgb; // prevent GC from collecting the backing buffer
        return tensor;
    }

    if (lower.endsWith(".png")) {
        const { width, height, data } = PNG.sync.read(buf); // RGBA
        const rgb = rgbaToRgb(data);
        const tensor = new ov.Tensor("u8", [height, width, 3], rgb);
        tensor._buffer = rgb; // prevent GC from collecting the backing buffer
        return tensor;
    }

    throw new Error(`Unsupported image format: ${filePath}. Supported formats: .jpg, .jpeg, .png`);
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

    const supportedExtensions = new Set([".jpg", ".jpeg", ".png"]);
    const entries = await fs.readdir(path);
    const files = entries
        .filter((name) => supportedExtensions.has(name.slice(name.lastIndexOf(".")).toLowerCase()))
        .map((name) => `${path}/${name}`);

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
            "$0 model_dir image_path [device] [prompt_lookup]",
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
                })
                .positional("prompt_lookup", {
                    type: "boolean",
                    describe: "Enable prompt lookup decoding (true/false)",
                    default: false,
                }),
        )
        .strict()
        .help()
        .parse();

    const {
        model_dir: modelDir,
        image_path: imagePathOrDir,
        device,
        prompt_lookup: promptLookup,
    } = argv;

    const images = await readImages(imagePathOrDir);

    const properties = { prompt_lookup: promptLookup };
    if (device === "GPU") {
        // Cache compiled models on disk for GPU to save time on the next run.
        // It's not beneficial for CPU.
        properties.CACHE_DIR = "vlm_cache";
    }

    const pipe = await VLMPipeline(modelDir, device, properties);
    const generationConfig = { max_new_tokens: 100 };
    if (promptLookup) {
        // Prompt lookup decoding generates candidate tokens from the input prompt
        // and verifies them in a single forward pass, speeding up generation when
        // the output repeats parts of the input (e.g., document Q&A).
        generationConfig.num_assistant_tokens = 5;
        generationConfig.max_ngram_size = 3;
    }

    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

    try {
        const chatHistory = new ChatHistory();

        let prompt = await rl.question("question:\n");
        chatHistory.push({ role: "user", content: prompt });
        const decodedResults = await pipe.generate(chatHistory, { generationConfig, streamer, images });
        process.stdout.write("\n");
        chatHistory.push({ role: "assistant", content: decodedResults.texts[0] });

        while (true) {
            prompt = await rl.question("----------\nquestion:\n");
            chatHistory.push({ role: "user", content: prompt });
            const decodedResults = await pipe.generate(chatHistory, { generationConfig, streamer });
            process.stdout.write("\n");
            chatHistory.push({ role: "assistant", content: decodedResults.texts[0] });
        }
    } catch (error) {
        console.log(error.message);
    } finally {
        rl.close();
    }
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});
