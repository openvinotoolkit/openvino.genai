// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from "node:fs/promises";
import { basename, extname, join } from "node:path";
import readline from "node:readline/promises";
import { addon as ov } from "openvino-node";
import jpegJs from "jpeg-js";
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
 * Reads one image file and converts it to an OpenVINO tensor in HWC RGB layout.
 * Uses jpeg-js (JPEG) and pngjs (PNG) to produce pixel values equivalent to PIL Image.open().convert("RGB").
 * @param {string} filePath - Path to a .jpg/.jpeg or .png file.
 * @returns {Promise<ov.Tensor>} Tensor with shape [height, width, 3] and type u8.
 */
async function readImage(filePath) {
    const buffer = await fs.readFile(filePath);
    const ext = extname(filePath).toLowerCase();

    let width, height, rgbaData;
    if (ext === ".jpg" || ext === ".jpeg") {
        ({ width, height, data: rgbaData } = jpegJs.decode(buffer, { useTArray: true }));
    } else if (ext === ".png") {
        ({ width, height, data: rgbaData } = PNG.sync.read(buffer));
    } else {
        throw new Error(`Unsupported image format: ${ext}`);
    }

    const pixelCount = width * height;
    const rgb = new Uint8Array(pixelCount * 3);
    for (let i = 0; i < pixelCount; i++) {
        rgb[i * 3] = rgbaData[i * 4];
        rgb[i * 3 + 1] = rgbaData[i * 4 + 1];
        rgb[i * 3 + 2] = rgbaData[i * 4 + 2];
    }

    const tensor = new ov.Tensor("u8", [height, width, 3], rgb);
    tensor._buffer = rgb; // prevent GC from collecting the backing buffer
    return tensor;
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
    const entries = await fs.readdir(path, { withFileTypes: true });
    const files = entries
        .filter((entry) => entry.isFile() && supportedExtensions.has(extname(entry.name).toLowerCase()))
        .map((entry) => join(path, entry.name));

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

    /**
     * Reads the next user prompt or returns null when stdin is closed (EOF).
     * @param {string} promptText - Prompt to display to the user.
     * @returns {Promise<string|null>} User input or null on EOF.
     */
    async function readPrompt(promptText) {
        try {
            return await rl.question(promptText);
        } catch {
            return null;
        }
    }

    try {
        const chatHistory = new ChatHistory();

        let prompt = await readPrompt("question:\n");
        if (!prompt) {
            return;
        }
        chatHistory.push({ role: "user", content: prompt });
        let decodedResults = await pipe.generate(chatHistory, { generationConfig, streamer, images });
        process.stdout.write("\n");
        chatHistory.push({ role: "assistant", content: decodedResults.texts[0] });

        while (true) {
            prompt = await readPrompt("----------\nquestion:\n");
            if (!prompt) {
                break;
            }
            chatHistory.push({ role: "user", content: prompt });
            decodedResults = await pipe.generate(chatHistory, { generationConfig, streamer });
            process.stdout.write("\n");
            chatHistory.push({ role: "assistant", content: decodedResults.texts[0] });
        }
    } finally {
        rl.close();
    }
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});
