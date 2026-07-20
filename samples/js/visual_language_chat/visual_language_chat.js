// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { basename } from "node:path";
import readline from "node:readline/promises";
import { ChatHistory, VLMPipeline } from "openvino-genai-node";
import { hideBin } from "yargs/helpers";
import yargs from "yargs/yargs";
import { readImages } from "../image_utils.js";

/**
 * Streams text chunks to stdout as they are generated.
 * @param {string} chunk - Text chunk produced by the model.
 * @returns {void}
 */
function streamer(chunk) {
    process.stdout.write(chunk);
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
