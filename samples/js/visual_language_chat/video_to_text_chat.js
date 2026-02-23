// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import fs from "node:fs/promises";
import { basename } from "node:path";
import readline from "node:readline/promises";
import { readVideoWithFfmpeg } from "../ffmpeg_utils.js";
import { VLMPipeline } from "openvino-genai-node";
import yargs from "yargs/yargs";
import { hideBin } from "yargs/helpers";

/**
 * Stream model output to stdout.
 * @param {string} chunk
 */
function streamer(chunk) {
    process.stdout.write(chunk);
}

/**
 * Read a single video or all videos from a directory.
 * @param {string} path
 * @return List of video tensors sorted by filename.
 */
async function readVideos(path) {
    const stat = await fs.stat(path);
    if (!stat.isDirectory()) {
        return [await readVideoWithFfmpeg(path)];
    }

    const entries = await fs.readdir(path);
    const files = entries.map((name) => `${path}/${name}`);

    files.sort((a, b) => a.localeCompare(b));

    const tensors = [];
    for (const file of files) tensors.push(await readVideoWithFfmpeg(file));

    return tensors;
}

/**
 * Entry point.
 * @returns {Promise<void>}
 */
async function main() {
    const argv = yargs(hideBin(process.argv))
        .scriptName(basename(process.argv[1]))
        .command(
            "$0 model_dir video_path [device]",
            "Run video-to-text chat",
            (yargsBuilder) => yargsBuilder
                .positional("model_dir", {
                    type: "string",
                    describe: "Path to the VLM model directory",
                    demandOption: true,
                })
                .positional("video_path", {
                    type: "string",
                    describe: "Path to a video file or a directory with videos",
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
    const videoPath = argv.video_path;
    const device = argv.device;

    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

    const properties = {};
    if (device === "GPU") {
        properties.CACHE_DIR = "vlm_cache";
    }

    const videos = await readVideos(videoPath);

    const pipe = await VLMPipeline(modelDir, device, properties);
    const generationConfig = { max_new_tokens: 100 };

    await pipe.startChat();

    try {
        let prompt = await rl.question("question:\n");
        await pipe.generate(prompt, {
            generationConfig,
            videos,
            streamer,
        });

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
