// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import { basename } from "node:path";
import readline from "node:readline/promises";
import ffmpegPath from "ffmpeg-static";
import ffprobeModule from "ffprobe-static";
import { VLMPipeline } from "openvino-genai-node";
import { addon as ov } from "openvino-node";
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
 * Run a child process and collect stdout into buffer.
 * @param {string} command
 * @param {string[]} args
 * @returns {Promise<Buffer>}
 */
function spawnChild(command, args) {
    return new Promise((resolve, reject) => {
        const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
        let stdout = [];
        let stderr = "";
        child.stdout.on("data", (d) => stdout.push(d));
        child.stderr.on("data", (d) => (stderr += d.toString()));
        child.on("error", reject);
        child.on("close", (code) => {
            if (code === 0) {
                if (stderr) reject(new Error(stderr));
                else resolve(Buffer.concat(stdout));
            }
            else reject(new Error(`${command} exited with code ${code}: ${stderr}`));
        });
    });
}

/**
 * Read video metadata via ffprobe.
 * @param {string} videoPath
 * @returns {Promise<{width: number, height: number, totalFrames: number}>}
 */
async function getVideoInfo(videoPath) {
    const ffprobePath = ffprobeModule.path;
    if (!ffprobePath) {
        throw new Error("ffprobe-static binary not found. Ensure ffprobe-static is installed.");
    }

    const stdout = await spawnChild(ffprobePath, [
        "-v", "error", // Log level
        "-select_streams", "v:0", // Select first video stream
        "-count_frames", // Count frames in the selected stream
        "-show_entries", // Select metadata entries
        "stream=width,height,nb_read_frames,r_frame_rate,duration", // Requested entries
        "-of", "json", // Output format
        videoPath, // Input video path
    ]);

    /**
     * Convert r_frame_rate to frames per second.
     */
    const parseFps = (value) => {
        if (!value || typeof value !== "string") return null;
        const parts = value.split("/").map((v) => Number(v));
        if (parts.length === 2 && Number.isFinite(parts[0]) && Number.isFinite(parts[1]) && parts[1] !== 0) {
            return parts[0] / parts[1];
        }
        const asNumber = Number(value);
        return Number.isFinite(asNumber) ? asNumber : null;
    }

    const parsed = JSON.parse(stdout.toString("utf8"));
    const stream = parsed?.streams?.[0];
    const width = Number(stream?.width);
    const height = Number(stream?.height);
    const nbReadFrames = Number(stream?.nb_read_frames);
    const duration = Number(stream?.duration);
    const fps = parseFps(stream?.r_frame_rate);

    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
        throw new Error("Unable to read video resolution via ffprobe.");
    }

    let totalFrames = Number.isFinite(nbReadFrames) && nbReadFrames > 0 ? nbReadFrames : null;
    if (!totalFrames && Number.isFinite(duration) && duration > 0 && Number.isFinite(fps) && fps > 0) {
        totalFrames = Math.max(1, Math.round(duration * fps));
    }

    if (!totalFrames) {
        throw new Error("Unable to read total frame count via ffprobe.");
    }

    return { width, height, totalFrames };
}

/**
 * Uniformly sample frames and pack them into an ov.Tensor.
 * @param {string} videoPath
 * @param {number} [numFrames=8]
 * @returns {Promise<ov.Tensor>}
 */
async function readVideo(videoPath, numFrames = 8) {
    if (!ffmpegPath) {
        throw new Error("ffmpeg-static binary not found. Ensure ffmpeg-static is installed.");
    }

    const { width, height, totalFrames } = await getVideoInfo(videoPath);
    const channels = 3;
    const frameSize = width * height * channels;
    const indices = [];
    for (let v = 0; v < totalFrames; v += Math.max(1, totalFrames / numFrames)) {
        indices.push(Math.trunc(v));
    }
    const expr = indices.map((i) => `eq(n\\,${i})`).join("+");

    let stdout = await spawnChild(ffmpegPath, [
        "-v", "error", // Log level
        "-i", videoPath, // Input video path
        "-vf", `select='${expr}'`, // Filter frames by indices
        "-frames:v", String(numFrames), // Limit number of output frames
        "-f", "rawvideo", // Output format
        "-vsync", 0, // Prevent from duplicating frames
        "-pix_fmt", "bgr24", // Pixel format. bgr24 is used to align with python sample. Use rgb24 if needed.
        "-", // Output to stdout
    ]);

    const framesCount = stdout.length / frameSize;
    if (framesCount === 0) {
        throw new Error("ffmpeg extracted 0 frames.");
    }

    return new ov.Tensor("u8", [framesCount, height, width, channels], stdout);
}

/**
 * Read a single video or all videos from a directory.
 * @param {string} path
 * @returns {Promise<ov.Tensor[]>}
 */
async function readVideos(path) {
    const stat = await fs.stat(path);
    if (!stat.isDirectory()) {
        return [await readVideo(path)];
    }

    const entries = await fs.readdir(path);
    const files = entries.map((name) => `${path}/${name}`);

    files.sort((a, b) => a.localeCompare(b));

    const tensors = [];
    for (const file of files) tensors.push(await readVideo(file));

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
