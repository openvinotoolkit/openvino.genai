// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { spawn } from "node:child_process";
import ffmpegPath from "ffmpeg-static";
import ffprobeModule from "ffprobe-static";
import { addon as ov } from "openvino-node";

/**
 * Run a child process and collect stdout into a buffer.
 * @param {string} command
 * @param {string[]} args
 * @returns {Promise<Buffer>}
 */
function spawnChild(command, args) {
    return new Promise((resolve, reject) => {
        const child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] });
        const stdout = [];
        let stderr = "";

        child.stdout.on("data", (chunk) => stdout.push(chunk));
        child.stderr.on("data", (chunk) => (stderr += chunk.toString()));
        child.on("error", reject);
        child.on("close", (code) => {
            if (code === 0) {
                resolve(Buffer.concat(stdout));
                return;
            }

            reject(new Error(`${command} exited with code ${code}: ${stderr}`));
        });
    });
}

/**
 * Read media stream info via ffprobe.
 * @param {string} mediaPath
 * @param {string} streamEntries
 * @returns {Promise<Record<string, unknown>>}
 */
async function probeMediaInfo(mediaPath, streamEntries) {
    const ffprobePath = ffprobeModule.path;
    if (!ffprobePath) {
        throw new Error("ffprobe-static binary not found. Ensure ffprobe-static is installed.");
    }

    const stdout = await spawnChild(ffprobePath, [
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", `stream=${streamEntries}`,
        "-of", "json",
        mediaPath,
    ]);

    const parsed = JSON.parse(stdout.toString("utf8"));
    return parsed?.streams?.[0] ?? {};
}

/**
 * Read image metadata via ffprobe.
 * @param {string} imagePath
 * @returns {Promise<{width: number, height: number}>}
 */
async function getImageInfo(imagePath) {
    const stream = await probeMediaInfo(imagePath, "width,height");
    const width = Number(stream?.width);
    const height = Number(stream?.height);

    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
        throw new Error("Unable to read image resolution via ffprobe.");
    }

    return { width, height };
}

/**
 * Read one image using ffmpeg and convert it to an OpenVINO tensor.
 * @param {string} imagePath
 * @returns {Promise<ov.Tensor>} Tensor in HWC layout with type u8.
 */
export async function readImageWithFfmpeg(imagePath) {
    if (!ffmpegPath) {
        throw new Error("ffmpeg-static binary not found. Ensure ffmpeg-static is installed.");
    }

    const { width, height } = await getImageInfo(imagePath);
    const channels = 3;
    const expectedSize = width * height * channels;

    const stdout = await spawnChild(ffmpegPath, [
        "-v", "error",
        "-i", imagePath,
        "-f", "rawvideo",
        "-vframes", "1",
        "-pix_fmt", "bgr24", // bgr24 is used to align with python sample. Use rgb24 as well.
        "-",
    ]);

    if (stdout.length < expectedSize) {
        throw new Error("ffmpeg returned incomplete image frame.");
    }

    return new ov.Tensor("u8", [height, width, channels], stdout.subarray(0, expectedSize));
}

/**
 * Read video metadata via ffprobe.
 * @param {string} videoPath
 * @returns {Promise<{width: number, height: number, totalFrames: number}>}
 */
async function getVideoInfo(videoPath) {
    const stream = await probeMediaInfo(videoPath, "width,height,nb_read_frames,r_frame_rate,duration");

    /**
     * Convert r_frame_rate to frames per second.
     * @param {unknown} value
     * @returns {number | null}
     */
    const parseFps = (value) => {
        if (!value || typeof value !== "string") return null;
        const parts = value.split("/").map((v) => Number(v));
        if (parts.length === 2 && Number.isFinite(parts[0]) && Number.isFinite(parts[1]) && parts[1] !== 0) {
            return parts[0] / parts[1];
        }

        const asNumber = Number(value);
        return Number.isFinite(asNumber) ? asNumber : null;
    };

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
 * Uniformly sample video frames and pack them into an OpenVINO tensor.
 * @param {string} videoPath
 * @param {number} [numFrames=8]
 * @returns {Promise<ov.Tensor>}
 */
export async function readVideoWithFfmpeg(videoPath, numFrames = 8) {
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

    const stdout = await spawnChild(ffmpegPath, [
        "-v", "error",
        "-i", videoPath,
        "-vf", `select='${expr}'`,
        "-frames:v", String(numFrames),
        "-f", "rawvideo",
        "-vsync", "0",
        "-pix_fmt", "bgr24",
        "-",
    ]);

    const framesCount = stdout.length / frameSize;
    if (framesCount === 0) {
        throw new Error("ffmpeg extracted 0 frames.");
    }

    return new ov.Tensor("u8", [framesCount, height, width, channels], stdout);
}
