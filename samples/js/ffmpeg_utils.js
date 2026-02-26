// Copyright (C) 2023-2026 Intel Corporation
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
 * Read audio metadata via ffprobe.
 * @param {string} audioPath
 * @returns {Promise<{ sampleRate: number, channels: number, durationSec: number }>}
 */
async function getAudioInfo(audioPath) {
    const ffprobePath = ffprobeModule.path;
    if (!ffprobePath) {
        throw new Error("ffprobe-static binary not found. Ensure ffprobe-static is installed.");
    }

    const stdout = await spawnChild(ffprobePath, [
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,duration:format=duration",
        "-of", "json",
        audioPath,
    ]);

    const parsed = JSON.parse(stdout.toString("utf8"));
    const stream = parsed?.streams?.[0] ?? {};

    const sampleRate = Number(stream?.sample_rate);
    const channels = Number(stream?.channels);
    const streamDuration = Number(stream?.duration);
    const formatDuration = Number(parsed?.format?.duration);
    const durationSec = Number.isFinite(streamDuration) && streamDuration > 0 ? streamDuration : formatDuration;

    if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
        throw new Error("Unable to read audio sample rate via ffprobe.");
    }

    if (!Number.isFinite(channels) || channels <= 0) {
        throw new Error("Unable to read audio channels count via ffprobe.");
    }

    if (!Number.isFinite(durationSec) || durationSec <= 0) {
        throw new Error("Unable to read audio duration via ffprobe.");
    }

    return { sampleRate, channels, durationSec };
}

/**
 * Read audio file and convert it into a Float32Array.
 * Decodes via ffmpeg to Float32 using audio stream parameters from metadata.
 * @param {string} audioPath
 * @returns {Promise<Float32Array>}
 */
export async function readAudio(audioPath) {
    if (!ffmpegPath) {
        throw new Error("ffmpeg-static binary not found. Ensure ffmpeg-static is installed.");
    }

    const { sampleRate, channels, durationSec } = await getAudioInfo(audioPath);

    const ffmpegArgs = [
        "-v", "error",
        "-i", audioPath,
        "-ar", String(sampleRate),
        "-ac", String(channels),
        "-f", "f32le",
        "-",
    ];

    const stdout = await spawnChild(ffmpegPath, ffmpegArgs);

    const bytesPerSample = 4; // 4 bytes for float32
    if (stdout.length === 0) {
        throw new Error("ffmpeg returned empty audio stream.");
    }

    if (stdout.length % bytesPerSample !== 0) {
        throw new Error("ffmpeg returned invalid f32le audio payload size.");
    }

    const totalValues = stdout.length / bytesPerSample;
    const samples = new Float32Array(stdout.buffer, stdout.byteOffset, totalValues);

    if (samples.length === 0) {
        throw new Error("Decoded audio contains 0 samples.");
    }

    if (totalValues % channels !== 0) {
        throw new Error("Decoded audio payload is not aligned with channel count.");
    }

    const numSamples = totalValues / channels;
    const expectedSamples = Math.max(1, Math.round(durationSec * sampleRate));
    const tolerance = Math.max(Math.round(sampleRate * 0.5), Math.round(expectedSamples * 0.05));
    if (numSamples + tolerance < expectedSamples) {
        throw new Error("Decoded audio is shorter than expected from ffprobe duration.");
    }

    return samples;
}
