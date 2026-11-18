// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { addon as ov } from "openvino-node";
import cv from "@u4/opencv4nodejs";

/**
 * Samples evenly spaced frame indices across a video.
 * @param {number} totalFrames - Number of decoded frames.
 * @param {number} numFrames - Maximum number of frames to reference.
 * @returns {number[]} Sorted list of sampled frame indices.
 */
function sampleFrameIndices(totalFrames, numFrames) {
    const count = Math.min(numFrames, totalFrames);
    if (count <= 1) {
        return [0];
    }
    const indices = [];
    for (let i = 0; i < count; i++) {
        indices.push(Math.round((i * (totalFrames - 1)) / (count - 1)));
    }
    return indices;
}

/**
 * Decodes a video file into an OpenVINO tensor and its metadata using @u4/opencv4nodejs.
 * @param {string} path - Path to the video file.
 * @param {number} numFrames - Number of frames to reference in metadata.
 * @returns {{ videoTensor: ov.Tensor, videoMetadata: { fps: number, frames_indices: number[] } }}
 */
export function readVideo(path, numFrames) {
    if (!Number.isInteger(numFrames) || numFrames <= 0) {
        throw new RangeError("numFrames must be a positive integer");
    }
    const capture = new cv.VideoCapture(path);
    const fps = capture.get(cv.CAP_PROP_FPS);

    const frames = [];
    let height = 0;
    let width = 0;
    for (let frame = capture.read(); !frame.empty; frame = capture.read()) {
        height = frame.rows;
        width = frame.cols;
        // OpenCV decodes frames as BGR; convert to the RGB layout the model expects.
        frames.push(frame.cvtColor(cv.COLOR_BGR2RGB).getData());
    }

    if (frames.length === 0) {
        throw new Error(`Failed to read frames from video: ${path}`);
    }

    const frameSize = height * width * 3;
    const data = new Uint8Array(frames.length * frameSize);
    frames.forEach((frameData, index) => data.set(frameData, index * frameSize));

    const videoTensor = new ov.Tensor("u8", [frames.length, height, width, 3], data);
    const videoMetadata = { fps, frames_indices: sampleFrameIndices(frames.length, numFrames) };
    return { videoTensor, videoMetadata };
}

/**
 * Saves a video tensor as an AVI file with MJPG-encoded frames using @u4/opencv4nodejs.
 * If the tensor holds several videos, each one is saved to its own file with a "_b<index>" suffix.
 * @param {string} filename - Path to the output video file.
 * @param {ov.Tensor} videoTensor - u8 tensor with shape [num_videos, num_frames, height, width, 3] in RGB.
 * @param {number} [fps=25] - Frame rate of the output video.
 */
export function saveVideo(filename, videoTensor, fps = 25) {
    const [batchSize, numFrames, height, width] = videoTensor.getShape();
    const frameSize = height * width * 3;
    const videoData = videoTensor.data;

    for (let b = 0; b < batchSize; b++) {
        let outputPath = filename;
        if (batchSize > 1) {
            const dot = filename.lastIndexOf(".");
            const base = dot === -1 ? filename : filename.slice(0, dot);
            const ext = dot === -1 ? "avi" : filename.slice(dot + 1);
            outputPath = `${base}_b${b}.${ext}`;
        }

        const fourcc = cv.VideoWriter.fourcc("MJPG");
        const writer = new cv.VideoWriter(outputPath, fourcc, fps, new cv.Size(width, height));

        for (let f = 0; f < numFrames; f++) {
            const offset = videoData.byteOffset + (b * numFrames + f) * frameSize;
            const rgbData = Buffer.from(videoData.buffer, offset, frameSize);
            const frame = new cv.Mat(rgbData, height, width, cv.CV_8UC3);
            // OpenCV expects frames in BGR order
            writer.write(frame.cvtColor(cv.COLOR_RGB2BGR));
        }

        writer.release();
        console.log(`Wrote ${outputPath} (${numFrames} frames, ${width}x${height} @ ${fps} fps)`);
    }
}
